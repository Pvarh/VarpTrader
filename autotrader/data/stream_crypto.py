"""Real-time crypto data streaming via CCXT WebSocket (ccxt.pro).

When ``ccxt.pro`` is installed the module uses native WebSocket streaming.
If only the standard ``ccxt`` package is available it falls back to a
polling loop that calls REST endpoints at a configurable interval.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from data.normalizer import DataNormalizer
from journal.models import OHLCV

# ---------------------------------------------------------------------------
# Detect ccxt.pro availability
# ---------------------------------------------------------------------------
_CCXT_PRO_AVAILABLE: bool = False

try:
    import ccxt.pro as ccxtpro  # type: ignore[import-untyped]

    _CCXT_PRO_AVAILABLE = True
    logger.info("ccxt_pro_available")
except ImportError:
    logger.warning(
        "ccxt_pro_not_installed | detail={detail}",
        detail=(
            "ccxt.pro (ccxt[pro]) is not installed. "
            "Falling back to REST polling. Install with: pip install ccxt[pro]"
        ),
    )

try:
    import ccxt  # standard synchronous ccxt, used as fallback
except ImportError:
    ccxt = None  # type: ignore[assignment]
    logger.error(
        "ccxt_not_installed | detail={detail}",
        detail="Neither ccxt nor ccxt.pro is available.",
    )


class CryptoStream:
    """CCXT Pro WebSocket stream for real-time crypto data.

    When ``ccxt.pro`` is available the class uses its ``watch_ohlcv``,
    ``watch_ticker``, and ``watch_order_book`` methods which maintain a
    persistent WebSocket under the hood.

    When ``ccxt.pro`` is **not** installed the class degrades gracefully
    to a REST-based polling loop using standard ``ccxt``.  A warning is
    logged once at startup so the operator knows streaming quality is
    reduced.

    Usage::

        stream = CryptoStream(exchange_id="binance")
        stream.on_candle(my_candle_handler)
        stream.on_price(my_price_handler)
        await stream.connect()
        await stream.run_forever(["BTC/USDT", "ETH/USDT"])
    """

    # Polling interval when ccxt.pro is unavailable
    _POLL_INTERVAL_SECONDS: float = 5.0

    # Reconnection parameters
    _RECONNECT_BASE_DELAY: float = 1.0
    _RECONNECT_MAX_DELAY: float = 60.0
    _RECONNECT_BACKOFF_FACTOR: float = 2.0

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str | None = None,
        secret: str | None = None,
    ) -> None:
        """Initialize CCXT Pro exchange for WebSocket streaming.

        Parameters
        ----------
        exchange_id:
            CCXT exchange identifier (``"binance"``, ``"bybit"``, etc.).
        api_key:
            Optional API key for authenticated endpoints.
        secret:
            Optional API secret for authenticated endpoints.

        Note
        ----
        Uses ``ccxt.pro`` (async) if available, falls back to polling
        with the synchronous ``ccxt`` library.
        """
        self._exchange_id: str = exchange_id
        self._api_key: str | None = api_key
        self._secret: str | None = secret
        self._exchange: Any = None
        self._running: bool = False
        self._use_pro: bool = _CCXT_PRO_AVAILABLE
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._latest_prices: dict[str, float] = {}
        self._latest_orderbooks: dict[str, dict[str, Any]] = {}

        logger.info(
            "crypto_stream_initialised | exchange={exchange} mode={mode}",
            exchange=exchange_id,
            mode="websocket" if self._use_pro else "polling",
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Initialize the async exchange connection.

        Creates the appropriate exchange instance based on whether
        ``ccxt.pro`` is available.

        Raises
        ------
        RuntimeError
            When neither ``ccxt.pro`` nor ``ccxt`` is installed, or when
            the exchange identifier is invalid.
        """
        config: dict[str, Any] = {"enableRateLimit": True}
        if self._api_key is not None:
            config["apiKey"] = self._api_key
        if self._secret is not None:
            config["secret"] = self._secret

        if self._use_pro:
            exchange_class = getattr(ccxtpro, self._exchange_id, None)
            if exchange_class is None:
                logger.error(
                    "unknown_pro_exchange | exchange_id={exchange_id}",
                    exchange_id=self._exchange_id,
                )
                raise RuntimeError(
                    f"Exchange '{self._exchange_id}' is not available in ccxt.pro"
                )
            self._exchange = exchange_class(config)
            logger.info("pro_exchange_created | exchange={exchange}", exchange=self._exchange_id)
        elif ccxt is not None:
            exchange_class = getattr(ccxt, self._exchange_id, None)
            if exchange_class is None:
                logger.error(
                    "unknown_exchange | exchange_id={exchange_id}",
                    exchange_id=self._exchange_id,
                )
                raise RuntimeError(
                    f"Exchange '{self._exchange_id}' is not available in ccxt"
                )
            self._exchange = exchange_class(config)
            logger.info(
                "fallback_exchange_created | exchange={exchange} mode=polling",
                exchange=self._exchange_id,
            )
        else:
            raise RuntimeError(
                "Neither ccxt.pro nor ccxt is installed. "
                "Install with: pip install ccxt  or  pip install ccxt[pro]"
            )

        self._running = True

    # ------------------------------------------------------------------
    # WebSocket watchers (ccxt.pro)
    # ------------------------------------------------------------------
    async def watch_ohlcv(
        self,
        symbols: list[str],
        timeframe: str = "1m",
    ) -> None:
        """Stream OHLCV candles for given symbols.

        Uses ``ccxt.pro``'s ``watch_ohlcv`` when available; otherwise
        falls back to ``fetch_ohlcv`` polling.

        Parameters
        ----------
        symbols:
            List of trading pairs (e.g. ``["BTC/USDT", "ETH/USDT"]``).
        timeframe:
            Candle timeframe (``"1m"``, ``"5m"``, ``"1h"``, etc.).
        """
        if not self._exchange:
            logger.error("watch_ohlcv_no_connection")
            return

        if self._use_pro:
            await self._watch_ohlcv_pro(symbols, timeframe)
        else:
            await self._poll_ohlcv(symbols, timeframe)

    async def _watch_ohlcv_pro(
        self,
        symbols: list[str],
        timeframe: str,
    ) -> None:
        """Stream OHLCV via ccxt.pro WebSocket.

        Runs concurrent watchers for each symbol.  Each watcher calls
        ``watch_ohlcv`` in a loop and dispatches candle callbacks.

        Parameters
        ----------
        symbols:
            Trading pairs to stream.
        timeframe:
            Candle timeframe.
        """
        async def _watch_single(symbol: str) -> None:
            while self._running:
                try:
                    ohlcv_list = await self._exchange.watch_ohlcv(symbol, timeframe)
                    for raw_candle in ohlcv_list:
                        bar = DataNormalizer.from_ccxt(raw_candle, symbol, timeframe)
                        self._latest_prices[symbol.upper()] = bar.close

                        for cb in self._callbacks.get("candle", []):
                            if asyncio.iscoroutinefunction(cb):
                                await cb(symbol, bar)
                            else:
                                cb(symbol, bar)

                    logger.debug(
                        "ohlcv_received | symbol={symbol} timeframe={timeframe} count={count}",
                        symbol=symbol,
                        timeframe=timeframe,
                        count=len(ohlcv_list),
                    )

                except Exception:
                    if not self._running:
                        break
                    logger.exception(
                        "watch_ohlcv_error | symbol={symbol} timeframe={timeframe}",
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                    await asyncio.sleep(self._RECONNECT_BASE_DELAY)

        tasks = [asyncio.create_task(_watch_single(s)) for s in symbols]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()

    async def _poll_ohlcv(
        self,
        symbols: list[str],
        timeframe: str,
    ) -> None:
        """Fallback REST polling for OHLCV when ccxt.pro is unavailable.

        Periodically calls ``fetch_ohlcv`` and delivers only the latest
        candle to callbacks.

        Parameters
        ----------
        symbols:
            Trading pairs to poll.
        timeframe:
            Candle timeframe.
        """
        last_timestamps: dict[str, int] = {}

        while self._running:
            for symbol in symbols:
                try:
                    raw_bars = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda s=symbol: self._exchange.fetch_ohlcv(
                            s, timeframe=timeframe, limit=5
                        ),
                    )
                    if not raw_bars:
                        continue

                    latest_raw = raw_bars[-1]
                    candle_ts = int(latest_raw[0])

                    # Only dispatch if this is a new candle
                    if candle_ts != last_timestamps.get(symbol):
                        last_timestamps[symbol] = candle_ts
                        bar = DataNormalizer.from_ccxt(latest_raw, symbol, timeframe)
                        self._latest_prices[symbol.upper()] = bar.close

                        for cb in self._callbacks.get("candle", []):
                            if asyncio.iscoroutinefunction(cb):
                                await cb(symbol, bar)
                            else:
                                cb(symbol, bar)

                        logger.debug(
                            "polled_ohlcv | timeframe={timeframe} mode=polling symbol={symbol} close={close}",
                            timeframe=timeframe,
                            symbol=symbol,
                            close=bar.close,
                        )

                except Exception:
                    logger.exception(
                        "poll_ohlcv_error | timeframe={timeframe} mode=polling symbol={symbol}",
                        timeframe=timeframe,
                        symbol=symbol,
                    )

            await asyncio.sleep(self._POLL_INTERVAL_SECONDS)

    async def watch_ticker(self, symbols: list[str]) -> None:
        """Stream ticker price updates.

        Uses ``ccxt.pro``'s ``watch_ticker`` when available; otherwise
        falls back to ``fetch_ticker`` polling.

        Parameters
        ----------
        symbols:
            List of trading pairs (e.g. ``["BTC/USDT"]``).
        """
        if not self._exchange:
            logger.error("watch_ticker_no_connection")
            return

        if self._use_pro:
            await self._watch_ticker_pro(symbols)
        else:
            await self._poll_ticker(symbols)

    async def _watch_ticker_pro(self, symbols: list[str]) -> None:
        """Stream tickers via ccxt.pro WebSocket.

        Runs concurrent watchers for each symbol.

        Parameters
        ----------
        symbols:
            Trading pairs to stream.
        """
        async def _watch_single(symbol: str) -> None:
            while self._running:
                try:
                    ticker = await self._exchange.watch_ticker(symbol)
                    price = float(ticker.get("last", 0.0))
                    if price > 0:
                        self._latest_prices[symbol.upper()] = price

                        for cb in self._callbacks.get("price", []):
                            if asyncio.iscoroutinefunction(cb):
                                await cb(symbol, price)
                            else:
                                cb(symbol, price)

                        logger.debug(
                            "ticker_received | symbol={symbol} price={price}",
                            symbol=symbol,
                            price=price,
                        )

                except Exception:
                    if not self._running:
                        break
                    logger.exception("watch_ticker_error | symbol={symbol}", symbol=symbol)
                    await asyncio.sleep(self._RECONNECT_BASE_DELAY)

        tasks = [asyncio.create_task(_watch_single(s)) for s in symbols]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()

    async def _poll_ticker(self, symbols: list[str]) -> None:
        """Fallback REST polling for tickers.

        Parameters
        ----------
        symbols:
            Trading pairs to poll.
        """
        while self._running:
            for symbol in symbols:
                try:
                    ticker = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda s=symbol: self._exchange.fetch_ticker(s),
                    )
                    price = float(ticker.get("last", 0.0))
                    if price > 0:
                        self._latest_prices[symbol.upper()] = price

                        for cb in self._callbacks.get("price", []):
                            if asyncio.iscoroutinefunction(cb):
                                await cb(symbol, price)
                            else:
                                cb(symbol, price)

                        logger.debug(
                            "polled_ticker | mode=polling symbol={symbol} price={price}",
                            symbol=symbol,
                            price=price,
                        )

                except Exception:
                    logger.exception(
                        "poll_ticker_error | mode=polling symbol={symbol}",
                        symbol=symbol,
                    )

            await asyncio.sleep(self._POLL_INTERVAL_SECONDS)

    async def watch_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> None:
        """Stream order book updates for a symbol.

        Uses ``ccxt.pro``'s ``watch_order_book`` when available; otherwise
        falls back to ``fetch_order_book`` polling.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).
        limit:
            Number of price levels on each side.
        """
        if not self._exchange:
            logger.error("watch_order_book_no_connection")
            return

        if self._use_pro:
            await self._watch_orderbook_pro(symbol, limit)
        else:
            await self._poll_orderbook(symbol, limit)

    async def _watch_orderbook_pro(self, symbol: str, limit: int) -> None:
        """Stream order book via ccxt.pro WebSocket.

        Parameters
        ----------
        symbol:
            Trading pair to stream.
        limit:
            Depth levels per side.
        """
        while self._running:
            try:
                orderbook = await self._exchange.watch_order_book(symbol, limit)

                bids = orderbook.get("bids", [])
                asks = orderbook.get("asks", [])
                best_bid = float(bids[0][0]) if bids else 0.0
                best_ask = float(asks[0][0]) if asks else 0.0
                spread = best_ask - best_bid
                spread_pct = (spread / best_ask * 100.0) if best_ask > 0 else 0.0

                snapshot: dict[str, Any] = {
                    "bids": bids[:limit],
                    "asks": asks[:limit],
                    "symbol": symbol,
                    "timestamp": orderbook.get("timestamp"),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "spread_pct": round(spread_pct, 4),
                }

                self._latest_orderbooks[symbol.upper()] = snapshot

                for cb in self._callbacks.get("orderbook", []):
                    if asyncio.iscoroutinefunction(cb):
                        await cb(symbol, snapshot)
                    else:
                        cb(symbol, snapshot)

                logger.debug(
                    "orderbook_received | symbol={symbol} limit={limit} best_bid={best_bid} best_ask={best_ask} spread={spread}",
                    symbol=symbol,
                    limit=limit,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    spread=spread,
                )

            except Exception:
                if not self._running:
                    break
                logger.exception(
                    "watch_orderbook_error | symbol={symbol} limit={limit}",
                    symbol=symbol,
                    limit=limit,
                )
                await asyncio.sleep(self._RECONNECT_BASE_DELAY)

    async def _poll_orderbook(self, symbol: str, limit: int) -> None:
        """Fallback REST polling for order books.

        Parameters
        ----------
        symbol:
            Trading pair to poll.
        limit:
            Depth levels per side.
        """
        while self._running:
            try:
                orderbook = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._exchange.fetch_order_book(symbol, limit=limit),
                )

                bids = orderbook.get("bids", [])
                asks = orderbook.get("asks", [])
                best_bid = float(bids[0][0]) if bids else 0.0
                best_ask = float(asks[0][0]) if asks else 0.0
                spread = best_ask - best_bid
                spread_pct = (spread / best_ask * 100.0) if best_ask > 0 else 0.0

                snapshot: dict[str, Any] = {
                    "bids": bids[:limit],
                    "asks": asks[:limit],
                    "symbol": symbol,
                    "timestamp": orderbook.get("timestamp"),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "spread_pct": round(spread_pct, 4),
                }

                self._latest_orderbooks[symbol.upper()] = snapshot

                for cb in self._callbacks.get("orderbook", []):
                    if asyncio.iscoroutinefunction(cb):
                        await cb(symbol, snapshot)
                    else:
                        cb(symbol, snapshot)

                logger.debug(
                    "polled_orderbook | symbol={symbol} limit={limit} mode=polling best_bid={best_bid} best_ask={best_ask}",
                    symbol=symbol,
                    limit=limit,
                    best_bid=best_bid,
                    best_ask=best_ask,
                )

            except Exception:
                logger.exception(
                    "poll_orderbook_error | symbol={symbol} limit={limit} mode=polling",
                    symbol=symbol,
                    limit=limit,
                )

            await asyncio.sleep(self._POLL_INTERVAL_SECONDS)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------
    def on_candle(self, callback: Callable) -> None:
        """Register callback for candle events.

        Parameters
        ----------
        callback:
            Callable receiving ``(symbol: str, bar: OHLCV)``.
        """
        self._callbacks["candle"].append(callback)
        logger.debug("callback_registered | event=candle")

    def on_price(self, callback: Callable) -> None:
        """Register callback for price updates.

        Parameters
        ----------
        callback:
            Callable receiving ``(symbol: str, price: float)``.
        """
        self._callbacks["price"].append(callback)
        logger.debug("callback_registered | event=price")

    def on_orderbook(self, callback: Callable) -> None:
        """Register callback for orderbook snapshots.

        Parameters
        ----------
        callback:
            Callable receiving ``(symbol: str, orderbook: dict)``.
        """
        self._callbacks["orderbook"].append(callback)
        logger.debug("callback_registered | event=orderbook")

    # ------------------------------------------------------------------
    # Price cache
    # ------------------------------------------------------------------
    def get_latest_price(self, symbol: str) -> float:
        """Get most recent cached price.

        The price is updated by incoming ticker or candle data.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).

        Returns
        -------
        float
            Latest cached price.  Returns ``0.0`` when no price has been
            received yet.
        """
        return self._latest_prices.get(symbol.upper(), 0.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def disconnect(self) -> None:
        """Close exchange connection.

        Sets ``_running`` to ``False`` and closes the underlying exchange
        connection.  For ``ccxt.pro`` exchanges this properly tears down
        the WebSocket; for standard ``ccxt`` it is a no-op.
        """
        self._running = False

        if self._exchange is not None:
            try:
                if self._use_pro and hasattr(self._exchange, "close"):
                    await self._exchange.close()
                    logger.info(
                        "pro_exchange_closed | exchange={exchange}",
                        exchange=self._exchange_id,
                    )
                else:
                    logger.info(
                        "exchange_connection_closed | exchange={exchange}",
                        exchange=self._exchange_id,
                    )
            except Exception:
                logger.exception("exchange_close_error")
            finally:
                self._exchange = None

    async def run_forever(
        self,
        symbols: list[str],
        timeframe: str = "1m",
    ) -> None:
        """Main loop: watch tickers and OHLCV with auto-reconnect.

        Starts concurrent tasks for ticker watching and OHLCV streaming.
        If a connection drops, the loop reconnects with exponential
        backoff.  Exits only when ``disconnect()`` is called.

        Parameters
        ----------
        symbols:
            List of trading pairs (e.g. ``["BTC/USDT", "ETH/USDT"]``).
        timeframe:
            Candle timeframe (``"1m"``, ``"5m"``, ``"1h"``, etc.).
        """
        delay = self._RECONNECT_BASE_DELAY

        while True:
            try:
                if not self._running:
                    await self.connect()

                logger.info(
                    "stream_run_forever_started | symbols={symbols} timeframe={timeframe} mode={mode}",
                    symbols=symbols,
                    timeframe=timeframe,
                    mode="websocket" if self._use_pro else "polling",
                )

                # Run ticker and OHLCV watchers concurrently
                ticker_task = asyncio.create_task(self.watch_ticker(symbols))
                ohlcv_task = asyncio.create_task(self.watch_ohlcv(symbols, timeframe))

                done, pending = await asyncio.wait(
                    [ticker_task, ohlcv_task],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check for exceptions in completed tasks
                for task in done:
                    if task.exception() is not None:
                        raise task.exception()  # type: ignore[misc]

                # If we reach here without exception, we were stopped gracefully
                if not self._running:
                    logger.info("stream_stopped_gracefully")
                    break

                # Reset delay on clean exit
                delay = self._RECONNECT_BASE_DELAY

            except asyncio.CancelledError:
                logger.info("stream_cancelled")
                break
            except Exception:
                logger.exception("stream_error")

            # Reconnect with backoff
            if not self._running:
                break

            logger.info("stream_reconnecting | delay_seconds={delay_seconds}", delay_seconds=delay)
            await asyncio.sleep(delay)
            delay = min(
                delay * self._RECONNECT_BACKOFF_FACTOR,
                self._RECONNECT_MAX_DELAY,
            )

            # Reset for reconnection
            if self._exchange is not None:
                try:
                    if self._use_pro and hasattr(self._exchange, "close"):
                        await self._exchange.close()
                except Exception:
                    logger.exception("exchange_close_before_reconnect_error")
            self._exchange = None
            self._running = False

            try:
                await self.connect()
                delay = self._RECONNECT_BASE_DELAY
                logger.info(
                    "stream_reconnected | exchange={exchange}",
                    exchange=self._exchange_id,
                )
            except RuntimeError:
                logger.warning(
                    "stream_reconnect_failed | exchange={exchange} next_delay={next_delay}",
                    exchange=self._exchange_id,
                    next_delay=delay,
                )
