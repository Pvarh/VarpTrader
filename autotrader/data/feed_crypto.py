"""Crypto market data feed powered by CCXT (REST) + Binance WebSocket (live)."""

from __future__ import annotations

from typing import Any

import ccxt
from loguru import logger

from data.normalizer import DataNormalizer
from journal.models import OHLCV


class CryptoFeed:
    """Fetches crypto market data via CCXT (Binance by default).

    Any exchange supported by CCXT can be used by passing its identifier
    (e.g. ``"binance"``, ``"bybit"``, ``"coinbasepro"``).  When API
    credentials are supplied the feed can access authenticated endpoints,
    but all public methods here only hit unauthenticated (public) routes.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str | None = None,
        secret: str | None = None,
    ) -> None:
        """Initialize CCXT exchange connection.

        Parameters
        ----------
        exchange_id:
            CCXT exchange identifier (``"binance"``, ``"bybit"``, etc.).
        api_key:
            Optional API key for authenticated endpoints.
        secret:
            Optional API secret for authenticated endpoints.

        Raises
        ------
        ccxt.ExchangeNotAvailable
            If *exchange_id* is not recognised by CCXT.
        """
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            logger.error("unknown_exchange | exchange_id={exchange_id}", exchange_id=exchange_id)
            raise ccxt.ExchangeNotAvailable(
                f"Exchange '{exchange_id}' is not available in CCXT"
            )

        config: dict[str, Any] = {"enableRateLimit": True}
        if api_key is not None:
            config["apiKey"] = api_key
        if secret is not None:
            config["secret"] = secret

        self._exchange: ccxt.Exchange = exchange_class(config)
        self._exchange_id: str = exchange_id
        self._ws: Any = None   # BinanceWebSocket, set by start_stream()

        logger.info(
            "crypto_feed_initialised | exchange={exchange} authenticated={authenticated}",
            exchange=exchange_id,
            authenticated=api_key is not None,
        )

    # ------------------------------------------------------------------
    # WebSocket streaming
    # ------------------------------------------------------------------
    def start_stream(self, symbols: list[str], timeframe: str = "5m") -> None:
        """Start a native Binance WebSocket for real-time prices and candles.

        Prices received via the WebSocket are cached and returned by
        :meth:`get_latest_price` without any REST round-trip.
        Closed kline candles are accumulated in a rolling buffer accessible
        via the underlying :class:`~data.binance_ws.BinanceWebSocket`.

        Parameters
        ----------
        symbols:
            Trading pairs e.g. ``["BTC/USDT", "ETH/USDT"]``.
        timeframe:
            Kline timeframe to subscribe to (default ``"5m"``).
        """
        from data.binance_ws import BinanceWebSocket

        self._ws = BinanceWebSocket()
        self._ws.start(symbols, timeframe=timeframe)
        logger.info(
            "crypto_feed_ws_started | symbols={} timeframe={}", symbols, timeframe
        )

    # ------------------------------------------------------------------
    # Historical bars
    # ------------------------------------------------------------------
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 200,
    ) -> list[OHLCV]:
        """Fetch historical OHLCV bars.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).
        timeframe:
            Bar size accepted by the exchange (``"1m"``, ``"5m"``, ``"1h"``,
            ``"1d"``, etc.).
        limit:
            Maximum number of bars to return.

        Returns
        -------
        list[OHLCV]
            Chronologically ordered normalised bars.  Returns an empty list
            when no data is available or on error.
        """
        try:
            raw_bars: list[list[float | int]] = self._exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
            )
            if not raw_bars:
                logger.warning(
                    "no_historical_data | symbol={symbol} timeframe={timeframe} limit={limit}",
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
                return []

            bars: list[OHLCV] = [
                DataNormalizer.from_ccxt(b, symbol, timeframe) for b in raw_bars
            ]

            logger.info(
                "historical_bars_fetched | symbol={symbol} timeframe={timeframe} limit={limit} count={count}",
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                count=len(bars),
            )
            return bars

        except ccxt.BaseError:
            logger.exception(
                "historical_bars_ccxt_error | symbol={symbol} timeframe={timeframe} limit={limit}",
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            return []
        except Exception:
            logger.exception(
                "historical_bars_error | symbol={symbol} timeframe={timeframe} limit={limit}",
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            return []

    # ------------------------------------------------------------------
    # Latest price
    # ------------------------------------------------------------------
    def get_latest_price(self, symbol: str) -> float:
        """Get current price — WebSocket cache first, REST fallback.

        When a WebSocket stream is running (started via :meth:`start_stream`)
        the price is returned directly from the in-memory cache with no
        network latency.  If the WebSocket has not received a price yet,
        the method falls back to a ``fetch_ticker`` REST call.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).

        Returns
        -------
        float
            Latest traded price.  Returns ``0.0`` on error.
        """
        # Fast path: WebSocket cache
        if self._ws is not None:
            ws_price = self._ws.get_latest_price(symbol)
            if ws_price > 0:
                logger.debug(
                    "latest_price_ws | symbol={symbol} price={price}",
                    symbol=symbol, price=ws_price,
                )
                return ws_price

        # Slow path: REST fallback
        try:
            ticker: dict[str, Any] = self._exchange.fetch_ticker(symbol)
            price = float(ticker["last"])
            logger.info("latest_price | symbol={symbol} price={price}", symbol=symbol, price=price)
            return price

        except ccxt.BaseError:
            logger.exception("latest_price_ccxt_error | symbol={symbol}", symbol=symbol)
            return 0.0
        except Exception:
            logger.exception("latest_price_error | symbol={symbol}", symbol=symbol)
            return 0.0

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------
    def get_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """Get the current order book.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).
        limit:
            Number of price levels on each side.

        Returns
        -------
        dict
            A dictionary with the following shape::

                {
                    "bids": [[price, amount], ...],
                    "asks": [[price, amount], ...],
                    "symbol": "BTC/USDT",
                    "timestamp": 1700000000000,
                    "best_bid": 42000.0,
                    "best_ask": 42001.0,
                    "spread": 1.0,
                    "spread_pct": 0.0024,
                }

            Returns an empty dict on error.
        """
        try:
            book: dict[str, Any] = self._exchange.fetch_order_book(symbol, limit=limit)

            bids: list[list[float]] = book.get("bids", [])
            asks: list[list[float]] = book.get("asks", [])

            best_bid: float = bids[0][0] if bids else 0.0
            best_ask: float = asks[0][0] if asks else 0.0
            spread: float = best_ask - best_bid
            spread_pct: float = (spread / best_ask * 100.0) if best_ask else 0.0

            result: dict[str, Any] = {
                "bids": bids,
                "asks": asks,
                "symbol": symbol,
                "timestamp": book.get("timestamp"),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "spread_pct": round(spread_pct, 4),
            }

            logger.info(
                "order_book_fetched | symbol={symbol} limit={limit} best_bid={best_bid} best_ask={best_ask} spread={spread} levels={levels}",
                symbol=symbol,
                limit=limit,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                levels=limit,
            )
            return result

        except ccxt.BaseError:
            logger.exception("order_book_ccxt_error | symbol={symbol} limit={limit}", symbol=symbol, limit=limit)
            return {}
        except Exception:
            logger.exception("order_book_error | symbol={symbol} limit={limit}", symbol=symbol, limit=limit)
            return {}
