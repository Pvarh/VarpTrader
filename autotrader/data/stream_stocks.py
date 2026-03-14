"""Real-time stock data streaming via Alpaca WebSocket."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

import websockets
from loguru import logger
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from data.normalizer import DataNormalizer
from journal.models import OHLCV


class StockStream:
    """Alpaca WebSocket data stream for real-time stock quotes and bars.

    Connects to the Alpaca data streaming API (v2) and provides
    callback-driven delivery of bars, quotes, and trades.  Supports both
    the free IEX feed and the paid SIP feed.

    Usage::

        stream = StockStream(api_key="...", secret_key="...")
        stream.on_bar(my_bar_handler)
        stream.on_quote(my_quote_handler)
        await stream.connect()
        await stream.subscribe_bars(["AAPL", "MSFT"])
        await stream.run_forever()
    """

    ALPACA_STREAM_URL = "wss://stream.data.alpaca.markets/v2/iex"
    ALPACA_STREAM_URL_SIP = "wss://stream.data.alpaca.markets/v2/sip"

    # Reconnection parameters
    _RECONNECT_BASE_DELAY: float = 1.0
    _RECONNECT_MAX_DELAY: float = 60.0
    _RECONNECT_BACKOFF_FACTOR: float = 2.0

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        use_sip: bool = False,
    ) -> None:
        """Initialize with Alpaca credentials.

        Parameters
        ----------
        api_key:
            Alpaca API key.
        secret_key:
            Alpaca secret key.
        use_sip:
            Use SIP (paid) feed vs IEX (free) feed.
        """
        self._api_key: str = api_key
        self._secret_key: str = secret_key
        self._url: str = (
            self.ALPACA_STREAM_URL_SIP if use_sip else self.ALPACA_STREAM_URL
        )
        self._ws: Any = None
        self._running: bool = False
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._latest_prices: dict[str, float] = {}
        self._latest_bars: dict[str, dict[str, Any]] = {}
        self._subscribed_bars: list[str] = []
        self._subscribed_quotes: list[str] = []
        self._subscribed_trades: list[str] = []

        logger.info(
            "stock_stream_initialised | feed={feed} url={url}",
            feed="SIP" if use_sip else "IEX",
            url=self._url,
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Establish WebSocket connection and authenticate.

        Opens the WebSocket to the Alpaca streaming endpoint and sends
        authentication credentials.  The server responds with an ``[{"T":
        "success", "msg": "connected"}]`` message followed by an auth
        confirmation.

        Raises
        ------
        ConnectionError
            When the initial connection or authentication fails.
        """
        try:
            self._ws = await websockets.connect(
                self._url,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
            )
            logger.info("websocket_connected | url={url}", url=self._url)

            # Read the initial welcome message
            welcome_raw = await self._ws.recv()
            welcome = json.loads(welcome_raw)
            logger.debug("ws_welcome | msg={msg}", msg=welcome)

            await self._authenticate()
            self._running = True

        except Exception as exc:
            logger.exception("websocket_connect_error | url={url}", url=self._url)
            raise ConnectionError(
                f"Failed to connect to Alpaca stream: {exc}"
            ) from exc

    async def _authenticate(self) -> None:
        """Send auth message with API keys.

        Sends the ``auth`` action containing the API key and secret.
        Waits for the server's auth response and raises on failure.

        Raises
        ------
        ConnectionError
            When the server rejects the credentials.
        """
        auth_msg = {
            "action": "auth",
            "key": self._api_key,
            "secret": self._secret_key,
        }
        await self._ws.send(json.dumps(auth_msg))

        auth_response_raw = await self._ws.recv()
        auth_response = json.loads(auth_response_raw)
        logger.debug("ws_auth_response | msg={msg}", msg=auth_response)

        # Alpaca wraps responses in a list
        responses = auth_response if isinstance(auth_response, list) else [auth_response]
        for resp in responses:
            if resp.get("T") == "error":
                error_msg = resp.get("msg", "unknown auth error")
                logger.error("ws_auth_failed | error={error}", error=error_msg)
                raise ConnectionError(f"Alpaca auth failed: {error_msg}")
            if resp.get("T") == "success" and resp.get("msg") == "authenticated":
                logger.info("ws_authenticated")
                return

        logger.warning("ws_auth_no_confirmation | response={response}", response=auth_response)

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    async def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: str = "1Min",
    ) -> None:
        """Subscribe to real-time bar data for given symbols.

        Parameters
        ----------
        symbols:
            List of ticker symbols (e.g. ``["AAPL", "MSFT"]``).
        timeframe:
            Bar timeframe.  Alpaca supports ``"1Min"`` for minute bars.
            Daily and other bars arrive via their own message types.

        Note
        ----
        The ``timeframe`` parameter is stored for normalisation but Alpaca
        v2 only supports minute bars on the real-time stream; higher
        timeframes should be aggregated client-side.
        """
        if not self._ws:
            logger.error("subscribe_bars_no_connection")
            return

        upper_symbols = [s.upper() for s in symbols]
        self._subscribed_bars = list(set(self._subscribed_bars + upper_symbols))

        sub_msg = {"action": "subscribe", "bars": upper_symbols}
        await self._ws.send(json.dumps(sub_msg))
        logger.info(
            "subscribed_bars | symbols={symbols} timeframe={timeframe}",
            symbols=upper_symbols,
            timeframe=timeframe,
        )

    async def subscribe_quotes(self, symbols: list[str]) -> None:
        """Subscribe to real-time quote updates.

        Parameters
        ----------
        symbols:
            List of ticker symbols (e.g. ``["AAPL", "MSFT"]``).
        """
        if not self._ws:
            logger.error("subscribe_quotes_no_connection")
            return

        upper_symbols = [s.upper() for s in symbols]
        self._subscribed_quotes = list(set(self._subscribed_quotes + upper_symbols))

        sub_msg = {"action": "subscribe", "quotes": upper_symbols}
        await self._ws.send(json.dumps(sub_msg))
        logger.info("subscribed_quotes | symbols={symbols}", symbols=upper_symbols)

    async def subscribe_trades(self, symbols: list[str]) -> None:
        """Subscribe to real-time trade data.

        Parameters
        ----------
        symbols:
            List of ticker symbols (e.g. ``["AAPL", "MSFT"]``).
        """
        if not self._ws:
            logger.error("subscribe_trades_no_connection")
            return

        upper_symbols = [s.upper() for s in symbols]
        self._subscribed_trades = list(set(self._subscribed_trades + upper_symbols))

        sub_msg = {"action": "subscribe", "trades": upper_symbols}
        await self._ws.send(json.dumps(sub_msg))
        logger.info("subscribed_trades | symbols={symbols}", symbols=upper_symbols)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------
    def on_bar(self, callback: Callable) -> None:
        """Register a callback for bar events.

        Parameters
        ----------
        callback:
            Callable receiving ``(symbol: str, bar: OHLCV)``.
        """
        self._callbacks["bar"].append(callback)
        logger.debug("callback_registered | event=bar")

    def on_quote(self, callback: Callable) -> None:
        """Register a callback for quote events.

        Parameters
        ----------
        callback:
            Callable receiving ``(symbol: str, bid: float, ask: float)``.
        """
        self._callbacks["quote"].append(callback)
        logger.debug("callback_registered | event=quote")

    def on_trade(self, callback: Callable) -> None:
        """Register a callback for trade events.

        Parameters
        ----------
        callback:
            Callable receiving ``(symbol: str, price: float, size: float)``.
        """
        self._callbacks["trade"].append(callback)
        logger.debug("callback_registered | event=trade")

    # ------------------------------------------------------------------
    # Price cache
    # ------------------------------------------------------------------
    def get_latest_price(self, symbol: str) -> float:
        """Get the most recent cached price for a symbol.

        The price is updated by incoming trade, quote, or bar messages.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"AAPL"``).

        Returns
        -------
        float
            Latest cached price.  Returns ``0.0`` when no price has been
            received yet.
        """
        return self._latest_prices.get(symbol.upper(), 0.0)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------
    async def _listen(self) -> None:
        """Main message loop -- parse incoming messages and dispatch to callbacks.

        Reads from the WebSocket in a loop, deserialises JSON messages,
        and routes each to ``_handle_message``.  The loop exits when the
        connection is closed or ``_running`` is set to ``False``.
        """
        try:
            async for raw_msg in self._ws:
                if not self._running:
                    break

                try:
                    messages = json.loads(raw_msg)
                except json.JSONDecodeError:
                    logger.warning("ws_invalid_json | raw={raw}", raw=raw_msg[:200])
                    continue

                # Alpaca sends messages as a JSON array
                if isinstance(messages, list):
                    for msg in messages:
                        await self._handle_message(msg)
                else:
                    await self._handle_message(messages)

        except ConnectionClosedOK:
            logger.info("ws_connection_closed_ok")
        except ConnectionClosedError as exc:
            logger.warning(
                "ws_connection_closed_error | code={code} reason={reason}",
                code=exc.code,
                reason=exc.reason,
            )
        except ConnectionClosed as exc:
            logger.warning(
                "ws_connection_closed | code={code} reason={reason}",
                code=exc.code,
                reason=exc.reason,
            )

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        """Process a single WebSocket message.

        Routes the message by its ``T`` (type) field:

        - ``"b"`` -- bar data
        - ``"q"`` -- quote data
        - ``"t"`` -- trade data
        - ``"subscription"`` -- subscription confirmation
        - ``"success"`` / ``"error"`` -- control messages

        Parameters
        ----------
        msg:
            Parsed JSON message dictionary.
        """
        msg_type = msg.get("T")

        if msg_type == "b":
            await self._handle_bar(msg)
        elif msg_type == "q":
            await self._handle_quote(msg)
        elif msg_type == "t":
            await self._handle_trade(msg)
        elif msg_type == "subscription":
            logger.info(
                "subscription_confirmed | bars={bars} quotes={quotes} trades={trades}",
                bars=msg.get("bars", []),
                quotes=msg.get("quotes", []),
                trades=msg.get("trades", []),
            )
        elif msg_type == "error":
            logger.error(
                "ws_server_error | code={code} error={error}",
                code=msg.get("code"),
                error=msg.get("msg"),
            )
        elif msg_type == "success":
            logger.debug("ws_success | msg={msg}", msg=msg.get("msg"))
        else:
            logger.debug(
                "ws_unknown_message_type | msg_type={msg_type} msg={msg}",
                msg_type=msg_type,
                msg=msg,
            )

    async def _handle_bar(self, msg: dict[str, Any]) -> None:
        """Process an incoming bar message and dispatch to callbacks.

        Alpaca bar format::

            {
                "T": "b", "S": "AAPL",
                "o": 150.0, "h": 151.0, "l": 149.5, "c": 150.5,
                "v": 12345, "t": "2025-01-15T14:30:00Z",
                "n": 100, "vw": 150.25
            }

        Parameters
        ----------
        msg:
            Parsed bar message dictionary.
        """
        symbol = msg.get("S", "")
        try:
            ts = datetime.fromisoformat(msg["t"].replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            vwap = float(msg["vw"]) if "vw" in msg and msg["vw"] is not None else None

            bar = OHLCV(
                timestamp=ts,
                open=float(msg["o"]),
                high=float(msg["h"]),
                low=float(msg["l"]),
                close=float(msg["c"]),
                volume=float(msg["v"]),
                symbol=symbol.upper(),
                timeframe="1m",
                market="stock",
                vwap=vwap,
            )

            self._latest_prices[symbol.upper()] = bar.close
            self._latest_bars[symbol.upper()] = msg

            for cb in self._callbacks.get("bar", []):
                if asyncio.iscoroutinefunction(cb):
                    await cb(symbol, bar)
                else:
                    cb(symbol, bar)

            logger.debug("bar_received | symbol={symbol} close={close}", symbol=symbol, close=bar.close)

        except (KeyError, ValueError, TypeError):
            logger.exception("bar_parse_error | symbol={symbol} msg={msg}", symbol=symbol, msg=msg)

    async def _handle_quote(self, msg: dict[str, Any]) -> None:
        """Process an incoming quote message and dispatch to callbacks.

        Alpaca quote format::

            {
                "T": "q", "S": "AAPL",
                "bp": 150.0, "bs": 1, "bx": "V",
                "ap": 150.01, "as": 2, "ax": "V",
                "t": "2025-01-15T14:30:00Z", "c": ["R"]
            }

        Parameters
        ----------
        msg:
            Parsed quote message dictionary.
        """
        symbol = msg.get("S", "")
        try:
            bid = float(msg.get("bp", 0.0))
            ask = float(msg.get("ap", 0.0))

            # Update latest price with the midpoint
            if bid > 0 and ask > 0:
                self._latest_prices[symbol.upper()] = (bid + ask) / 2.0

            for cb in self._callbacks.get("quote", []):
                if asyncio.iscoroutinefunction(cb):
                    await cb(symbol, bid, ask)
                else:
                    cb(symbol, bid, ask)

            logger.debug("quote_received | symbol={symbol} bid={bid} ask={ask}", symbol=symbol, bid=bid, ask=ask)

        except (KeyError, ValueError, TypeError):
            logger.exception("quote_parse_error | symbol={symbol} msg={msg}", symbol=symbol, msg=msg)

    async def _handle_trade(self, msg: dict[str, Any]) -> None:
        """Process an incoming trade message and dispatch to callbacks.

        Alpaca trade format::

            {
                "T": "t", "S": "AAPL",
                "p": 150.0, "s": 100,
                "t": "2025-01-15T14:30:00Z",
                "i": 12345, "x": "V", "c": ["@"]
            }

        Parameters
        ----------
        msg:
            Parsed trade message dictionary.
        """
        symbol = msg.get("S", "")
        try:
            price = float(msg.get("p", 0.0))
            size = float(msg.get("s", 0.0))

            if price > 0:
                self._latest_prices[symbol.upper()] = price

            for cb in self._callbacks.get("trade", []):
                if asyncio.iscoroutinefunction(cb):
                    await cb(symbol, price, size)
                else:
                    cb(symbol, price, size)

            logger.debug(
                "trade_received | symbol={symbol} price={price} size={size}",
                symbol=symbol,
                price=price,
                size=size,
            )

        except (KeyError, ValueError, TypeError):
            logger.exception("trade_parse_error | symbol={symbol} msg={msg}", symbol=symbol, msg=msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def disconnect(self) -> None:
        """Close the WebSocket connection.

        Sets ``_running`` to ``False`` and closes the underlying socket.
        Safe to call multiple times.
        """
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
                logger.info("websocket_disconnected")
            except Exception:
                logger.exception("websocket_disconnect_error")
            finally:
                self._ws = None

    async def _resubscribe(self) -> None:
        """Re-send all active subscriptions after a reconnect.

        Called internally by ``run_forever`` after a successful reconnect
        to restore the previous subscription state.
        """
        if self._subscribed_bars:
            await self.subscribe_bars(self._subscribed_bars)
        if self._subscribed_quotes:
            await self.subscribe_quotes(self._subscribed_quotes)
        if self._subscribed_trades:
            await self.subscribe_trades(self._subscribed_trades)

    async def run_forever(self) -> None:
        """Connect, authenticate, and listen with auto-reconnect.

        Enters an infinite loop that maintains the WebSocket connection.
        On disconnect, it waits with exponential backoff before
        reconnecting and re-subscribing.  The loop exits only when
        ``disconnect()`` is called.
        """
        delay = self._RECONNECT_BASE_DELAY

        while True:
            try:
                if not self._running:
                    await self.connect()

                await self._resubscribe()
                await self._listen()

                # If _running is False, we were told to stop
                if not self._running:
                    logger.info("stream_stopped_gracefully")
                    break

            except ConnectionError:
                logger.exception("stream_connection_error")
            except Exception:
                logger.exception("stream_unexpected_error")

            # Reconnect with exponential backoff
            if not self._running:
                break

            logger.info("stream_reconnecting | delay_seconds={delay_seconds}", delay_seconds=delay)
            await asyncio.sleep(delay)
            delay = min(delay * self._RECONNECT_BACKOFF_FACTOR, self._RECONNECT_MAX_DELAY)

            # Reset connection state for reconnect
            self._ws = None
            self._running = False

            try:
                await self.connect()
                # Reset delay on successful reconnect
                delay = self._RECONNECT_BASE_DELAY
                logger.info("stream_reconnected")
            except ConnectionError:
                logger.warning("stream_reconnect_failed | next_delay={next_delay}", next_delay=delay)
