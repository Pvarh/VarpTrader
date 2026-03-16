"""Native Binance WebSocket client for real-time crypto price and kline data.

Uses the ``websockets`` library to connect directly to Binance's public
combined-stream endpoint.  No API key required — all streams are public.

Usage::

    ws = BinanceWebSocket()
    ws.start(["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    price = ws.get_latest_price("BTC/USDT")       # fast, non-blocking
    candles = ws.get_recent_candles("BTC/USDT", 5) # last 5 closed candles

    ws.stop()
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from journal.models import OHLCV

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
_RECONNECT_BASE   = 2.0
_RECONNECT_MAX    = 60.0
_CANDLE_BUFFER    = 200   # closed candles kept per symbol


def _symbol_to_stream(symbol: str) -> str:
    """Convert 'BTC/USDT' -> 'btcusdt'."""
    return symbol.replace("/", "").lower()


def _stream_to_symbol(stream_name: str) -> str:
    """Convert 'btcusdt@ticker' -> 'BTC/USDT' (best effort)."""
    base = stream_name.split("@")[0].upper()
    # Known quote currencies in order of length (longer first to avoid partial match)
    for quote in ("USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"):
        if base.endswith(quote):
            base_asset = base[: -len(quote)]
            return f"{base_asset}/{quote}"
    return base


class BinanceWebSocket:
    """Real-time price and kline feed via Binance public WebSocket streams.

    Subscribes to ``<symbol>@ticker`` and ``<symbol>@kline_5m`` streams
    for each requested symbol.  Prices are updated on every ticker event;
    complete (closed) 5-minute candles are accumulated in a rolling buffer
    so signal evaluation can call ``get_recent_candles()`` without any
    REST requests.

    The WebSocket loop runs in a daemon background thread — it will not
    block the main process.

    Attributes
    ----------
    connected : bool
        ``True`` once the WebSocket has successfully authenticated at least once.
    """

    def __init__(self) -> None:
        self._prices: dict[str, float] = {}
        # symbol -> deque of closed OHLCV candles (5m)
        self._candles: dict[str, deque] = defaultdict(lambda: deque(maxlen=_CANDLE_BUFFER))
        self._symbols: list[str] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.connected = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, symbols: list[str], timeframe: str = "5m") -> None:
        """Start the WebSocket in a daemon background thread.

        Parameters
        ----------
        symbols:
            List of trading pairs e.g. ``["BTC/USDT", "ETH/USDT"]``.
        timeframe:
            Kline timeframe.  Binance accepts ``"1m"``, ``"5m"``, ``"15m"``,
            ``"1h"``, ``"4h"``, ``"1d"``, etc.
        """
        if self._running:
            logger.warning("binance_ws_already_running")
            return

        self._symbols = symbols
        self._timeframe = timeframe
        self._running = True

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(symbols, timeframe),
            daemon=True,
            name="binance-ws",
        )
        self._thread.start()
        logger.info(
            "binance_ws_started | symbols={} timeframe={}",
            symbols, timeframe,
        )

    def stop(self) -> None:
        """Gracefully stop the WebSocket loop."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("binance_ws_stopped")

    def get_latest_price(self, symbol: str) -> float:
        """Return the latest cached ticker price.

        Parameters
        ----------
        symbol:
            Trading pair e.g. ``"BTC/USDT"``.

        Returns
        -------
        float
            Most recent ``lastPrice`` from the ticker stream, or ``0.0``
            if no data has been received yet.
        """
        return self._prices.get(symbol.upper(), 0.0)

    def get_recent_candles(self, symbol: str, limit: int = 200) -> list[OHLCV]:
        """Return the most recent closed kline candles from the WS buffer.

        Parameters
        ----------
        symbol:
            Trading pair e.g. ``"BTC/USDT"``.
        limit:
            Maximum number of candles to return (newest last).

        Returns
        -------
        list[OHLCV]
            Chronologically ordered list of closed candles from the buffer.
        """
        buf = self._candles.get(symbol.upper())
        if not buf:
            return []
        candles = list(buf)[-limit:]
        return candles

    def is_ready(self, symbol: str, min_candles: int = 50) -> bool:
        """Return True when the buffer has enough closed candles to run signals.

        Parameters
        ----------
        symbol:
            Trading pair.
        min_candles:
            Minimum number of closed candles required.
        """
        return len(self._candles.get(symbol.upper(), [])) >= min_candles

    # ------------------------------------------------------------------
    # Internal asyncio loop (runs in background thread)
    # ------------------------------------------------------------------

    def _run_loop(self, symbols: list[str], timeframe: str) -> None:
        """Entry point for the background thread.  Runs its own event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        delay = _RECONNECT_BASE

        while self._running:
            try:
                self._loop.run_until_complete(
                    self._connect_and_stream(symbols, timeframe)
                )
                delay = _RECONNECT_BASE   # clean exit, reset backoff
            except Exception:
                if not self._running:
                    break
                logger.exception("binance_ws_error | reconnecting in {}s", delay)
                time.sleep(delay)
                delay = min(delay * 2, _RECONNECT_MAX)

        self._loop.close()

    async def _connect_and_stream(
        self, symbols: list[str], timeframe: str
    ) -> None:
        """Build the combined-stream URL and consume messages until disconnect."""
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            logger.error("websockets_not_installed | pip install websockets")
            self._running = False
            return

        streams: list[str] = []
        for sym in symbols:
            s = _symbol_to_stream(sym)
            streams.append(f"{s}@ticker")
            streams.append(f"{s}@kline_{timeframe}")

        url = f"{_BINANCE_WS_BASE}?streams={'/'.join(streams)}"
        logger.info("binance_ws_connecting | url_prefix={}", url[:80])

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self.connected = True
            logger.info("binance_ws_connected | streams={}", len(streams))

            async for raw in ws:
                if not self._running:
                    break
                try:
                    self._handle_message(json.loads(raw))
                except Exception:
                    logger.exception("binance_ws_message_error")

        self.connected = False

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """Route an incoming combined-stream message to the right handler."""
        stream = msg.get("stream", "")
        data   = msg.get("data", {})

        if "@ticker" in stream:
            self._handle_ticker(data)
        elif "@kline_" in stream:
            self._handle_kline(data, stream)

    def _handle_ticker(self, data: dict[str, Any]) -> None:
        """Update price cache from a 24h ticker event."""
        raw_sym = data.get("s", "")          # e.g. "BTCUSDT"
        price   = float(data.get("c", 0))    # last price
        if raw_sym and price > 0:
            # Store under normalised key e.g. "BTC/USDT"
            sym = _stream_to_symbol(raw_sym.lower() + "@ticker")
            self._prices[sym.upper()] = price
            logger.debug("binance_ws_price | symbol={} price={}", sym, price)

    def _handle_kline(self, data: dict[str, Any], stream: str) -> None:
        """Append a closed kline to the buffer."""
        k = data.get("k", {})
        if not k.get("x", False):   # x = is candle closed?
            return

        raw_sym = data.get("s", k.get("s", ""))
        sym     = _stream_to_symbol(raw_sym.lower() + "@kline")
        tf      = k.get("i", "5m")

        candle = OHLCV(
            timestamp=datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            symbol=sym,
            timeframe=tf,
            market="crypto",
        )
        self._candles[sym.upper()].append(candle)
        logger.debug(
            "binance_ws_candle | symbol={} close={} candles_buffered={}",
            sym, candle.close, len(self._candles[sym.upper()]),
        )
