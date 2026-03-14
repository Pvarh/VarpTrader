"""Stock market data feed.

Real-time prices come from the Alpaca market-data WebSocket (free IEX
feed, 200 calls/min).  Historical OHLCV bars are fetched from the
Polygon REST API when a key is configured, otherwise yfinance is used
as a free fallback.  This separation avoids hammering low-rate-limit
REST endpoints during live trading.
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from loguru import logger

from data.normalizer import DataNormalizer
from data.stream_stocks import StockStream
from journal.models import OHLCV


class StockFeed:
    """Unified stock data feed: Alpaca WebSocket for live, Polygon/yfinance for historical.

    Call :meth:`start_stream` once at startup to begin the background
    WebSocket connection.  After that, :meth:`get_latest_price` reads
    from the in-memory cache with zero network latency.  Historical
    methods (:meth:`get_historical_bars`, :meth:`get_first_candle`,
    :meth:`get_average_volume`) never touch the WebSocket.
    """

    def __init__(
        self,
        alpaca_api_key: str | None = None,
        alpaca_secret_key: str | None = None,
        polygon_api_key: str | None = None,
        use_sip: bool = False,
    ) -> None:
        self._alpaca_api_key = alpaca_api_key or ""
        self._alpaca_secret_key = alpaca_secret_key or ""
        self._polygon_api_key = polygon_api_key
        self._use_sip = use_sip

        # Alpaca WebSocket stream (lazy-started via start_stream)
        self._stream: StockStream | None = None
        self._stream_thread: threading.Thread | None = None
        self._stream_loop: asyncio.AbstractEventLoop | None = None

        logger.info(
            "stock_feed_initialised | alpaca_ws={alpaca} polygon_rest={polygon}",
            alpaca=bool(self._alpaca_api_key),
            polygon=bool(self._polygon_api_key),
        )

    # ------------------------------------------------------------------
    # Alpaca WebSocket lifecycle
    # ------------------------------------------------------------------
    def start_stream(self, symbols: list[str]) -> None:
        """Start the Alpaca WebSocket in a background daemon thread.

        Subscribes to minute bars, quotes, and trades for *symbols*.
        Safe to call multiple times -- subsequent calls are no-ops.
        """
        if self._stream is not None:
            logger.debug("stream_already_running")
            return
        if not self._alpaca_api_key:
            logger.warning("alpaca_ws_skipped | reason=no_api_key")
            return

        self._stream = StockStream(
            api_key=self._alpaca_api_key,
            secret_key=self._alpaca_secret_key,
            use_sip=self._use_sip,
        )

        def _run(syms: list[str]) -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._stream_loop = loop
            try:
                loop.run_until_complete(self._stream.connect())
                loop.run_until_complete(self._stream.subscribe_bars(syms))
                loop.run_until_complete(self._stream.subscribe_quotes(syms))
                loop.run_until_complete(self._stream.subscribe_trades(syms))
                loop.run_until_complete(self._stream.run_forever())
            except Exception:
                logger.exception("alpaca_ws_thread_error")
            finally:
                loop.close()

        self._stream_thread = threading.Thread(
            target=_run, args=(symbols,), daemon=True, name="alpaca-ws"
        )
        self._stream_thread.start()
        logger.info("alpaca_ws_started | symbols={symbols}", symbols=symbols)

    def stop_stream(self) -> None:
        """Gracefully disconnect the Alpaca WebSocket."""
        if self._stream is None:
            return
        if self._stream_loop is not None and self._stream_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._stream.disconnect(), self._stream_loop
            )
        self._stream = None
        logger.info("alpaca_ws_stopped")

    # ------------------------------------------------------------------
    # Latest price  (Alpaca WS cache → yfinance fallback)
    # ------------------------------------------------------------------
    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent price for *symbol*.

        Primary source is the Alpaca WebSocket in-memory cache (zero
        latency, no rate limit).  Falls back to a yfinance snapshot
        if the stream has no data for the symbol yet.
        """
        # Try WebSocket cache first
        if self._stream is not None:
            price = self._stream.get_latest_price(symbol)
            if price > 0.0:
                return price

        # Fallback: yfinance fast_info
        try:
            ticker = yf.Ticker(symbol)
            price = float(ticker.fast_info["lastPrice"])
            logger.debug(
                "price_fallback_yfinance | symbol={symbol} price={price}",
                symbol=symbol,
                price=price,
            )
            return price
        except Exception:
            logger.exception("latest_price_error | symbol={symbol}", symbol=symbol)
            return 0.0

    # ------------------------------------------------------------------
    # Historical bars  (Polygon REST → yfinance fallback)
    # ------------------------------------------------------------------
    def get_historical_bars(
        self,
        symbol: str,
        period: str = "5d",
        interval: str = "5m",
    ) -> list[OHLCV]:
        """Fetch historical OHLCV bars.

        Uses the Polygon REST API when a key is configured (higher
        quality, VWAP included).  Falls back to yfinance otherwise.
        """
        if self._polygon_api_key:
            bars = self._fetch_polygon_bars(symbol, period, interval)
            if bars:
                return bars
            logger.warning(
                "polygon_fallback_to_yfinance | symbol={symbol}",
                symbol=symbol,
            )

        return self._fetch_yfinance_bars(symbol, period, interval)

    def _fetch_polygon_bars(
        self, symbol: str, period: str, interval: str
    ) -> list[OHLCV]:
        """Fetch bars from the Polygon.io REST API."""
        try:
            import requests

            multiplier, timespan = self._parse_interval_for_polygon(interval)
            from_date, to_date = self._period_to_dates(period)

            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}"
                f"/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            )
            resp = requests.get(
                url,
                params={
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": "50000",
                    "apiKey": self._polygon_api_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                return []

            bars: list[OHLCV] = []
            for raw in results:
                bars.append(DataNormalizer.from_polygon(raw, symbol, interval))

            logger.info(
                "polygon_bars_fetched | symbol={symbol} count={count}",
                symbol=symbol,
                count=len(bars),
            )
            return bars

        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                logger.warning(
                    "polygon_rate_limited | symbol={symbol} period={period}",
                    symbol=symbol,
                    period=period,
                )
            else:
                logger.exception(
                    "polygon_bars_error | symbol={symbol} period={period}",
                    symbol=symbol,
                    period=period,
                )
            return []

        except Exception:
            logger.exception(
                "polygon_bars_error | symbol={symbol} period={period}",
                symbol=symbol,
                period=period,
            )
            return []

    def _fetch_yfinance_bars(
        self, symbol: str, period: str, interval: str
    ) -> list[OHLCV]:
        """Fetch bars from yfinance (free fallback)."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df is None or df.empty:
                logger.warning(
                    "no_historical_data | symbol={symbol} period={period} interval={interval}",
                    symbol=symbol,
                    period=period,
                    interval=interval,
                )
                return []

            bars: list[OHLCV] = []
            for idx, row in df.iterrows():
                row_dict: dict[str, Any] = row.to_dict()
                row_dict["Datetime"] = idx
                bars.append(
                    DataNormalizer.from_yfinance(row_dict, symbol, interval)
                )

            logger.info(
                "yfinance_bars_fetched | symbol={symbol} period={period} count={count}",
                symbol=symbol,
                period=period,
                count=len(bars),
            )
            return bars

        except Exception:
            logger.exception(
                "yfinance_bars_error | symbol={symbol} period={period}",
                symbol=symbol,
                period=period,
            )
            return []

    # ------------------------------------------------------------------
    # First candle after market open
    # ------------------------------------------------------------------
    def get_first_candle(
        self,
        symbol: str,
        date: str | None = None,
    ) -> OHLCV | None:
        """Get the first 15-min candle after market open for the given date."""
        try:
            if date is None:
                date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

            ticker = yf.Ticker(symbol)
            try:
                df = ticker.history(start=date, interval="15m")
            except Exception:
                logger.warning(
                    "first_candle_fetch_failed | symbol={symbol} date={date}",
                    symbol=symbol,
                    date=date,
                )
                return None
            if df is None or not hasattr(df, "empty") or df.empty or len(df) == 0:
                logger.warning(
                    "no_first_candle_data | symbol={symbol} date={date}",
                    symbol=symbol,
                    date=date,
                )
                return None

            first_row = df.iloc[0]
            row_dict: dict[str, Any] = first_row.to_dict()
            row_dict["Datetime"] = df.index[0]

            bar = DataNormalizer.from_yfinance(row_dict, symbol, "15m")
            logger.info(
                "first_candle_fetched | symbol={symbol} date={date} close={close}",
                symbol=symbol,
                date=date,
                close=bar.close,
            )
            return bar

        except Exception:
            logger.exception(
                "first_candle_error | symbol={symbol} date={date}",
                symbol=symbol,
                date=date,
            )
            return None

    # ------------------------------------------------------------------
    # Average volume
    # ------------------------------------------------------------------
    def get_average_volume(self, symbol: str, period: str = "5d") -> float:
        """Calculate average daily volume over *period*."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")
            if df is None or df.empty:
                logger.warning(
                    "no_volume_data | symbol={symbol} period={period}",
                    symbol=symbol,
                    period=period,
                )
                return 0.0

            avg_vol = float(df["Volume"].mean())
            logger.info(
                "average_volume_computed | symbol={symbol} avg_volume={avg_volume}",
                symbol=symbol,
                avg_volume=avg_vol,
            )
            return avg_vol

        except Exception:
            logger.exception(
                "average_volume_error | symbol={symbol} period={period}",
                symbol=symbol,
                period=period,
            )
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_interval_for_polygon(interval: str) -> tuple[int, str]:
        """Convert yfinance-style interval to Polygon (multiplier, timespan)."""
        mapping: dict[str, tuple[int, str]] = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "1h": (1, "hour"),
            "1d": (1, "day"),
            "1wk": (1, "week"),
            "1mo": (1, "month"),
        }
        return mapping.get(interval, (5, "minute"))

    @staticmethod
    def _period_to_dates(period: str) -> tuple[str, str]:
        """Convert yfinance-style period to (from_date, to_date) ISO strings."""
        from datetime import timedelta

        now = datetime.now(tz=timezone.utc)
        to_date = now.strftime("%Y-%m-%d")

        period_map: dict[str, int] = {
            "1d": 1,
            "2d": 2,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
        }
        days = period_map.get(period, 5)
        from_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        return from_date, to_date
