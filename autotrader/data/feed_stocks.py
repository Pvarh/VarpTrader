"""Stock market data feed.

Real-time prices come from the Alpaca market-data WebSocket. Historical
bars prefer Polygon, but the Polygon path is guarded by a local
5-calls-per-minute budget with TTL caches so the app stays inside low
rate-limit plans and falls back cleanly when needed.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from loguru import logger

from data.normalizer import DataNormalizer
from data.stream_stocks import StockStream
from journal.models import OHLCV


class StockFeed:
    """Unified stock data feed with local Polygon throttling and caching."""

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

        self._stream: StockStream | None = None
        self._stream_thread: threading.Thread | None = None
        self._stream_loop: asyncio.AbstractEventLoop | None = None

        self._polygon_calls_per_minute = max(
            1, int(os.getenv("POLYGON_CALLS_PER_MINUTE", "5"))
        )
        self._polygon_request_times: deque[float] = deque()
        self._polygon_lock = threading.Lock()

        self._bars_cache: dict[tuple[str, str, str], tuple[float, list[OHLCV]]] = {}
        self._aux_cache: dict[tuple[str, str, str], tuple[float, object]] = {}
        self._cache_lock = threading.Lock()

        logger.info(
            "stock_feed_initialised | alpaca_ws={alpaca} polygon_rest={polygon} polygon_budget_per_minute={budget}",
            alpaca=bool(self._alpaca_api_key),
            polygon=bool(self._polygon_api_key),
            budget=self._polygon_calls_per_minute,
        )

    # ------------------------------------------------------------------
    # Alpaca WebSocket lifecycle
    # ------------------------------------------------------------------
    def start_stream(self, symbols: list[str]) -> None:
        """Start the Alpaca WebSocket in a background daemon thread."""
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
    # Latest price
    # ------------------------------------------------------------------
    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent price for *symbol*."""
        if self._stream is not None:
            price = self._stream.get_latest_price(symbol)
            if price > 0.0:
                return price

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
    # Historical bars
    # ------------------------------------------------------------------
    def get_historical_bars(
        self,
        symbol: str,
        period: str = "5d",
        interval: str = "5m",
    ) -> list[OHLCV]:
        """Fetch historical OHLCV bars with Polygon budget + cache guards."""
        cache_key = (symbol.upper(), period, interval)
        cached = self._get_cached_bars(cache_key, allow_stale=False)
        if cached:
            logger.debug(
                "historical_bars_cache_hit | provider=polygon symbol={symbol} period={period} interval={interval}",
                symbol=symbol,
                period=period,
                interval=interval,
            )
            return cached

        if self._polygon_api_key and self._should_use_polygon(interval):
            if self._consume_polygon_budget():
                bars = self._fetch_polygon_bars(symbol, period, interval)
                if bars:
                    self._set_cached_bars(cache_key, bars)
                    return bars
            else:
                stale = self._get_cached_bars(cache_key, allow_stale=True)
                if stale:
                    logger.warning(
                        "polygon_budget_exhausted_using_stale_cache | symbol={symbol} period={period} interval={interval}",
                        symbol=symbol,
                        period=period,
                        interval=interval,
                    )
                    return stale
                logger.warning(
                    "polygon_budget_exhausted | symbol={symbol} period={period} interval={interval}",
                    symbol=symbol,
                    period=period,
                    interval=interval,
                )

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

            bars = [
                DataNormalizer.from_polygon(raw, symbol, interval)
                for raw in results
            ]
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
                bars.append(DataNormalizer.from_yfinance(row_dict, symbol, interval))

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
        """Get the first 15-minute candle after market open for the given date."""
        try:
            if date is None:
                date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

            dt = datetime.strptime(date, "%Y-%m-%d")
            if dt.weekday() >= 5:
                return None

            cache_key = (symbol.upper(), date, "first_candle")
            cached = self._get_aux_cache(cache_key, ttl_seconds=6 * 60 * 60)
            if cached is not None:
                return cached if isinstance(cached, OHLCV) else None

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
            self._set_aux_cache(cache_key, bar)
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
        """Calculate average daily volume over *period* with a small TTL cache."""
        try:
            cache_key = (symbol.upper(), period, "avg_volume")
            cached = self._get_aux_cache(cache_key, ttl_seconds=30 * 60)
            if isinstance(cached, float):
                return cached

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
            self._set_aux_cache(cache_key, avg_vol)
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
    def _consume_polygon_budget(self) -> bool:
        """Reserve one Polygon request slot within the local minute budget."""
        now = time.monotonic()
        with self._polygon_lock:
            while self._polygon_request_times and now - self._polygon_request_times[0] >= 60:
                self._polygon_request_times.popleft()

            if len(self._polygon_request_times) >= self._polygon_calls_per_minute:
                return False

            self._polygon_request_times.append(now)
            return True

    def _bars_ttl_seconds(self, period: str, interval: str) -> int:
        """Return a TTL matched to the requested bar granularity."""
        if interval == "1h":
            return 60 * 60
        if interval in {"5m", "15m", "30m"}:
            return 6 * 60
        if interval == "1d":
            return 6 * 60 * 60
        return 15 * 60

    @staticmethod
    def _should_use_polygon(interval: str) -> bool:
        """Reserve Polygon budget for minute aggregates on low-tier plans."""
        return interval in {"1m", "5m", "15m", "30m"}

    def _get_cached_bars(
        self,
        cache_key: tuple[str, str, str],
        *,
        allow_stale: bool,
    ) -> list[OHLCV]:
        """Get cached bars when present and still fresh unless stale is allowed."""
        cached = None
        with self._cache_lock:
            cached = self._bars_cache.get(cache_key)
        if cached is None:
            return []

        created_at, bars = cached
        age = time.monotonic() - created_at
        ttl = self._bars_ttl_seconds(cache_key[1], cache_key[2])
        if allow_stale or age <= ttl:
            return bars
        return []

    def _set_cached_bars(
        self,
        cache_key: tuple[str, str, str],
        bars: list[OHLCV],
    ) -> None:
        """Store historical bars in the in-memory cache."""
        with self._cache_lock:
            self._bars_cache[cache_key] = (time.monotonic(), bars)

    def _get_aux_cache(
        self,
        cache_key: tuple[str, str, str],
        *,
        ttl_seconds: int,
    ) -> object | None:
        """Get a cached auxiliary value if still fresh."""
        cached = None
        with self._cache_lock:
            cached = self._aux_cache.get(cache_key)
        if cached is None:
            return None

        created_at, value = cached
        if time.monotonic() - created_at <= ttl_seconds:
            return value
        return None

    def _set_aux_cache(
        self,
        cache_key: tuple[str, str, str],
        value: object,
    ) -> None:
        """Store a small auxiliary cached value."""
        with self._cache_lock:
            self._aux_cache[cache_key] = (time.monotonic(), value)

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
