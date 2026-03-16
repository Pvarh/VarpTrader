"""Historical data loading for backtesting.

Provides utilities to load OHLCV data from multiple sources (yfinance,
CSV files) and to resample candles to larger timeframes.  All methods
return lists of :class:`journal.models.OHLCV` dataclass instances so
they integrate seamlessly with the rest of the trading system.
"""

from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from journal.models import OHLCV

# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------

_TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def _parse_timeframe_minutes(timeframe: str) -> int:
    """Convert a timeframe string like ``'5m'`` or ``'1h'`` to minutes.

    Args:
        timeframe: Human-readable timeframe (e.g. ``'15m'``, ``'1h'``,
            ``'1d'``).

    Returns:
        Number of minutes represented by *timeframe*.

    Raises:
        ValueError: If *timeframe* cannot be parsed.
    """
    if timeframe in _TIMEFRAME_MINUTES:
        return _TIMEFRAME_MINUTES[timeframe]

    # Attempt dynamic parse
    if timeframe.endswith("m"):
        return int(timeframe[:-1])
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 60
    if timeframe.endswith("d"):
        return int(timeframe[:-1]) * 1440

    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _timeframe_to_pandas_freq(timeframe: str) -> str:
    """Convert a timeframe string to a pandas-compatible frequency rule.

    Args:
        timeframe: Human-readable timeframe.

    Returns:
        A string usable by ``pandas.DataFrame.resample`` (e.g. ``'15min'``,
        ``'1h'``, ``'1D'``).
    """
    mapping: dict[str, str] = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }
    if timeframe in mapping:
        return mapping[timeframe]

    if timeframe.endswith("m"):
        return f"{timeframe[:-1]}min"
    if timeframe.endswith("h"):
        return timeframe
    if timeframe.endswith("d"):
        return f"{timeframe[:-1]}D"

    raise ValueError(f"Unsupported timeframe for pandas resample: {timeframe}")


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------


class HistoricalDataLoader:
    """Load historical OHLCV data for backtesting.

    All ``load_*`` methods return a chronologically sorted list of
    :class:`OHLCV` objects ready for consumption by
    :class:`backtest.engine.BacktestEngine`.
    """

    # ------------------------------------------------------------------ #
    # yfinance
    # ------------------------------------------------------------------ #

    def load_from_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "5m",
    ) -> list[OHLCV]:
        """Download historical data from yfinance and return as OHLCV list.

        Args:
            symbol: Ticker symbol recognised by Yahoo Finance (e.g.
                ``'AAPL'``, ``'BTC-USD'``).
            start_date: Start date in ``'YYYY-MM-DD'`` format.
            end_date: End date in ``'YYYY-MM-DD'`` format.
            interval: Bar interval understood by yfinance (e.g. ``'1m'``,
                ``'5m'``, ``'1h'``, ``'1d'``).

        Returns:
            Chronologically sorted list of :class:`OHLCV` objects.

        Raises:
            ImportError: If the *yfinance* package is not installed.
        """
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "yfinance is required for load_from_yfinance.  "
                "Install it with:  pip install yfinance"
            )

        log = logger.bind(
            symbol=symbol, start=start_date, end=end_date, interval=interval,
        )
        log.info("downloading_yfinance_data")

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            log.warning("no_data_returned_from_yfinance")
            return []

        market = "crypto" if "/" in symbol or symbol.endswith("-USD") else "stock"
        timeframe = interval

        candles: list[OHLCV] = []
        for ts, row in df.iterrows():
            candles.append(
                OHLCV(
                    timestamp=ts.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    symbol=symbol,
                    timeframe=timeframe,
                    market=market,
                )
            )

        candles.sort(key=lambda c: c.timestamp)
        log.info("yfinance_data_loaded", candle_count=len(candles))
        return candles

    # ------------------------------------------------------------------ #
    # CSV
    # ------------------------------------------------------------------ #

    def load_from_csv(
        self,
        filepath: str,
        symbol: str,
        market: str,
        timeframe: str,
    ) -> list[OHLCV]:
        """Load OHLCV data from a CSV file.

        The CSV must contain a header row with at least the columns:
        ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``.
        An optional ``vwap`` column is also supported.

        Args:
            filepath: Path to the CSV file.
            symbol: Trading symbol to attach to each bar.
            market: ``'stock'`` or ``'crypto'``.
            timeframe: Bar timeframe label (e.g. ``'5m'``, ``'1h'``).

        Returns:
            Chronologically sorted list of :class:`OHLCV` objects.
        """
        log = logger.bind(filepath=filepath, symbol=symbol, market=market)

        candles: list[OHLCV] = []
        with open(filepath, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ts = datetime.fromisoformat(row["timestamp"])
                vwap: Optional[float] = None
                if "vwap" in row and row["vwap"]:
                    vwap = float(row["vwap"])

                candles.append(
                    OHLCV(
                        timestamp=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        symbol=symbol,
                        timeframe=timeframe,
                        market=market,
                        vwap=vwap,
                    )
                )

        candles.sort(key=lambda c: c.timestamp)
        log.info("csv_data_loaded", candle_count=len(candles))
        return candles

    # ------------------------------------------------------------------ #
    # Polygon.io
    # ------------------------------------------------------------------ #

    def load_from_polygon(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
        api_key: str | None = None,
    ) -> list[OHLCV]:
        """Download historical OHLCV data from Polygon.io REST API.

        The free Polygon tier supports **daily** bars (``"1d"``) for the
        previous two years.  Intraday bars (``"1m"``, ``"5m"``, ``"1h"``)
        require a paid subscription.

        Args:
            symbol: Ticker symbol (e.g. ``'NVDA'``, ``'AAPL'``, ``'SPY'``).
            start_date: Start date in ``'YYYY-MM-DD'`` format.
            end_date: End date in ``'YYYY-MM-DD'`` format.
            timeframe: Bar size — ``'1m'``, ``'5m'``, ``'15m'``, ``'1h'``,
                ``'4h'``, or ``'1d'``.
            api_key: Polygon API key.  Falls back to the
                ``POLYGON_API_KEY`` environment variable.

        Returns:
            Chronologically sorted list of :class:`OHLCV` objects.

        Raises:
            ValueError: If *timeframe* is not recognised.
            RuntimeError: If no API key is available.
        """
        try:
            import requests  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("requests is required: pip install requests")

        key = api_key or os.getenv("POLYGON_API_KEY", "")
        if not key:
            raise RuntimeError(
                "No POLYGON_API_KEY found.  Pass api_key= or set the env var."
            )

        multiplier, timespan = self._parse_polygon_timeframe(timeframe)

        log = logger.bind(
            symbol=symbol, start=start_date, end=end_date, timeframe=timeframe,
        )
        log.info("downloading_polygon_data")

        market = "crypto" if "/" in symbol else "stock"
        # Polygon uses different ticker format for crypto
        poly_ticker = symbol.replace("/", "") if market == "crypto" else symbol

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{poly_ticker}"
            f"/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        )
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": key,
        }

        candles: list[OHLCV] = []
        next_url: str | None = url

        while next_url:
            try:
                resp = requests.get(next_url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                log.exception("polygon_request_failed")
                break

            status = data.get("status", "")
            if status not in ("OK", "DELAYED"):
                log.warning(
                    "polygon_bad_status | status={} message={}",
                    status, data.get("message", ""),
                )
                break

            for bar in data.get("results", []):
                ts = datetime.fromtimestamp(bar["t"] / 1000)
                candles.append(
                    OHLCV(
                        timestamp=ts,
                        open=float(bar["o"]),
                        high=float(bar["h"]),
                        low=float(bar["l"]),
                        close=float(bar["c"]),
                        volume=float(bar["v"]),
                        symbol=symbol,
                        timeframe=timeframe,
                        market=market,
                        vwap=float(bar.get("vw", 0)) or None,
                    )
                )

            # Polygon paginates via "next_url" field
            next_url = data.get("next_url")
            if next_url:
                params = {"apiKey": key}   # next_url already has other params
                time.sleep(0.2)            # respect rate limit

        candles.sort(key=lambda c: c.timestamp)
        log.info("polygon_data_loaded | count={}", len(candles))
        return candles

    @staticmethod
    def _parse_polygon_timeframe(timeframe: str) -> tuple[int, str]:
        """Convert a timeframe string to a (multiplier, timespan) tuple.

        Args:
            timeframe: Human-readable timeframe (e.g. ``'5m'``, ``'1h'``,
                ``'1d'``).

        Returns:
            ``(multiplier, timespan)`` accepted by the Polygon aggregates
            endpoint (e.g. ``(5, 'minute')`` for ``'5m'``).

        Raises:
            ValueError: For unrecognised timeframe strings.
        """
        _MAP: dict[str, tuple[int, str]] = {
            "1m":  (1,  "minute"),
            "5m":  (5,  "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "1h":  (1,  "hour"),
            "4h":  (4,  "hour"),
            "1d":  (1,  "day"),
        }
        if timeframe in _MAP:
            return _MAP[timeframe]

        # Dynamic parse
        if timeframe.endswith("m"):
            return (int(timeframe[:-1]), "minute")
        if timeframe.endswith("h"):
            return (int(timeframe[:-1]), "hour")
        if timeframe.endswith("d"):
            return (int(timeframe[:-1]), "day")

        raise ValueError(f"Unsupported timeframe for Polygon: {timeframe}")

    # ------------------------------------------------------------------ #
    # Resample
    # ------------------------------------------------------------------ #

    def resample(
        self,
        candles: list[OHLCV],
        target_timeframe: str,
    ) -> list[OHLCV]:
        """Resample candles to a larger timeframe (e.g. 5m -> 15m, 5m -> 1h).

        Uses *pandas* for efficient grouping and aggregation.

        Args:
            candles: Source candle list (must all share the same *symbol*
                and *market*).
            target_timeframe: Desired output timeframe (e.g. ``'15m'``,
                ``'1h'``, ``'1d'``).

        Returns:
            A new list of :class:`OHLCV` objects at the target timeframe,
            sorted chronologically.

        Raises:
            ImportError: If *pandas* is not installed.
            ValueError: If *target_timeframe* cannot be parsed.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pandas is required for resample.  "
                "Install it with:  pip install pandas"
            )

        if not candles:
            return []

        freq = _timeframe_to_pandas_freq(target_timeframe)

        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        resampled = (
            df.resample(freq)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        symbol = candles[0].symbol
        market = candles[0].market

        result: list[OHLCV] = []
        for ts, row in resampled.iterrows():
            result.append(
                OHLCV(
                    timestamp=ts.to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    symbol=symbol,
                    timeframe=target_timeframe,
                    market=market,
                )
            )

        logger.info(
            "candles_resampled",
            source_count=len(candles),
            target_count=len(result),
            target_timeframe=target_timeframe,
        )
        return result
