"""Unified OHLCV normalizer for multiple market-data sources."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from journal.models import OHLCV


class DataNormalizer:
    """Normalizes raw market data from various sources into unified OHLCV format.

    Supports yfinance (stocks), CCXT (crypto), and Polygon.io (stocks/crypto).
    Every static method returns a single ``OHLCV`` dataclass instance so that
    downstream consumers never need to know which upstream provider was used.
    """

    # ------------------------------------------------------------------
    # yfinance
    # ------------------------------------------------------------------
    @staticmethod
    def from_yfinance(row: dict[str, Any], symbol: str, timeframe: str) -> OHLCV:
        """Convert a yfinance bar dict to OHLCV.

        Parameters
        ----------
        row:
            A dictionary with keys ``Open``, ``High``, ``Low``, ``Close``,
            ``Volume``, and an optional ``Datetime`` or ``Date`` key holding
            the bar timestamp.  When the row originates from a
            ``pandas.DataFrame`` produced by ``yf.Ticker.history()``, the
            timestamp is typically the index; callers should add it to the
            dict before calling this method.
        symbol:
            Ticker symbol (e.g. ``"AAPL"``).
        timeframe:
            Bar size such as ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``.

        Returns
        -------
        OHLCV
            Normalised bar with ``market="stock"``.
        """
        ts_raw = row.get("Datetime") or row.get("Date") or row.get("timestamp")
        if isinstance(ts_raw, str):
            ts = datetime.fromisoformat(ts_raw)
        elif isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            # pandas Timestamp or similar -- convert via isoformat round-trip
            ts = datetime.fromisoformat(str(ts_raw))

        # Ensure timezone-aware (UTC)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        vwap: float | None = None
        if "VWAP" in row and row["VWAP"] is not None:
            try:
                vwap = float(row["VWAP"])
            except (TypeError, ValueError):
                vwap = None

        bar = OHLCV(
            timestamp=ts,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
            symbol=symbol.upper(),
            timeframe=timeframe,
            market="stock",
            vwap=vwap,
        )
        logger.debug(
            "normalised_yfinance_bar | symbol={symbol} ts={ts}",
            symbol=bar.symbol,
            ts=bar.timestamp.isoformat(),
        )
        return bar

    # ------------------------------------------------------------------
    # CCXT
    # ------------------------------------------------------------------
    @staticmethod
    def from_ccxt(bar: list[float | int], symbol: str, timeframe: str) -> OHLCV:
        """Convert a CCXT OHLCV list ``[timestamp, o, h, l, c, v]`` to OHLCV.

        Parameters
        ----------
        bar:
            Six-element list returned by ``exchange.fetch_ohlcv()``.
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).
        timeframe:
            Bar size such as ``"1m"``, ``"5m"``, ``"1h"``.

        Returns
        -------
        OHLCV
            Normalised bar with ``market="crypto"``.
        """
        ts = datetime.fromtimestamp(bar[0] / 1000.0, tz=timezone.utc)

        ohlcv = OHLCV(
            timestamp=ts,
            open=float(bar[1]),
            high=float(bar[2]),
            low=float(bar[3]),
            close=float(bar[4]),
            volume=float(bar[5]),
            symbol=symbol.upper(),
            timeframe=timeframe,
            market="crypto",
        )
        logger.debug(
            "normalised_ccxt_bar | symbol={symbol} ts={ts}",
            symbol=ohlcv.symbol,
            ts=ohlcv.timestamp.isoformat(),
        )
        return ohlcv

    # ------------------------------------------------------------------
    # Polygon.io
    # ------------------------------------------------------------------
    @staticmethod
    def from_polygon(bar: dict[str, Any], symbol: str, timeframe: str) -> OHLCV:
        """Convert a Polygon.io bar dict to OHLCV.

        Parameters
        ----------
        bar:
            Dictionary with Polygon aggregate keys:
            ``t`` (Unix-ms timestamp), ``o``, ``h``, ``l``, ``c``, ``v``,
            and optional ``vw`` (VWAP).
        symbol:
            Ticker symbol (e.g. ``"AAPL"``).
        timeframe:
            Bar size such as ``"1m"``, ``"5m"``, ``"1d"``.

        Returns
        -------
        OHLCV
            Normalised bar with ``market="stock"``.
        """
        ts = datetime.fromtimestamp(bar["t"] / 1000.0, tz=timezone.utc)

        vwap: float | None = None
        if "vw" in bar and bar["vw"] is not None:
            try:
                vwap = float(bar["vw"])
            except (TypeError, ValueError):
                vwap = None

        ohlcv = OHLCV(
            timestamp=ts,
            open=float(bar["o"]),
            high=float(bar["h"]),
            low=float(bar["l"]),
            close=float(bar["c"]),
            volume=float(bar["v"]),
            symbol=symbol.upper(),
            timeframe=timeframe,
            market="stock",
            vwap=vwap,
        )
        logger.debug(
            "normalised_polygon_bar | symbol={symbol} ts={ts}",
            symbol=ohlcv.symbol,
            ts=ohlcv.timestamp.isoformat(),
        )
        return ohlcv
