"""Technical indicator calculations for signal evaluation.

Provides reusable, stateless indicator functions used by signal strategies
and the main AutoTrader orchestrator.  All functions operate on plain
Python lists and return lists of the same length as the input so that
indexing is straightforward (``result[-1]`` is the most recent value).
"""

from __future__ import annotations

from datetime import date
import math
from math import isnan

import numpy as np
from loguru import logger

from journal.models import OHLCV


class Indicators:
    """Stateless technical indicator calculator."""

    # ------------------------------------------------------------------
    # SMA
    # ------------------------------------------------------------------
    @staticmethod
    def sma(closes: list[float], period: int) -> list[float]:
        """Compute Simple Moving Average.

        Parameters
        ----------
        closes:
            List of closing prices, oldest first.
        period:
            Look-back window length.

        Returns
        -------
        list[float]
            SMA series of the same length as *closes*.  The first
            ``period - 1`` values are ``NaN`` because insufficient data
            is available.
        """
        if not closes or period <= 0:
            return []
        n = len(closes)
        if period > n:
            return [float("nan")] * n

        result: list[float] = [float("nan")] * (period - 1)
        arr = np.array(closes, dtype=np.float64)
        cumsum = np.cumsum(arr)
        # cumsum[i] = sum(arr[0..i])
        # SMA at index i (for i >= period-1) = (cumsum[i] - cumsum[i-period]) / period
        # except for i == period-1 where it's just cumsum[period-1] / period
        sma_vals = np.empty(n - period + 1, dtype=np.float64)
        sma_vals[0] = cumsum[period - 1] / period
        sma_vals[1:] = (cumsum[period:] - cumsum[:-period]) / period
        result.extend(sma_vals.tolist())
        return result

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    @staticmethod
    def ema(closes: list[float], period: int) -> list[float]:
        """Compute Exponential Moving Average for the full series.

        The first ``period`` values are seeded with the SMA of the first
        *period* closing prices.  Subsequent values use the standard
        smoothing multiplier ``k = 2 / (period + 1)``.

        Parameters
        ----------
        closes:
            List of closing prices, oldest first.
        period:
            EMA look-back period.

        Returns
        -------
        list[float]
            EMA series of the same length as *closes*.  The first
            ``period - 1`` values are ``NaN`` (insufficient data).
        """
        if not closes or period <= 0:
            return []
        n = len(closes)
        if period > n:
            return [float("nan")] * n

        k = 2.0 / (period + 1)
        result = [float("nan")] * n

        # Seed: SMA of the first `period` values
        seed = sum(closes[:period]) / period
        result[period - 1] = seed

        # Recursive EMA
        prev = seed
        for i in range(period, n):
            ema_val = closes[i] * k + prev * (1.0 - k)
            result[i] = ema_val
            prev = ema_val

        return result

    # ------------------------------------------------------------------
    # RSI  (Wilder's smoothing)
    # ------------------------------------------------------------------
    @staticmethod
    def rsi(closes: list[float], period: int = 14) -> list[float]:
        """Compute RSI using Wilder's smoothing method.

        Parameters
        ----------
        closes:
            List of closing prices, oldest first.
        period:
            RSI look-back period (default 14).

        Returns
        -------
        list[float]
            RSI series of the same length as *closes*.  The first
            *period* values are ``50.0`` (neutral placeholder) because
            we need *period* price changes (i.e. ``period + 1`` prices)
            to produce the first meaningful RSI.
        """
        if not closes or period <= 0:
            return []
        n = len(closes)
        if n < 2:
            return [50.0] * n

        result = [50.0] * n

        # Price changes
        deltas = np.diff(np.array(closes, dtype=np.float64))  # length n-1
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        if len(deltas) < period:
            # Not enough data for a single RSI value
            return result

        # Initial average gain / loss (simple mean of first `period` changes)
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        # First RSI value at index `period`
        if avg_loss == 0.0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))

        # Wilder's smoothing for subsequent values
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0.0:
                result[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return result

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------
    @staticmethod
    def bollinger_bands(
        closes: list[float],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[list[float], list[float], list[float]]:
        """Compute Bollinger Bands.

        Parameters
        ----------
        closes:
            List of closing prices, oldest first.
        period:
            Moving average window (default 20).
        std_dev:
            Number of standard deviations for the upper / lower bands
            (default 2.0).

        Returns
        -------
        tuple[list[float], list[float], list[float]]
            ``(upper, middle, lower)`` each as ``list[float]`` of the same
            length as *closes*.  The first ``period - 1`` values are ``NaN``.
        """
        if not closes or period <= 0:
            return ([], [], [])
        n = len(closes)
        if period > n:
            nans = [float("nan")] * n
            return (list(nans), list(nans), list(nans))

        arr = np.array(closes, dtype=np.float64)
        upper = [float("nan")] * n
        middle = [float("nan")] * n
        lower = [float("nan")] * n

        for i in range(period - 1, n):
            window = arr[i - period + 1 : i + 1]
            ma = float(np.mean(window))
            sd = float(np.std(window, ddof=0))  # population std, standard for BB
            middle[i] = ma
            upper[i] = ma + std_dev * sd
            lower[i] = ma - std_dev * sd

        return (upper, middle, lower)

    # ------------------------------------------------------------------
    # ATR  (Average True Range)
    # ------------------------------------------------------------------
    @staticmethod
    def atr(candles: list[OHLCV], period: int = 14) -> list[float]:
        """Compute Average True Range.

        Uses Wilder's smoothing (same as RSI) for the running average.

        Parameters
        ----------
        candles:
            List of OHLCV bars, oldest first.
        period:
            ATR look-back period (default 14).

        Returns
        -------
        list[float]
            ATR series of the same length as *candles*.  The first
            *period* values use a simple average of available true ranges
            (the very first bar has TR = high - low because there is no
            previous close).
        """
        if not candles or period <= 0:
            return []
        n = len(candles)
        if n == 1:
            return [candles[0].high - candles[0].low]

        # True ranges
        tr = [0.0] * n
        tr[0] = candles[0].high - candles[0].low

        for i in range(1, n):
            high_low = candles[i].high - candles[i].low
            high_prev_close = abs(candles[i].high - candles[i - 1].close)
            low_prev_close = abs(candles[i].low - candles[i - 1].close)
            tr[i] = max(high_low, high_prev_close, low_prev_close)

        result = [0.0] * n

        if n <= period:
            # Not enough data for a full-period ATR; use simple running avg
            running = 0.0
            for i in range(n):
                running += tr[i]
                result[i] = running / (i + 1)
            return result

        # Seed with SMA of the first `period` true ranges
        seed = sum(tr[:period]) / period
        # Fill early values with simple running averages
        running = 0.0
        for i in range(period):
            running += tr[i]
            result[i] = running / (i + 1)
        result[period - 1] = seed

        # Wilder's smoothing
        prev_atr = seed
        for i in range(period, n):
            atr_val = (prev_atr * (period - 1) + tr[i]) / period
            result[i] = atr_val
            prev_atr = atr_val

        return result

    # ------------------------------------------------------------------
    # ADX  (Average Directional Index)
    # ------------------------------------------------------------------
    @staticmethod
    def adx(candles: list[OHLCV], period: int = 14) -> list[float]:
        """Compute Average Directional Index.

        Uses Wilder's smoothing for +DI, -DI, and ADX.

        Parameters
        ----------
        candles:
            List of OHLCV bars, oldest first.
        period:
            ADX look-back period (default 14).

        Returns
        -------
        list[float]
            ADX series of the same length as *candles*.  The first
            ``2 * period`` values are approximate (not enough data for
            a fully warmed-up ADX) and default to 0.0.
        """
        if not candles or period <= 0:
            return []
        n = len(candles)
        if n < 2:
            return [0.0] * n

        # True Range, +DM, -DM
        tr = [0.0] * n
        plus_dm = [0.0] * n
        minus_dm = [0.0] * n

        for i in range(1, n):
            high_low = candles[i].high - candles[i].low
            high_prev = abs(candles[i].high - candles[i - 1].close)
            low_prev = abs(candles[i].low - candles[i - 1].close)
            tr[i] = max(high_low, high_prev, low_prev)

            up_move = candles[i].high - candles[i - 1].high
            down_move = candles[i - 1].low - candles[i].low

            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0

        result = [0.0] * n
        if n < period + 1:
            return result

        # Seed smoothed TR, +DM, -DM with sum of first period values
        smooth_tr = sum(tr[1 : period + 1])
        smooth_plus = sum(plus_dm[1 : period + 1])
        smooth_minus = sum(minus_dm[1 : period + 1])

        # First DI values
        plus_di = 100.0 * smooth_plus / smooth_tr if smooth_tr else 0.0
        minus_di = 100.0 * smooth_minus / smooth_tr if smooth_tr else 0.0
        di_sum = plus_di + minus_di
        dx_vals: list[float] = []
        if di_sum:
            dx_vals.append(100.0 * abs(plus_di - minus_di) / di_sum)
        else:
            dx_vals.append(0.0)

        # Continue Wilder smoothing
        for i in range(period + 1, n):
            smooth_tr = smooth_tr - smooth_tr / period + tr[i]
            smooth_plus = smooth_plus - smooth_plus / period + plus_dm[i]
            smooth_minus = smooth_minus - smooth_minus / period + minus_dm[i]

            plus_di = 100.0 * smooth_plus / smooth_tr if smooth_tr else 0.0
            minus_di = 100.0 * smooth_minus / smooth_tr if smooth_tr else 0.0
            di_sum = plus_di + minus_di
            if di_sum:
                dx_vals.append(100.0 * abs(plus_di - minus_di) / di_sum)
            else:
                dx_vals.append(0.0)

        # ADX = smoothed average of DX
        if len(dx_vals) < period:
            # Not enough data for ADX; use simple average of available DX
            for i, dx in enumerate(dx_vals):
                result[period + i] = dx
            return result

        # Seed ADX with SMA of first `period` DX values
        adx_val = sum(dx_vals[:period]) / period
        result[2 * period] = adx_val

        for i in range(period, len(dx_vals)):
            adx_val = (adx_val * (period - 1) + dx_vals[i]) / period
            idx = period + i  # map dx_vals index back to candle index
            if idx < n:
                result[idx] = adx_val

        return result

    # ------------------------------------------------------------------
    # VWAP  (Volume Weighted Average Price, intraday, daily reset)
    # ------------------------------------------------------------------
    @staticmethod
    def vwap(candles: list[OHLCV]) -> list[float]:
        """Compute intraday VWAP that resets each calendar day.

        The typical price ``(high + low + close) / 3`` is used as the
        representative price for each bar.

        Parameters
        ----------
        candles:
            List of OHLCV bars, oldest first.

        Returns
        -------
        list[float]
            VWAP series of the same length as *candles*.  If a bar has
            zero volume the VWAP carries forward the previous value (or
            uses the typical price if it is the first bar of the day).
        """
        if not candles:
            return []

        n = len(candles)
        result = [0.0] * n

        cum_tp_vol = 0.0
        cum_vol = 0.0
        current_date: date | None = None

        for i, bar in enumerate(candles):
            bar_date = bar.timestamp.date()

            # Reset accumulators on a new trading day
            if bar_date != current_date:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                current_date = bar_date

            typical_price = (bar.high + bar.low + bar.close) / 3.0
            cum_tp_vol += typical_price * bar.volume
            cum_vol += bar.volume

            if cum_vol > 0:
                result[i] = cum_tp_vol / cum_vol
            else:
                # Zero-volume bar: carry forward or use typical price
                result[i] = result[i - 1] if i > 0 and current_date == candles[i - 1].timestamp.date() else typical_price

        return result

    # ------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------
    @staticmethod
    def macd(
        closes: list[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[list[float], list[float], list[float]]:
        """Compute MACD line, signal line, and histogram.

        Args:
            closes: List of closing prices.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal line EMA period.

        Returns:
            Tuple of (macd_line, signal_line, histogram), each same length as input.
            Early values are NaN where insufficient data exists.
        """
        fast_ema = Indicators.ema(closes, fast)
        slow_ema = Indicators.ema(closes, slow)

        macd_line: list[float] = []
        for f, s in zip(fast_ema, slow_ema):
            if math.isnan(f) or math.isnan(s):
                macd_line.append(float("nan"))
            else:
                macd_line.append(f - s)

        # Signal line = EMA of the MACD line (only over non-NaN values)
        valid_start = slow - 1
        macd_valid = macd_line[valid_start:]
        if len(macd_valid) >= signal:
            signal_ema = Indicators.ema(macd_valid, signal)
            signal_line = [float("nan")] * valid_start + signal_ema
        else:
            signal_line = [float("nan")] * len(closes)

        histogram: list[float] = []
        for m, s in zip(macd_line, signal_line):
            if math.isnan(m) or math.isnan(s):
                histogram.append(float("nan"))
            else:
                histogram.append(m - s)

        return macd_line, signal_line, histogram
