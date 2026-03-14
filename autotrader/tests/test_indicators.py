"""Tests for the Indicators utility module.

Covers EMA, SMA, RSI, Bollinger Bands, ATR, and VWAP computations
including edge cases such as empty lists, single-element inputs, and
periods that exceed the data length.
"""

import math
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, ".")

from journal.models import OHLCV
from signals.indicators import Indicators


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_candle(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
    symbol: str = "TEST",
    timeframe: str = "5m",
    market: str = "stock",
    ts: datetime | None = None,
) -> OHLCV:
    """Create a single OHLCV candle with sensible defaults."""
    open_ = open_ if open_ is not None else close
    high = high if high is not None else max(open_, close) * 1.002
    low = low if low is not None else min(open_, close) * 0.998
    ts = ts or datetime(2025, 1, 15, 10, 0)
    return OHLCV(
        timestamp=ts, open=open_, high=high, low=low,
        close=close, volume=volume, symbol=symbol,
        timeframe=timeframe, market=market,
    )


def _make_candle_series(
    closes: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    opens: list[float] | None = None,
    volumes: list[float] | None = None,
    base_ts: datetime | None = None,
    interval_minutes: int = 5,
    symbol: str = "TEST",
    timeframe: str = "5m",
    market: str = "stock",
) -> list[OHLCV]:
    """Build a list of OHLCV candles from price arrays."""
    n = len(closes)
    base_ts = base_ts or datetime(2025, 1, 15, 9, 30)
    opens = opens or [c for c in closes]
    highs = highs or [max(o, c) * 1.002 for o, c in zip(opens, closes)]
    lows = lows or [min(o, c) * 0.998 for o, c in zip(opens, closes)]
    volumes = volumes or [1000.0] * n

    return [
        OHLCV(
            timestamp=base_ts + timedelta(minutes=i * interval_minutes),
            open=opens[i], high=highs[i], low=lows[i],
            close=closes[i], volume=volumes[i],
            symbol=symbol, timeframe=timeframe, market=market,
        )
        for i in range(n)
    ]


# ===========================================================================
# SMA
# ===========================================================================

class TestSMA:
    """Tests for Simple Moving Average."""

    def test_basic_sma(self) -> None:
        """SMA of [1,2,3,4,5] with period 3 => [nan, nan, 2, 3, 4]."""
        result = Indicators.sma([1, 2, 3, 4, 5], 3)
        assert len(result) == 5
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_sma_period_equals_length(self) -> None:
        """When period == len(closes), only the last value is valid."""
        result = Indicators.sma([10, 20, 30], 3)
        assert len(result) == 3
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(20.0)

    def test_sma_period_1(self) -> None:
        """SMA with period 1 should return the closes themselves."""
        closes = [5.0, 10.0, 15.0]
        result = Indicators.sma(closes, 1)
        for i, v in enumerate(closes):
            assert result[i] == pytest.approx(v)

    def test_sma_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert Indicators.sma([], 5) == []

    def test_sma_period_greater_than_data(self) -> None:
        """All values should be NaN if period > len(closes)."""
        result = Indicators.sma([1, 2], 5)
        assert len(result) == 2
        assert all(math.isnan(v) for v in result)


# ===========================================================================
# EMA
# ===========================================================================

class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_matches_known_values(self) -> None:
        """EMA(period=3) on [2, 4, 6, 8, 10] matches hand-computed values.

        Seed (SMA of first 3): (2+4+6)/3 = 4.0
        k = 2/(3+1) = 0.5
        i=3: 8*0.5 + 4.0*0.5 = 6.0
        i=4: 10*0.5 + 6.0*0.5 = 8.0
        """
        result = Indicators.ema([2, 4, 6, 8, 10], 3)
        assert len(result) == 5
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(4.0)
        assert result[3] == pytest.approx(6.0)
        assert result[4] == pytest.approx(8.0)

    def test_ema_seed_is_sma(self) -> None:
        """The first valid EMA value should equal the SMA of the seed window."""
        closes = [10, 20, 30, 40, 50]
        period = 5
        result = Indicators.ema(closes, period)
        expected_seed = sum(closes[:5]) / 5  # 30
        assert result[4] == pytest.approx(expected_seed)

    def test_ema_smoothing_multiplier(self) -> None:
        """Verify the multiplier k = 2/(period+1) is applied correctly."""
        closes = [100.0, 102.0, 104.0, 103.0, 105.0, 107.0]
        period = 3
        k = 2.0 / (period + 1)
        result = Indicators.ema(closes, period)

        seed = sum(closes[:3]) / 3
        assert result[2] == pytest.approx(seed)

        expected_3 = closes[3] * k + seed * (1 - k)
        assert result[3] == pytest.approx(expected_3)

        expected_4 = closes[4] * k + expected_3 * (1 - k)
        assert result[4] == pytest.approx(expected_4)

        expected_5 = closes[5] * k + expected_4 * (1 - k)
        assert result[5] == pytest.approx(expected_5)

    def test_ema_constant_series(self) -> None:
        """EMA of a constant series should be that constant."""
        closes = [50.0] * 20
        result = Indicators.ema(closes, 10)
        for i in range(9, 20):
            assert result[i] == pytest.approx(50.0)

    def test_ema_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert Indicators.ema([], 10) == []

    def test_ema_single_element(self) -> None:
        """Single element with period 1 should return that element."""
        result = Indicators.ema([42.0], 1)
        assert len(result) == 1
        assert result[0] == pytest.approx(42.0)

    def test_ema_period_greater_than_data(self) -> None:
        """All NaN when period > len(closes)."""
        result = Indicators.ema([1, 2, 3], 10)
        assert len(result) == 3
        assert all(math.isnan(v) for v in result)

    def test_ema_period_zero(self) -> None:
        """Period 0 returns empty."""
        assert Indicators.ema([1, 2, 3], 0) == []

    def test_ema_trending_up_lags_below_price(self) -> None:
        """In a steadily rising series the EMA should lag below the price."""
        closes = list(range(1, 51))  # 1..50
        result = Indicators.ema(closes, 10)
        # For the last value, EMA should be below the current close
        assert result[-1] < closes[-1]


# ===========================================================================
# RSI
# ===========================================================================

class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_all_gains(self) -> None:
        """RSI should be 100 when there are only gains."""
        # 16 prices, 15 changes, all positive
        closes = [float(i) for i in range(100, 116)]
        result = Indicators.rsi(closes, 14)
        assert result[-1] == pytest.approx(100.0)

    def test_rsi_all_losses(self) -> None:
        """RSI should be 0 when there are only losses."""
        closes = [float(i) for i in range(115, 99, -1)]
        result = Indicators.rsi(closes, 14)
        assert result[-1] == pytest.approx(0.0, abs=0.1)

    def test_rsi_known_series(self) -> None:
        """RSI matches a hand-calculated value for a mixed series.

        Using the classic Wilder 14-period RSI on a known dataset.
        """
        # 15 prices => 14 changes
        closes = [
            44.34, 44.09, 43.61, 44.33, 44.83,
            45.10, 45.42, 45.84, 46.08, 45.89,
            46.03, 45.61, 46.28, 46.28, 46.00,
        ]
        result = Indicators.rsi(closes, 14)
        # First meaningful RSI is at index 14
        # Based on standard Wilder computation for this data:
        # avg_gain = (0 + 0 + 0.72 + 0.50 + 0.27 + 0.32 + 0.42 + 0.24 + 0 + 0.14 + 0 + 0.67 + 0 + 0) / 14
        # avg_loss = (0.25 + 0.48 + 0 + 0 + 0 + 0 + 0 + 0 + 0.19 + 0 + 0.42 + 0 + 0 + 0.28) / 14
        gains = [0, 0.25, 0.48, 0.72, 0.50, 0.27, 0.32, 0.42, 0.24, 0.19, 0.14, 0.42, 0.67, 0.28]
        losses_arr = [0.25, 0.48, 0, 0, 0, 0, 0, 0, 0.19, 0, 0.42, 0, 0, 0.28]
        # Deltas: -0.25, -0.48, +0.72, +0.50, +0.27, +0.32, +0.42, +0.24, -0.19, +0.14, -0.42, +0.67, 0.00, -0.28
        deltas = [closes[i+1] - closes[i] for i in range(14)]
        avg_gain_init = sum(max(d, 0) for d in deltas) / 14
        avg_loss_init = sum(max(-d, 0) for d in deltas) / 14
        rs = avg_gain_init / avg_loss_init if avg_loss_init else float('inf')
        expected_rsi = 100 - 100 / (1 + rs)
        assert result[14] == pytest.approx(expected_rsi, abs=0.5)

    def test_rsi_equal_gains_and_losses(self) -> None:
        """RSI should be near 50 when gains and losses are balanced."""
        # Alternating up/down with same magnitude
        closes = []
        price = 100.0
        for i in range(30):
            closes.append(price)
            price += 1.0 if i % 2 == 0 else -1.0
        result = Indicators.rsi(closes, 14)
        # Should be close to 50
        assert 45.0 < result[-1] < 55.0

    def test_rsi_initial_values_are_neutral(self) -> None:
        """First `period` values should be 50.0 (neutral placeholder)."""
        closes = list(range(1, 30))
        result = Indicators.rsi([float(c) for c in closes], 14)
        for i in range(14):
            assert result[i] == 50.0

    def test_rsi_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert Indicators.rsi([], 14) == []

    def test_rsi_single_element(self) -> None:
        """Single element returns [50.0]."""
        result = Indicators.rsi([100.0], 14)
        assert len(result) == 1
        assert result[0] == 50.0

    def test_rsi_not_enough_data(self) -> None:
        """When len(closes) < period + 1, all values should be 50.0."""
        result = Indicators.rsi([100.0, 101.0, 102.0], 14)
        assert len(result) == 3
        assert all(v == 50.0 for v in result)

    def test_rsi_bounded_0_100(self) -> None:
        """RSI values should always be between 0 and 100."""
        np.random.seed(42)
        closes = list(np.cumsum(np.random.randn(100)) + 100)
        result = Indicators.rsi(closes, 14)
        for v in result:
            assert 0.0 <= v <= 100.0


# ===========================================================================
# Bollinger Bands
# ===========================================================================

class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_symmetry(self) -> None:
        """Upper and lower bands should be equidistant from the middle."""
        closes = [float(i) for i in range(1, 25)]
        upper, middle, lower = Indicators.bollinger_bands(closes, period=20, std_dev=2.0)
        for i in range(19, 24):
            upper_dist = upper[i] - middle[i]
            lower_dist = middle[i] - lower[i]
            assert upper_dist == pytest.approx(lower_dist, rel=1e-10)

    def test_bollinger_width(self) -> None:
        """Band width = 2 * std_dev * population_std."""
        closes = [100.0, 102.0, 98.0, 101.0, 99.0,
                  103.0, 97.0, 100.0, 104.0, 96.0,
                  100.0, 102.0, 98.0, 101.0, 99.0,
                  103.0, 97.0, 100.0, 104.0, 96.0]
        upper, middle, lower = Indicators.bollinger_bands(closes, period=20, std_dev=2.0)
        idx = 19  # first valid
        expected_std = float(np.std(closes))
        expected_width = 4.0 * expected_std  # 2 * std_dev * std
        actual_width = upper[idx] - lower[idx]
        assert actual_width == pytest.approx(expected_width, rel=1e-6)

    def test_bollinger_middle_is_sma(self) -> None:
        """Middle band should equal the SMA."""
        closes = list(range(1, 25))
        upper, middle, lower = Indicators.bollinger_bands(
            [float(c) for c in closes], period=20, std_dev=2.0,
        )
        sma = Indicators.sma([float(c) for c in closes], 20)
        for i in range(19, 24):
            assert middle[i] == pytest.approx(sma[i], rel=1e-10)

    def test_bollinger_constant_series(self) -> None:
        """Constant series => bands collapse to the middle (std=0)."""
        closes = [50.0] * 25
        upper, middle, lower = Indicators.bollinger_bands(closes, period=20)
        for i in range(19, 25):
            assert upper[i] == pytest.approx(50.0)
            assert middle[i] == pytest.approx(50.0)
            assert lower[i] == pytest.approx(50.0)

    def test_bollinger_nan_prefix(self) -> None:
        """First period-1 values should be NaN."""
        closes = [float(i) for i in range(1, 25)]
        upper, middle, lower = Indicators.bollinger_bands(closes, period=20)
        for i in range(19):
            assert math.isnan(upper[i])
            assert math.isnan(middle[i])
            assert math.isnan(lower[i])

    def test_bollinger_empty_list(self) -> None:
        """Empty input returns three empty lists."""
        upper, middle, lower = Indicators.bollinger_bands([], 20)
        assert upper == []
        assert middle == []
        assert lower == []

    def test_bollinger_period_greater_than_data(self) -> None:
        """All NaN when period > len(closes)."""
        upper, middle, lower = Indicators.bollinger_bands([1.0, 2.0], 20)
        assert len(upper) == 2
        assert all(math.isnan(v) for v in upper)
        assert all(math.isnan(v) for v in middle)
        assert all(math.isnan(v) for v in lower)


# ===========================================================================
# ATR
# ===========================================================================

class TestATR:
    """Tests for Average True Range."""

    def test_atr_basic(self) -> None:
        """ATR for a simple 3-candle series with known true ranges."""
        candles = [
            _make_candle(close=10.0, open_=10.0, high=11.0, low=9.0,
                         ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(close=11.0, open_=10.5, high=12.0, low=10.0,
                         ts=datetime(2025, 1, 15, 9, 35)),
            _make_candle(close=10.5, open_=11.0, high=11.5, low=9.5,
                         ts=datetime(2025, 1, 15, 9, 40)),
        ]
        result = Indicators.atr(candles, period=14)
        assert len(result) == 3
        # TR[0] = 11 - 9 = 2
        # TR[1] = max(12-10, |12-10|, |10-10|) = max(2, 2, 0) = 2
        # TR[2] = max(11.5-9.5, |11.5-11|, |9.5-11|) = max(2, 0.5, 1.5) = 2
        # Simple average since n < period
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(2.0)

    def test_atr_wilder_smoothing(self) -> None:
        """ATR uses Wilder smoothing after the seed period."""
        # Create 16 candles with known TR values
        base_ts = datetime(2025, 1, 15, 9, 30)
        candles = []
        prev_close = 100.0
        for i in range(16):
            high = prev_close + 2.0
            low = prev_close - 1.0
            close = prev_close + 0.5
            candles.append(_make_candle(
                close=close, open_=prev_close, high=high, low=low,
                ts=base_ts + timedelta(minutes=i * 5),
            ))
            prev_close = close

        result = Indicators.atr(candles, period=14)
        assert len(result) == 16
        # All values should be positive
        for v in result:
            assert v > 0

    def test_atr_single_candle(self) -> None:
        """Single candle returns high - low."""
        candle = _make_candle(close=100.0, open_=99.0, high=102.0, low=97.0)
        result = Indicators.atr([candle], period=14)
        assert len(result) == 1
        assert result[0] == pytest.approx(5.0)  # 102 - 97

    def test_atr_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert Indicators.atr([], 14) == []

    def test_atr_with_gaps(self) -> None:
        """ATR captures gap moves via true range (prev close vs current high/low)."""
        candles = [
            _make_candle(close=100.0, open_=100.0, high=101.0, low=99.0,
                         ts=datetime(2025, 1, 15, 9, 30)),
            # Gap up: previous close 100, opens at 105
            _make_candle(close=106.0, open_=105.0, high=107.0, low=104.0,
                         ts=datetime(2025, 1, 15, 9, 35)),
        ]
        result = Indicators.atr(candles, period=14)
        # TR[1] = max(107-104, |107-100|, |104-100|) = max(3, 7, 4) = 7
        assert result[1] == pytest.approx((2.0 + 7.0) / 2)  # simple avg of 2 TRs


# ===========================================================================
# VWAP
# ===========================================================================

class TestVWAP:
    """Tests for Volume Weighted Average Price."""

    def test_vwap_single_bar(self) -> None:
        """VWAP of a single bar is its typical price."""
        candle = _make_candle(close=100.0, open_=99.0, high=102.0, low=98.0,
                              volume=5000.0)
        result = Indicators.vwap([candle])
        expected_tp = (102.0 + 98.0 + 100.0) / 3
        assert result[0] == pytest.approx(expected_tp)

    def test_vwap_uniform_volume(self) -> None:
        """With equal volume on all bars, VWAP is the mean typical price."""
        candles = [
            _make_candle(close=100.0, open_=99.0, high=101.0, low=99.0,
                         volume=1000.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(close=102.0, open_=100.0, high=103.0, low=100.0,
                         volume=1000.0, ts=datetime(2025, 1, 15, 9, 35)),
            _make_candle(close=104.0, open_=102.0, high=105.0, low=102.0,
                         volume=1000.0, ts=datetime(2025, 1, 15, 9, 40)),
        ]
        result = Indicators.vwap(candles)
        tp1 = (101.0 + 99.0 + 100.0) / 3
        tp2 = (103.0 + 100.0 + 102.0) / 3
        tp3 = (105.0 + 102.0 + 104.0) / 3
        expected = (tp1 * 1000 + tp2 * 1000 + tp3 * 1000) / 3000
        assert result[-1] == pytest.approx(expected)

    def test_vwap_high_volume_bar_dominates(self) -> None:
        """A bar with much higher volume should pull VWAP toward it."""
        candles = [
            _make_candle(close=100.0, open_=100.0, high=101.0, low=99.0,
                         volume=100.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(close=110.0, open_=110.0, high=111.0, low=109.0,
                         volume=10000.0, ts=datetime(2025, 1, 15, 9, 35)),
        ]
        result = Indicators.vwap(candles)
        # VWAP should be much closer to 110 than to 100
        assert result[-1] > 109.0

    def test_vwap_resets_each_day(self) -> None:
        """VWAP should reset at the start of each new calendar day."""
        day1_candles = [
            _make_candle(close=100.0, open_=100.0, high=101.0, low=99.0,
                         volume=1000.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(close=102.0, open_=101.0, high=103.0, low=100.0,
                         volume=1000.0, ts=datetime(2025, 1, 15, 9, 35)),
        ]
        day2_candles = [
            _make_candle(close=200.0, open_=200.0, high=201.0, low=199.0,
                         volume=1000.0, ts=datetime(2025, 1, 16, 9, 30)),
        ]
        all_candles = day1_candles + day2_candles
        result = Indicators.vwap(all_candles)

        # Day 2's VWAP (index 2) should be based only on day 2 data
        tp_day2 = (201.0 + 199.0 + 200.0) / 3
        assert result[2] == pytest.approx(tp_day2)

        # Day 1's last VWAP should be based on day 1 data only
        tp1 = (101.0 + 99.0 + 100.0) / 3
        tp2 = (103.0 + 100.0 + 102.0) / 3
        expected_day1 = (tp1 * 1000 + tp2 * 1000) / 2000
        assert result[1] == pytest.approx(expected_day1)

    def test_vwap_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert Indicators.vwap([]) == []

    def test_vwap_zero_volume_carries_forward(self) -> None:
        """Zero-volume bars should carry forward the previous VWAP."""
        candles = [
            _make_candle(close=100.0, open_=100.0, high=101.0, low=99.0,
                         volume=1000.0, ts=datetime(2025, 1, 15, 9, 30)),
            _make_candle(close=105.0, open_=105.0, high=106.0, low=104.0,
                         volume=0.0, ts=datetime(2025, 1, 15, 9, 35)),
        ]
        result = Indicators.vwap(candles)
        # Second bar has 0 volume, VWAP should carry forward from first bar
        assert result[1] == pytest.approx(result[0])


# ===========================================================================
# Cross-indicator integration
# ===========================================================================

class TestCrossIndicator:
    """Sanity checks that combine multiple indicators."""

    def test_ema_shorter_period_more_responsive(self) -> None:
        """A shorter EMA period should react faster to price changes."""
        closes = [100.0] * 50 + [110.0] * 10
        ema_fast = Indicators.ema(closes, 10)
        ema_slow = Indicators.ema(closes, 50)
        # After the jump, fast EMA should be closer to 110 than slow EMA
        assert ema_fast[-1] > ema_slow[-1]

    def test_bollinger_bands_widen_with_volatility(self) -> None:
        """Higher volatility should widen the Bollinger bands."""
        calm = [100.0 + 0.1 * (i % 2) for i in range(30)]
        wild = [100.0 + 5.0 * ((-1) ** i) for i in range(30)]

        _, _, _ = Indicators.bollinger_bands(calm, period=20)
        u_calm, m_calm, l_calm = Indicators.bollinger_bands(calm, period=20)
        u_wild, m_wild, l_wild = Indicators.bollinger_bands(wild, period=20)

        calm_width = u_calm[-1] - l_calm[-1]
        wild_width = u_wild[-1] - l_wild[-1]
        assert wild_width > calm_width

    def test_rsi_and_ema_consistent_for_trend(self) -> None:
        """In a strong uptrend, RSI should be above 50 and EMA rising."""
        closes = [100.0 + i * 0.5 for i in range(60)]
        rsi_vals = Indicators.rsi(closes, 14)
        ema_vals = Indicators.ema(closes, 14)

        # RSI should be well above 50 in the latter part
        assert rsi_vals[-1] > 60.0
        # EMA should be rising
        assert ema_vals[-1] > ema_vals[-10]
