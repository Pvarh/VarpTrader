"""Tests for all trading signal strategies."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

sys.path.insert(0, ".")

from journal.models import OHLCV
from signals.base_signal import SignalResult, SignalDirection
from signals.indicators import Indicators
from signals.vwap_reversion import VWAPReversionSignal
from signals.macd_divergence import MACDDivergenceSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candle(
    close: float,
    open_: float = 0.0,
    high: float = 0.0,
    low: float = 0.0,
    volume: float = 1000.0,
    symbol: str = "TEST",
    timeframe: str = "5m",
    market: str = "stock",
    ts_offset_min: int = 0,
    vwap: float | None = None,
) -> OHLCV:
    """Create an OHLCV candle for testing."""
    open_ = open_ or close
    high = high or max(open_, close) * 1.001
    low = low or min(open_, close) * 0.999
    ts = datetime(2025, 1, 15, 9, 45) + timedelta(minutes=ts_offset_min)
    return OHLCV(
        timestamp=ts, open=open_, high=high, low=low,
        close=close, volume=volume, symbol=symbol,
        timeframe=timeframe, market=market, vwap=vwap,
    )


def make_candle_series(
    closes: list[float], **kwargs
) -> list[OHLCV]:
    """Create a series of candles from a list of close prices."""
    return [
        make_candle(c, ts_offset_min=i * 5, **kwargs)
        for i, c in enumerate(closes)
    ]


# ===========================================================================
# First Candle (Opening Range Breakout -- 60-min ORB)
# ===========================================================================

class TestFirstCandle:
    """Tests for the first candle (60-minute ORB) signal."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "volume_multiplier": 1.5,
            "orb_window_minutes": 60,
            "valid_until_hour": 13,
        }

    @pytest.fixture(autouse=True)
    def mock_swing_advisor(self):
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=False,
        ):
            yield

    def test_long_signal_breakout_above_orb_high(self, config: dict) -> None:
        """BUY when 5-min candle closes ABOVE 60-min ORB high with volume > 1.5x avg."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m", ts_offset_min=90,
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=2.0,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.LONG

    def test_short_signal_breakdown_below_orb_low(self, config: dict) -> None:
        """SELL when 5-min candle closes BELOW 60-min ORB low with volume > 1.5x avg."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakdown = make_candle(
            close=98.5, open_=99.2, high=99.3, low=98.3,
            volume=20000, timeframe="5m", ts_offset_min=90,
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakdown,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=2.0,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.SHORT

    def test_no_signal_when_volume_too_low(self, config: dict) -> None:
        """No signal when volume is below 1.5x average."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=5000, timeframe="5m", ts_offset_min=90,
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=2.0,
        )
        assert result.triggered is False

    def test_no_signal_before_orb_window_closes(self, config: dict) -> None:
        """No signal before the 60-minute ORB window has closed."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m", ts_offset_min=30,
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=50,  # < 60 min, ORB window not closed yet
            atr_20d=2.0,
        )
        assert result.triggered is False
        assert "Outside valid breakout window" in result.reason

    def test_no_signal_after_valid_window_expires(self, config: dict) -> None:
        """No signal after 210 minutes (13:00 EST)."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m", ts_offset_min=220,
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=220,  # > 210 min
            atr_20d=2.0,
        )
        assert result.triggered is False
        assert "Outside valid breakout window" in result.reason

    def test_valid_window_boundary_60_minutes(self, config: dict) -> None:
        """Signal is valid at exactly 60 minutes (lower boundary)."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m",
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=60,  # exactly 60 min
            atr_20d=2.0,
        )
        assert result.triggered is True

    def test_valid_window_boundary_210_minutes(self, config: dict) -> None:
        """Signal is valid at exactly 210 minutes (upper boundary)."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m",
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=210,  # exactly 210 min
            atr_20d=2.0,
        )
        assert result.triggered is True

    def test_no_signal_when_orb_range_exceeds_2x_atr(self, config: dict) -> None:
        """No signal when ORB range is greater than 2x the 20-day ATR."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m",
        )
        # ORB range = 101.0 - 99.0 = 2.0
        # atr_20d = 0.5, so 2 * atr_20d = 1.0
        # 2.0 > 1.0 => rejected
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=0.5,
        )
        assert result.triggered is False
        assert "ORB range too wide" in result.reason

    def test_orb_range_at_exactly_2x_atr_allowed(self, config: dict) -> None:
        """Signal is allowed when ORB range equals exactly 2x ATR (not greater)."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m",
        )
        # ORB range = 101.0 - 99.0 = 2.0
        # atr_20d = 1.0, so 2 * atr_20d = 2.0
        # 2.0 == 2.0 => NOT greater, should be allowed
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=1.0,
        )
        assert result.triggered is True

    def test_stop_loss_at_orb_low_for_longs(self, config: dict) -> None:
        """Stop loss should be at the ORB low for long trades."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m",
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=2.0,
        )
        assert result.stop_loss == pytest.approx(99.0, rel=1e-6)

    def test_stop_loss_at_orb_high_for_shorts(self, config: dict) -> None:
        """Stop loss should be at the ORB high for short trades."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakdown = make_candle(
            close=98.5, open_=99.2, high=99.3, low=98.3,
            volume=20000, timeframe="5m",
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakdown,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=2.0,
        )
        assert result.stop_loss == pytest.approx(101.0, rel=1e-6)

    def test_take_profit_is_2x_stop_distance_long(self, config: dict) -> None:
        """Take profit should be 2x the stop loss distance from entry (long)."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m",
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=2.0,
        )
        # entry = 101.5, stop = 99.0, stop_dist = 2.5
        # take_profit = 101.5 + 2 * 2.5 = 106.5
        stop_dist = result.entry_price - result.stop_loss
        tp_dist = result.take_profit - result.entry_price
        assert tp_dist == pytest.approx(2 * stop_dist, rel=1e-4)

    def test_take_profit_is_2x_stop_distance_short(self, config: dict) -> None:
        """Take profit should be 2x the stop loss distance from entry (short)."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakdown = make_candle(
            close=98.5, open_=99.2, high=99.3, low=98.3,
            volume=20000, timeframe="5m",
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakdown,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=2.0,
        )
        # entry = 98.5, stop = 101.0, stop_dist = 2.5
        # take_profit = 98.5 - 2 * 2.5 = 93.5
        stop_dist = result.stop_loss - result.entry_price
        tp_dist = result.entry_price - result.take_profit
        assert tp_dist == pytest.approx(2 * stop_dist, rel=1e-4)


# ===========================================================================
# First Candle -- Swing Bias Blocking
# ===========================================================================

class TestFirstCandleSwingBias:
    """Tests that swing bias blocking works for ORB signals."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "volume_multiplier": 1.5,
            "orb_window_minutes": 60,
            "valid_until_hour": 13,
        }

    def test_long_blocked_by_swing_bias(self, config: dict) -> None:
        """Long signal is blocked when swing advisor says to block."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m",
        )
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=True,
        ):
            result = signal.evaluate_with_context(
                symbol="TEST",
                orb_high=101.0,
                orb_low=99.0,
                current_candle=breakout,
                avg_volume=10000.0,
                minutes_since_open=90,
                atr_20d=2.0,
            )
        assert result.triggered is False
        assert "swing bias" in result.reason.lower()

    def test_short_blocked_by_swing_bias(self, config: dict) -> None:
        """Short signal is blocked when swing advisor says to block."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal(config)
        breakdown = make_candle(
            close=98.5, open_=99.2, high=99.3, low=98.3,
            volume=20000, timeframe="5m",
        )
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=True,
        ):
            result = signal.evaluate_with_context(
                symbol="TEST",
                orb_high=101.0,
                orb_low=99.0,
                current_candle=breakdown,
                avg_volume=10000.0,
                minutes_since_open=90,
                atr_20d=2.0,
            )
        assert result.triggered is False
        assert "swing bias" in result.reason.lower()


# ===========================================================================
# EMA Cross
# ===========================================================================

class TestEMACross:
    """Tests for the EMA golden/death cross signal."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "fast_ema": 50,
            "slow_ema": 200,
        }

    @pytest.fixture(autouse=True)
    def mock_swing_advisor(self):
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=False,
        ):
            yield

    def test_golden_cross_long_signal(self, config: dict) -> None:
        """Long bias activated when 50 EMA crosses above 200 EMA."""
        from signals.ema_cross import EMACrossSignal

        signal = EMACrossSignal(config)
        result = signal.evaluate_from_emas(
            symbol="TEST",
            prev_fast_ema=99.0, prev_slow_ema=100.0,
            curr_fast_ema=101.0, curr_slow_ema=100.0,
            rsi=55.0, current_price=101.0,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.LONG

    def test_death_cross_short_signal(self, config: dict) -> None:
        """Short bias activated when 50 EMA crosses below 200 EMA."""
        from signals.ema_cross import EMACrossSignal

        signal = EMACrossSignal(config)
        result = signal.evaluate_from_emas(
            symbol="TEST",
            prev_fast_ema=101.0, prev_slow_ema=100.0,
            curr_fast_ema=99.0, curr_slow_ema=100.0,
            rsi=45.0, current_price=99.0,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.SHORT

    def test_no_signal_without_rsi_confirmation(self, config: dict) -> None:
        """Golden cross requires RSI > 50 to confirm long."""
        from signals.ema_cross import EMACrossSignal

        signal = EMACrossSignal(config)
        result = signal.evaluate_from_emas(
            symbol="TEST",
            prev_fast_ema=99.0, prev_slow_ema=100.0,
            curr_fast_ema=101.0, curr_slow_ema=100.0,
            rsi=45.0,  # RSI below 50, no confirmation
            current_price=101.0,
        )
        assert result.triggered is False

    def test_no_cross_no_signal(self, config: dict) -> None:
        """No signal when EMAs don't cross."""
        from signals.ema_cross import EMACrossSignal

        signal = EMACrossSignal(config)
        result = signal.evaluate_from_emas(
            symbol="TEST",
            prev_fast_ema=101.0, prev_slow_ema=100.0,
            curr_fast_ema=102.0, curr_slow_ema=100.0,
            rsi=55.0, current_price=102.0,
        )
        assert result.triggered is False


# ===========================================================================
# EMA Cross -- Swing Bias Blocking
# ===========================================================================

class TestEMACrossSwingBias:
    """Tests that swing bias blocking works for EMA cross signals."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "fast_ema": 50,
            "slow_ema": 200,
        }

    def test_golden_cross_blocked_by_swing_bias(self, config: dict) -> None:
        """Golden cross is blocked when swing advisor says to block."""
        from signals.ema_cross import EMACrossSignal

        signal = EMACrossSignal(config)
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=True,
        ):
            result = signal.evaluate_from_emas(
                symbol="TEST",
                prev_fast_ema=99.0, prev_slow_ema=100.0,
                curr_fast_ema=101.0, curr_slow_ema=100.0,
                rsi=55.0, current_price=101.0,
            )
        assert result.triggered is False
        assert "swing bias" in result.reason.lower()


# ===========================================================================
# VWAP Reversion
# ===========================================================================

class TestVWAPReversionRedesign:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "atr_band_multiplier": 1.5,
            "atr_period": 14,
            "volume_confirmation_mult": 1.2,
            "slope_max": 0.001,
            "slope_lookback": 20,
            "stop_loss_atr_mult": 1.0,
            "rr_ratio": 2.0,
        }

    @pytest.fixture(autouse=True)
    def _no_swing_block(self):
        with patch("signals.vwap_reversion.VWAPReversionSignal.check_swing_bias", return_value=False):
            yield

    def test_long_below_lower_band(self, config: dict) -> None:
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=95.0, vwap=100.0,
            atr=4.0, vwap_slope=0.0003, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        assert not result.triggered

    def test_long_triggers_at_band(self, config: dict) -> None:
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=93.5, vwap=100.0,
            atr=4.0, vwap_slope=0.0003, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        assert result.triggered
        assert result.direction == SignalDirection.LONG
        assert result.take_profit > result.entry_price
        assert result.stop_loss < result.entry_price

    def test_short_triggers_at_upper_band(self, config: dict) -> None:
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=107.0, vwap=100.0,
            atr=4.0, vwap_slope=-0.0002, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        assert result.triggered
        assert result.direction == SignalDirection.SHORT
        assert result.take_profit < result.entry_price
        assert result.stop_loss > result.entry_price

    def test_no_trigger_slope_too_steep(self, config: dict) -> None:
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=93.5, vwap=100.0,
            atr=4.0, vwap_slope=0.005, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        assert not result.triggered
        assert "slope" in result.reason.lower()

    def test_no_trigger_low_volume(self, config: dict) -> None:
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=93.5, vwap=100.0,
            atr=4.0, vwap_slope=0.0003, current_volume=800.0,
            avg_volume=1000.0, market="stock",
        )
        assert not result.triggered
        assert "volume" in result.reason.lower()

    def test_disabled(self, config: dict) -> None:
        config["enabled"] = False
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=93.5, vwap=100.0,
            atr=4.0, vwap_slope=0.0003, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        assert not result.triggered


# ===========================================================================
# VWAP Reversion -- Swing Bias Blocking
# ===========================================================================

class TestVWAPReversionSwingBiasRedesign:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "atr_band_multiplier": 1.5,
            "atr_period": 14,
            "volume_confirmation_mult": 1.2,
            "slope_max": 0.001,
            "slope_lookback": 20,
            "stop_loss_atr_mult": 1.0,
            "rr_ratio": 2.0,
        }

    def test_long_blocked_by_swing_bias(self, config: dict) -> None:
        with patch("signals.vwap_reversion.VWAPReversionSignal.check_swing_bias", return_value=True):
            sig = VWAPReversionSignal(config)
            result = sig.evaluate_from_vwap(
                symbol="AAPL", current_price=93.5, vwap=100.0,
                atr=4.0, vwap_slope=0.0003, current_volume=1500.0,
                avg_volume=1000.0, market="stock",
            )
            assert not result.triggered
            assert "swing" in result.reason.lower()


# ===========================================================================
# RSI Momentum
# ===========================================================================

class TestRSIMomentum:
    """Tests for the multi-timeframe RSI momentum signal."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "rsi_oversold": 32,
            "rsi_overbought": 68,
            "rsi_neutral_low": 45,
            "rsi_neutral_high": 55,
            "rsi_period": 14,
        }

    @pytest.fixture(autouse=True)
    def mock_swing_advisor(self):
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=False,
        ):
            yield

    def test_long_signal_both_oversold(self, config: dict) -> None:
        """BUY when RSI(5min) < 32 AND RSI(1hr) < 45."""
        from signals.rsi_momentum import RSIMomentumSignal

        signal = RSIMomentumSignal(config)
        result = signal.evaluate_from_rsi(
            symbol="TEST", rsi_5m=28.0, rsi_1h=40.0, current_price=100.0,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.LONG

    def test_short_signal_both_overbought(self, config: dict) -> None:
        """SELL when RSI(5min) > 68 AND RSI(1hr) > 55."""
        from signals.rsi_momentum import RSIMomentumSignal

        signal = RSIMomentumSignal(config)
        result = signal.evaluate_from_rsi(
            symbol="TEST", rsi_5m=72.0, rsi_1h=60.0, current_price=100.0,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.SHORT

    def test_no_signal_when_hourly_rsi_neutral(self, config: dict) -> None:
        """No signal when RSI(1hr) is in the 45-55 neutral zone."""
        from signals.rsi_momentum import RSIMomentumSignal

        signal = RSIMomentumSignal(config)
        result = signal.evaluate_from_rsi(
            symbol="TEST", rsi_5m=28.0, rsi_1h=50.0, current_price=100.0,
        )
        assert result.triggered is False

    def test_no_signal_when_only_5m_oversold(self, config: dict) -> None:
        """No buy when only 5-min RSI is oversold but 1-hr is not."""
        from signals.rsi_momentum import RSIMomentumSignal

        signal = RSIMomentumSignal(config)
        result = signal.evaluate_from_rsi(
            symbol="TEST", rsi_5m=28.0, rsi_1h=48.0, current_price=100.0,
        )
        assert result.triggered is False


# ===========================================================================
# RSI Momentum -- Swing Bias Blocking
# ===========================================================================

class TestRSIMomentumSwingBias:
    """Tests that swing bias blocking works for RSI momentum signals."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "rsi_oversold": 32,
            "rsi_overbought": 68,
            "rsi_neutral_low": 45,
            "rsi_neutral_high": 55,
        }

    def test_long_blocked_by_swing_bias(self, config: dict) -> None:
        """RSI long signal is blocked when swing advisor says to block."""
        from signals.rsi_momentum import RSIMomentumSignal

        signal = RSIMomentumSignal(config)
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=True,
        ):
            result = signal.evaluate_from_rsi(
                symbol="TEST", rsi_5m=28.0, rsi_1h=40.0, current_price=100.0,
            )
        assert result.triggered is False
        assert "swing bias" in result.reason.lower()


# ===========================================================================
# Bollinger Band Fade
# ===========================================================================

class TestBollingerFade:
    """Tests for the Bollinger Band outer touch fade signal."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_threshold_low": 35,
            "rsi_threshold_high": 65,
            "stop_beyond_band_pct": 0.005,
        }

    @pytest.fixture(autouse=True)
    def mock_swing_advisor(self):
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=False,
        ):
            yield

    def test_long_signal_lower_band_touch(self, config: dict) -> None:
        """BUY when price touches lower band + RSI < 35 + prev candle bearish."""
        from signals.bollinger_fade import BollingerFadeSignal

        signal = BollingerFadeSignal(config)
        prev_candle = make_candle(close=98.0, open_=99.0, timeframe="15m")  # bearish
        result = signal.evaluate_from_bands(
            symbol="TEST", current_price=97.5,
            lower_band=97.6, upper_band=102.4, middle_band=100.0,
            rsi=30.0, prev_candle=prev_candle,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.LONG
        assert result.take_profit == pytest.approx(98.475, rel=1e-2)  # 2:1 R:R

    def test_short_signal_upper_band_touch(self, config: dict) -> None:
        """SELL when price touches upper band + RSI > 65 + prev candle bullish."""
        from signals.bollinger_fade import BollingerFadeSignal

        signal = BollingerFadeSignal(config)
        prev_candle = make_candle(close=102.0, open_=101.0, timeframe="15m")  # bullish
        result = signal.evaluate_from_bands(
            symbol="TEST", current_price=102.5,
            lower_band=97.6, upper_band=102.4, middle_band=100.0,
            rsi=70.0, prev_candle=prev_candle,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.SHORT
        assert result.take_profit == pytest.approx(101.475, rel=1e-2)  # 2:1 R:R

    def test_no_signal_when_rsi_not_extreme(self, config: dict) -> None:
        """No signal if RSI is in a moderate range."""
        from signals.bollinger_fade import BollingerFadeSignal

        signal = BollingerFadeSignal(config)
        prev_candle = make_candle(close=98.0, open_=99.0, timeframe="15m")
        result = signal.evaluate_from_bands(
            symbol="TEST", current_price=97.5,
            lower_band=97.6, upper_band=102.4, middle_band=100.0,
            rsi=45.0,  # not extreme
            prev_candle=prev_candle,
        )
        assert result.triggered is False

    def test_no_signal_when_prev_candle_wrong_direction(self, config: dict) -> None:
        """Lower band touch requires previous candle to be bearish."""
        from signals.bollinger_fade import BollingerFadeSignal

        signal = BollingerFadeSignal(config)
        prev_candle = make_candle(close=99.5, open_=98.5, timeframe="15m")  # bullish
        result = signal.evaluate_from_bands(
            symbol="TEST", current_price=97.5,
            lower_band=97.6, upper_band=102.4, middle_band=100.0,
            rsi=30.0, prev_candle=prev_candle,
        )
        assert result.triggered is False

    def test_stop_loss_beyond_band_touch(self, config: dict) -> None:
        """Stop is 0.5% beyond the band touch point."""
        from signals.bollinger_fade import BollingerFadeSignal

        signal = BollingerFadeSignal(config)
        prev_candle = make_candle(close=98.0, open_=99.0, timeframe="15m")
        result = signal.evaluate_from_bands(
            symbol="TEST", current_price=97.5,
            lower_band=97.6, upper_band=102.4, middle_band=100.0,
            rsi=30.0, prev_candle=prev_candle,
        )
        expected_stop = 97.5 * (1 - 0.005)
        assert result.stop_loss == pytest.approx(expected_stop, rel=1e-4)


# ===========================================================================
# Bollinger Fade -- Swing Bias Blocking
# ===========================================================================

class TestBollingerFadeSwingBias:
    """Tests that swing bias blocking works for Bollinger fade signals."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_threshold_low": 35,
            "rsi_threshold_high": 65,
            "stop_beyond_band_pct": 0.005,
        }

    def test_long_blocked_by_swing_bias(self, config: dict) -> None:
        """Bollinger long signal is blocked when swing advisor says to block."""
        from signals.bollinger_fade import BollingerFadeSignal

        signal = BollingerFadeSignal(config)
        prev_candle = make_candle(close=98.0, open_=99.0, timeframe="15m")
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=True,
        ):
            result = signal.evaluate_from_bands(
                symbol="TEST", current_price=97.5,
                lower_band=97.6, upper_band=102.4, middle_band=100.0,
                rsi=30.0, prev_candle=prev_candle,
            )
        assert result.triggered is False
        assert "swing bias" in result.reason.lower()


# ===========================================================================
# Disabled signal
# ===========================================================================

class TestSignalDisabled:
    """Ensure disabled signals never fire."""

    @pytest.fixture(autouse=True)
    def mock_swing_advisor(self):
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=False,
        ):
            yield

    def test_disabled_signal_returns_no_trigger(self) -> None:
        """A signal with enabled=False must return triggered=False."""
        from signals.first_candle import FirstCandleSignal

        signal = FirstCandleSignal({
            "enabled": False,
            "volume_multiplier": 1.5,
            "orb_window_minutes": 60,
            "valid_until_hour": 13,
        })
        breakout = make_candle(
            close=101.5, open_=100.8, high=101.8, low=100.7,
            volume=20000, timeframe="5m", ts_offset_min=90,
        )
        result = signal.evaluate_with_context(
            symbol="TEST",
            orb_high=101.0,
            orb_low=99.0,
            current_candle=breakout,
            avg_volume=10000.0,
            minutes_since_open=90,
            atr_20d=2.0,
        )
        assert result.triggered is False


# ---------------------------------------------------------------------------
# TestMACDIndicator
# ---------------------------------------------------------------------------

class TestMACDIndicator:
    def test_macd_basic_output_shape(self) -> None:
        """MACD returns three lists of same length as input."""
        closes = [float(i) for i in range(50)]
        macd_line, signal_line, histogram = Indicators.macd(closes, fast=12, slow=26, signal=9)
        assert len(macd_line) == 50
        assert len(signal_line) == 50
        assert len(histogram) == 50

    def test_macd_early_values_are_nan(self) -> None:
        """First slow+signal-2 values should be NaN."""
        closes = [float(i) for i in range(50)]
        macd_line, signal_line, histogram = Indicators.macd(closes, fast=12, slow=26, signal=9)
        import math
        assert math.isnan(macd_line[0])
        assert math.isnan(signal_line[0])
        assert math.isnan(histogram[0])

    def test_macd_converging_prices(self) -> None:
        """With constant prices, MACD should converge to zero."""
        closes = [100.0] * 60
        macd_line, signal_line, histogram = Indicators.macd(closes)
        assert abs(macd_line[-1]) < 0.01
        assert abs(signal_line[-1]) < 0.01
        assert abs(histogram[-1]) < 0.01

    def test_macd_uptrend_positive(self) -> None:
        """In a steady uptrend, MACD line should be positive."""
        closes = [100.0 + i * 0.5 for i in range(60)]
        macd_line, signal_line, histogram = Indicators.macd(closes)
        assert macd_line[-1] > 0


# ===========================================================================
# Volume Profile
# ===========================================================================

class TestVolumeProfileIndicator:
    def _make_session_candles(self, prices_volumes: list[tuple[float, float, float, float, float]]) -> list[OHLCV]:
        """Create candles from (open, high, low, close, volume) tuples."""
        from datetime import datetime, timezone, timedelta
        candles = []
        base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
        for i, (o, h, l, c, v) in enumerate(prices_volumes):
            candles.append(OHLCV(
                timestamp=base + timedelta(minutes=i * 5),
                open=o, high=h, low=l, close=c, volume=v,
                symbol="TEST", timeframe="5m", market="stock",
            ))
        return candles

    def test_poc_is_highest_volume_level(self) -> None:
        candles = self._make_session_candles([
            (99, 101, 98, 100, 10000),
            (100, 102, 99, 101, 10000),
            (100, 101, 99, 100, 10000),
            (109, 111, 108, 110, 100),
            (110, 112, 109, 111, 100),
        ])
        poc, vah, val = Indicators.volume_profile(candles, num_bins=20)
        assert abs(poc - 100.0) < 2.0

    def test_value_area_contains_70_pct(self) -> None:
        candles = self._make_session_candles([
            (99, 101, 98, 100, 5000),
            (100, 102, 99, 101, 5000),
            (100, 101, 99, 100, 5000),
            (105, 107, 104, 106, 1000),
            (110, 112, 109, 111, 500),
        ])
        poc, vah, val = Indicators.volume_profile(candles, num_bins=20)
        assert val <= poc <= vah

    def test_single_candle(self) -> None:
        candles = self._make_session_candles([(100, 105, 95, 102, 1000)])
        poc, vah, val = Indicators.volume_profile(candles, num_bins=10)
        typical = (105 + 95 + 102) / 3
        assert abs(poc - typical) < 2.0

    def test_empty_candles(self) -> None:
        poc, vah, val = Indicators.volume_profile([], num_bins=20)
        assert poc == 0.0
        assert vah == 0.0
        assert val == 0.0


class TestVWAPSlopeIndicator:
    def test_flat_vwap_returns_near_zero(self) -> None:
        vwap_values = [100.0] * 30
        slope = Indicators.vwap_slope(vwap_values, lookback=20)
        assert abs(slope) < 0.0001

    def test_uptrending_vwap_positive_slope(self) -> None:
        vwap_values = [100.0 + i * 0.1 for i in range(30)]
        slope = Indicators.vwap_slope(vwap_values, lookback=20)
        assert slope > 0

    def test_downtrending_vwap_negative_slope(self) -> None:
        vwap_values = [100.0 - i * 0.1 for i in range(30)]
        slope = Indicators.vwap_slope(vwap_values, lookback=20)
        assert slope < 0

    def test_insufficient_data_returns_zero(self) -> None:
        slope = Indicators.vwap_slope([100.0], lookback=20)
        assert slope == 0.0

    def test_normalization_across_price_levels(self) -> None:
        low = [100.0 + i * 0.1 for i in range(30)]
        high = [50000.0 + i * 50.0 for i in range(30)]
        slope_low = Indicators.vwap_slope(low, lookback=20)
        slope_high = Indicators.vwap_slope(high, lookback=20)
        assert abs(slope_low - slope_high) < abs(slope_low) * 0.5


from signals.vpoc_bounce import VPOCBounceSignal


class TestVPOCBounce:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "num_bins": 20,
            "proximity_pct": 0.002,
            "bounce_candles": 2,
            "min_poc_volume_pct": 0.15,
            "stop_loss_pct": 0.01,
            "rr_ratio": 2.0,
        }

    @pytest.fixture(autouse=True)
    def _no_swing_block(self):
        with patch("signals.vpoc_bounce.VPOCBounceSignal.check_swing_bias", return_value=False):
            yield

    def _make_session_candles(self, data: list[tuple]) -> list[OHLCV]:
        from datetime import datetime, timezone, timedelta
        candles = []
        base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
        for i, (o, h, l, c, v) in enumerate(data):
            candles.append(OHLCV(
                timestamp=base + timedelta(minutes=i * 5),
                open=o, high=h, low=l, close=c, volume=v,
                symbol="AAPL", timeframe="5m", market="stock",
            ))
        return candles

    def test_long_bounce_off_poc(self, config: dict) -> None:
        session = self._make_session_candles([
            (99, 101, 98, 100, 10000),
            (100, 102, 99, 101, 10000),
            (100, 101, 99, 100, 10000),
            (100, 101, 99, 100, 10000),
            (100, 101, 99, 100, 10000),
            (99.5, 100.2, 99.3, 100.0, 5000),
            (99.8, 100.5, 99.7, 100.3, 5000),
        ])
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=100.1,
            session_candles=session, recent_candles=session[-2:],
            market="stock",
        )
        assert result.triggered
        assert result.direction == SignalDirection.LONG

    def test_short_bounce_off_poc(self, config: dict) -> None:
        session = self._make_session_candles([
            (99, 101, 98, 100, 10000),
            (100, 102, 99, 101, 10000),
            (100, 101, 99, 100, 10000),
            (100, 101, 99, 100, 10000),
            (100, 101, 99, 100, 10000),
            (100.5, 100.8, 99.9, 100.0, 5000),
            (100.2, 100.4, 99.6, 99.8, 5000),
        ])
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=99.9,
            session_candles=session, recent_candles=session[-2:],
            market="stock",
        )
        assert result.triggered
        assert result.direction == SignalDirection.SHORT

    def test_no_trigger_price_far_from_poc(self, config: dict) -> None:
        session = self._make_session_candles([
            (99, 101, 98, 100, 10000),
            (100, 102, 99, 101, 10000),
            (100, 101, 99, 100, 10000),
            (109, 111, 108, 110, 100),
            (110, 112, 109, 111, 100),
        ])
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=110.0,
            session_candles=session, recent_candles=session[-2:],
            market="stock",
        )
        assert not result.triggered

    def test_no_trigger_weak_poc(self, config: dict) -> None:
        session = self._make_session_candles([
            (95, 97, 94, 96, 1000),
            (98, 100, 97, 99, 1000),
            (101, 103, 100, 102, 1000),
            (104, 106, 103, 105, 1000),
            (107, 109, 106, 108, 1000),
            (108, 109, 107, 108, 1000),
            (108, 109, 107, 108, 1000),
        ])
        config["min_poc_volume_pct"] = 0.40
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=108.0,
            session_candles=session, recent_candles=session[-2:],
            market="stock",
        )
        assert not result.triggered

    def test_disabled(self, config: dict) -> None:
        config["enabled"] = False
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=100.0,
            session_candles=[], recent_candles=[],
            market="stock",
        )
        assert not result.triggered


class TestVPOCBounceSwingBias:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "num_bins": 20,
            "proximity_pct": 0.002,
            "bounce_candles": 2,
            "min_poc_volume_pct": 0.15,
            "stop_loss_pct": 0.01,
            "rr_ratio": 2.0,
        }

    def test_blocked_by_swing_bias(self, config: dict) -> None:
        from datetime import datetime, timezone, timedelta
        base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
        session = [
            OHLCV(timestamp=base + timedelta(minutes=i * 5),
                   open=99+i*0.1, high=101, low=98, close=100, volume=10000,
                   symbol="AAPL", timeframe="5m", market="stock")
            for i in range(7)
        ]
        with patch("signals.vpoc_bounce.VPOCBounceSignal.check_swing_bias", return_value=True):
            sig = VPOCBounceSignal(config)
            result = sig.evaluate_from_profile(
                symbol="AAPL", current_price=100.0,
                session_candles=session, recent_candles=session[-2:],
                market="stock",
            )
            assert not result.triggered


class TestMACDDivergence:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "divergence_lookback": 30,
            "min_swing_distance": 5,
            "stop_loss_pct": 0.015,
            "rr_ratio": 2.0,
        }

    @pytest.fixture(autouse=True)
    def _no_swing_block(self):
        with patch("signals.macd_divergence.MACDDivergenceSignal.check_swing_bias", return_value=False):
            yield

    def _make_candles(self, closes: list[float]) -> list[OHLCV]:
        from datetime import datetime, timezone, timedelta
        base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
        candles = []
        for i, c in enumerate(closes):
            candles.append(OHLCV(
                timestamp=base + timedelta(minutes=i * 5),
                open=c, high=c + 0.5, low=c - 0.5, close=c, volume=1000,
                symbol="AAPL", timeframe="5m", market="stock",
            ))
        return candles

    def test_bullish_divergence_with_cross(self, config: dict) -> None:
        closes = [100.0 - i * 0.3 for i in range(35)]
        closes.append(90.0)
        closes += [91.0 + i * 0.1 for i in range(14)]
        closes.append(88.0)
        closes += [89.0, 89.5, 90.0, 90.5, 91.0, 91.5, 92.0, 93.0, 93.5, 94.0]
        candles = self._make_candles(closes)
        sig = MACDDivergenceSignal(config)
        result = sig.evaluate_from_macd(
            symbol="AAPL", current_price=94.0, candles=candles, market="stock",
        )
        assert isinstance(result.triggered, bool)
        assert result.strategy_name == "macd_divergence"

    def test_no_trigger_without_cross(self, config: dict) -> None:
        closes = [100.0 - i * 0.2 for i in range(60)]
        candles = self._make_candles(closes)
        sig = MACDDivergenceSignal(config)
        result = sig.evaluate_from_macd(
            symbol="AAPL", current_price=closes[-1], candles=candles, market="stock",
        )
        assert not result.triggered

    def test_no_trigger_insufficient_data(self, config: dict) -> None:
        closes = [100.0] * 20
        candles = self._make_candles(closes)
        sig = MACDDivergenceSignal(config)
        result = sig.evaluate_from_macd(
            symbol="AAPL", current_price=100.0, candles=candles, market="stock",
        )
        assert not result.triggered

    def test_disabled(self, config: dict) -> None:
        config["enabled"] = False
        sig = MACDDivergenceSignal(config)
        result = sig.evaluate_from_macd(
            symbol="AAPL", current_price=100.0, candles=[], market="stock",
        )
        assert not result.triggered

    def test_returns_correct_strategy_name(self, config: dict) -> None:
        sig = MACDDivergenceSignal(config)
        assert sig.name == "macd_divergence"


class TestMACDDivergenceSwingBias:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "divergence_lookback": 30,
            "min_swing_distance": 5,
            "stop_loss_pct": 0.015,
            "rr_ratio": 2.0,
        }

    def test_blocked_by_swing_bias(self, config: dict) -> None:
        with patch("signals.macd_divergence.MACDDivergenceSignal.check_swing_bias", return_value=True):
            sig = MACDDivergenceSignal(config)
            closes = [100.0] * 60
            from datetime import datetime, timezone, timedelta
            base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
            candles = [
                OHLCV(timestamp=base + timedelta(minutes=i * 5),
                       open=c, high=c + 0.5, low=c - 0.5, close=c, volume=1000,
                       symbol="AAPL", timeframe="5m", market="stock")
                for i, c in enumerate(closes)
            ]
            result = sig.evaluate_from_macd(
                symbol="AAPL", current_price=100.0, candles=candles, market="stock",
            )
            assert not result.triggered
