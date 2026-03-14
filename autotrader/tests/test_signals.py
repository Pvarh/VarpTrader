"""Tests for all trading signal strategies."""

import sys
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

sys.path.insert(0, ".")

from journal.models import OHLCV
from signals.base_signal import SignalResult, SignalDirection


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

class TestVWAPReversion:
    """Tests for the VWAP mean reversion signal."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "vwap_deviation_pct": 0.3,  # 0.3% -> 0.003 after /100
            "momentum_candles": 3,
        }

    @pytest.fixture(autouse=True)
    def mock_swing_advisor(self):
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=False,
        ):
            yield

    def test_long_signal_below_vwap_with_bullish_momentum(self, config: dict) -> None:
        """BUY when price is 0.3% below VWAP and last 3 candles show bullish momentum."""
        from signals.vwap_reversion import VWAPReversionSignal

        signal = VWAPReversionSignal(config)
        vwap = 100.0
        price = 99.65  # 0.35% below VWAP (below the 0.3% band = 99.7)
        # Last 3 candles bullish: each close > open
        candles = [
            make_candle(close=99.5, open_=99.3, ts_offset_min=i * 5)
            for i in range(3)
        ]
        result = signal.evaluate_from_vwap(
            symbol="TEST", current_price=price, vwap=vwap,
            recent_candles=candles, minutes_since_open=60,
            minutes_before_close=120,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.LONG

    def test_short_signal_above_vwap_with_bearish_momentum(self, config: dict) -> None:
        """SELL when price is 0.3% above VWAP and last 3 candles bearish."""
        from signals.vwap_reversion import VWAPReversionSignal

        signal = VWAPReversionSignal(config)
        vwap = 100.0
        price = 100.35  # 0.35% above VWAP (above the 0.3% band = 100.3)
        candles = [
            make_candle(close=100.3, open_=100.5, ts_offset_min=i * 5)
            for i in range(3)
        ]
        result = signal.evaluate_from_vwap(
            symbol="TEST", current_price=price, vwap=vwap,
            recent_candles=candles, minutes_since_open=60,
            minutes_before_close=120,
        )
        assert result.triggered is True
        assert result.direction == SignalDirection.SHORT

    def test_no_signal_when_within_vwap_band(self, config: dict) -> None:
        """No signal when price is within 0.3% of VWAP."""
        from signals.vwap_reversion import VWAPReversionSignal

        signal = VWAPReversionSignal(config)
        result = signal.evaluate_from_vwap(
            symbol="TEST", current_price=100.1, vwap=100.0,
            recent_candles=make_candle_series([100.0, 100.05, 100.1]),
            minutes_since_open=60, minutes_before_close=120,
        )
        assert result.triggered is False

    def test_no_signal_outside_valid_trading_hours(self, config: dict) -> None:
        """No signal in mid-session (only first 3 hours and last 1 hour)."""
        from signals.vwap_reversion import VWAPReversionSignal

        signal = VWAPReversionSignal(config)
        result = signal.evaluate_from_vwap(
            symbol="TEST", current_price=99.5, vwap=100.0,
            recent_candles=make_candle_series([99.4, 99.45, 99.5]),
            minutes_since_open=200,  # > 180 min
            minutes_before_close=120,  # > 60 min
        )
        assert result.triggered is False


# ===========================================================================
# VWAP Reversion -- Swing Bias Blocking
# ===========================================================================

class TestVWAPReversionSwingBias:
    """Tests that swing bias blocking works for VWAP reversion signals."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "vwap_deviation_pct": 0.3,
            "momentum_candles": 3,
        }

    def test_long_blocked_by_swing_bias(self, config: dict) -> None:
        """VWAP long signal is blocked when swing advisor says to block."""
        from signals.vwap_reversion import VWAPReversionSignal

        signal = VWAPReversionSignal(config)
        candles = [
            make_candle(close=99.5, open_=99.3, ts_offset_min=i * 5)
            for i in range(3)
        ]
        with patch(
            "analysis.swing_advisor.SwingAdvisor.should_block_trade",
            return_value=True,
        ):
            result = signal.evaluate_from_vwap(
                symbol="TEST", current_price=99.65, vwap=100.0,
                recent_candles=candles, minutes_since_open=60,
                minutes_before_close=120,
            )
        assert result.triggered is False
        assert "swing bias" in result.reason.lower()


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
        assert result.take_profit == pytest.approx(100.0, rel=1e-2)  # midline

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
        assert result.take_profit == pytest.approx(100.0, rel=1e-2)

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
