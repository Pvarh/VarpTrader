"""Tests for EMAPullbackSignal (EMA pullback / trend continuation strategy)."""

import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, ".")

from signals.base_signal import SignalDirection
from signals.ema_pullback import EMAPullbackSignal


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

PULLBACK_CONF = {
    "enabled": True,
    "fast_ema": 20,
    "slow_ema": 50,
    "pullback_pct": 0.002,
    "rsi_max_long": 55,
    "rsi_min_short": 45,
    "stop_loss_pct": 0.015,
    "volume_confirmation": True,
}


@pytest.fixture(autouse=True)
def allow_swing_bias():
    """Bypass swing advisor so tests focus on pullback logic."""
    with patch(
        "signals.ema_pullback.EMAPullbackSignal.check_swing_bias",
        return_value=True,
    ):
        yield


# ===========================================================================
# LONG signals
# ===========================================================================

class TestLongSignal:

    def test_fires_at_ema20_in_uptrend(self):
        """Price exactly at EMA20, uptrend confirmed, RSI < 55 → LONG."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="BTC/USDT",
            current_price=100.00,   # exactly at EMA20
            ema_fast=100.00,        # EMA20
            ema_slow=95.00,         # EMA50 below → uptrend
            rsi=48.0,
        )
        assert result.triggered
        assert result.direction == SignalDirection.LONG
        assert result.symbol == "BTC/USDT"

    def test_fires_slightly_above_ema20(self):
        """Price 0.15% above EMA20 (within 0.2% window) → LONG."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        ema20 = 1000.0
        price = ema20 * 1.0015   # 0.15% above → within pullback_pct=0.002
        result = sig.evaluate_pullback(
            symbol="AAPL",
            current_price=price,
            ema_fast=ema20,
            ema_slow=950.0,
            rsi=50.0,
        )
        assert result.triggered
        assert result.direction == SignalDirection.LONG

    def test_no_signal_price_too_far_above_ema20(self):
        """Price 0.5% above EMA20 (outside 0.2% window) → no signal."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        ema20 = 1000.0
        price = ema20 * 1.005   # 0.5% above → outside pullback_pct
        result = sig.evaluate_pullback(
            symbol="AAPL",
            current_price=price,
            ema_fast=ema20,
            ema_slow=950.0,
            rsi=50.0,
        )
        assert not result.triggered

    def test_no_signal_rsi_too_high_for_long(self):
        """RSI = 57 (>= rsi_max_long=55) in uptrend → no signal."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="NVDA",
            current_price=500.0,
            ema_fast=500.0,
            ema_slow=480.0,
            rsi=57.0,
        )
        assert not result.triggered

    def test_rsi_at_max_long_boundary(self):
        """RSI = 55 exactly (not less than rsi_max_long) → no signal."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="ETH/USDT",
            current_price=3000.0,
            ema_fast=3000.0,
            ema_slow=2900.0,
            rsi=55.0,  # not < 55
        )
        assert not result.triggered

    def test_no_signal_price_below_ema50(self):
        """Price below EMA50 in uptrend (above_floor=False) → no signal."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        # EMA20=100 > EMA50=110 would be downtrend, use correct uptrend
        # but set price below EMA50
        result = sig.evaluate_pullback(
            symbol="SPY",
            current_price=94.0,    # below EMA50=95
            ema_fast=100.0,        # EMA20 > EMA50 → uptrend
            ema_slow=95.0,
            rsi=45.0,
        )
        assert not result.triggered

    def test_stop_loss_below_entry_for_long(self):
        """Long SL should be below entry price."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="BTC/USDT",
            current_price=50000.0,
            ema_fast=50000.0,
            ema_slow=48000.0,
            rsi=48.0,
        )
        assert result.triggered
        assert result.stop_loss < result.entry_price
        assert result.take_profit > result.entry_price

    def test_take_profit_is_2x_stop_distance(self):
        """TP should be 2× the SL distance from entry."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="BTC/USDT",
            current_price=1000.0,
            ema_fast=1000.0,
            ema_slow=950.0,
            rsi=48.0,
        )
        assert result.triggered
        stop_dist = result.entry_price - result.stop_loss
        tp_dist = result.take_profit - result.entry_price
        assert abs(tp_dist - 2 * stop_dist) < 0.01


# ===========================================================================
# SHORT signals
# ===========================================================================

class TestShortSignal:

    def test_fires_at_ema20_in_downtrend(self):
        """Price exactly at EMA20, downtrend confirmed, RSI > 45 → SHORT."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="BTC/USDT",
            current_price=100.00,
            ema_fast=100.00,        # EMA20
            ema_slow=105.00,        # EMA50 above → downtrend
            rsi=52.0,
        )
        assert result.triggered
        assert result.direction == SignalDirection.SHORT

    def test_fires_slightly_below_ema20(self):
        """Price 0.15% below EMA20 (within 0.2% window) in downtrend → SHORT."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        ema20 = 1000.0
        price = ema20 * 0.9985   # 0.15% below
        result = sig.evaluate_pullback(
            symbol="TSLA",
            current_price=price,
            ema_fast=ema20,
            ema_slow=1050.0,       # EMA50 above → downtrend
            rsi=52.0,
        )
        assert result.triggered
        assert result.direction == SignalDirection.SHORT

    def test_no_signal_price_too_far_below_ema20(self):
        """Price 0.5% below EMA20 (outside window) in downtrend → no signal."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        ema20 = 1000.0
        price = ema20 * 0.995   # 0.5% below
        result = sig.evaluate_pullback(
            symbol="TSLA",
            current_price=price,
            ema_fast=ema20,
            ema_slow=1050.0,
            rsi=52.0,
        )
        assert not result.triggered

    def test_no_signal_rsi_too_low_for_short(self):
        """RSI = 43 (<= rsi_min_short=45) in downtrend → no signal."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="NVDA",
            current_price=500.0,
            ema_fast=500.0,
            ema_slow=520.0,
            rsi=43.0,
        )
        assert not result.triggered

    def test_rsi_at_min_short_boundary(self):
        """RSI = 45 exactly (not > rsi_min_short) → no signal."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="QQQ",
            current_price=400.0,
            ema_fast=400.0,
            ema_slow=420.0,
            rsi=45.0,  # not > 45
        )
        assert not result.triggered

    def test_no_signal_price_above_ema50(self):
        """Price above EMA50 in downtrend (below_ceiling=False) → no signal."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="SPY",
            current_price=115.0,   # above EMA50=110
            ema_fast=100.0,        # EMA20 < EMA50 → downtrend
            ema_slow=110.0,
            rsi=52.0,
        )
        assert not result.triggered

    def test_stop_loss_above_entry_for_short(self):
        """Short SL should be above entry price."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="BTC/USDT",
            current_price=50000.0,
            ema_fast=50000.0,
            ema_slow=52000.0,
            rsi=52.0,
        )
        assert result.triggered
        assert result.stop_loss > result.entry_price
        assert result.take_profit < result.entry_price


# ===========================================================================
# Disabled / edge cases
# ===========================================================================

class TestEdgeCases:

    def test_disabled_signal_never_fires(self):
        """Returns not-triggered when enabled=False."""
        cfg = {**PULLBACK_CONF, "enabled": False}
        sig = EMAPullbackSignal(cfg)
        result = sig.evaluate_pullback(
            symbol="BTC/USDT",
            current_price=100.0,
            ema_fast=100.0,
            ema_slow=95.0,
            rsi=48.0,
        )
        assert not result.triggered

    def test_emas_equal_no_signal(self):
        """EMA20 == EMA50 (no trend) → no signal."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="ETH/USDT",
            current_price=3000.0,
            ema_fast=3000.0,
            ema_slow=3000.0,  # equal EMAs
            rsi=50.0,
        )
        assert not result.triggered

    def test_generic_evaluate_never_triggers(self):
        """The generic evaluate() stub always returns triggered=False."""
        from journal.models import OHLCV
        from datetime import datetime

        sig = EMAPullbackSignal(PULLBACK_CONF)
        candles = [
            OHLCV(
                timestamp=datetime(2025, 1, 1),
                open=100, high=101, low=99, close=100,
                volume=1000, symbol="TEST", timeframe="5m", market="stock",
            )
        ]
        result = sig.evaluate("TEST", candles, 100.0, "stock")
        assert not result.triggered

    def test_swing_bias_blocks_signal(self):
        """Swing advisor blocking → not triggered even with valid pullback."""
        with patch(
            "signals.ema_pullback.EMAPullbackSignal.check_swing_bias",
            return_value=False,
        ):
            sig = EMAPullbackSignal(PULLBACK_CONF)
            result = sig.evaluate_pullback(
                symbol="BTC/USDT",
                current_price=100.0,
                ema_fast=100.0,
                ema_slow=95.0,
                rsi=48.0,
            )
            assert not result.triggered

    def test_strategy_name(self):
        """Signal name property returns expected value."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        assert sig.name == "ema_pullback"

    def test_confidence_is_set(self):
        """Triggered result should have confidence > 0."""
        sig = EMAPullbackSignal(PULLBACK_CONF)
        result = sig.evaluate_pullback(
            symbol="SOL/USDT",
            current_price=150.0,
            ema_fast=150.0,
            ema_slow=140.0,
            rsi=48.0,
        )
        assert result.triggered
        assert result.confidence > 0
