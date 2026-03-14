"""Tests for risk management modules."""

import sys

import pytest

sys.path.insert(0, ".")


# ===========================================================================
# Position Sizer
# ===========================================================================

class TestPositionSizer:
    """Tests for ATR-based position sizing."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "risk": {
                "position_size_pct": 0.02,
                "atr_period": 14,
                "atr_multiplier": 1.5,
                "stop_loss_pct": 0.015,
            }
        }

    def test_position_size_scales_with_atr(self, config: dict) -> None:
        """Higher ATR (more volatile) should yield smaller position size."""
        from risk.position_sizer import PositionSizer

        sizer = PositionSizer(config["risk"])
        account_value = 100_000.0
        # Use a lower entry price so the 10% value cap doesn't flatten both results
        size_low_vol = sizer.calculate_size(
            account_value=account_value, entry_price=50.0, atr=5.0,
        )
        size_high_vol = sizer.calculate_size(
            account_value=account_value, entry_price=50.0, atr=50.0,
        )
        assert size_high_vol < size_low_vol

    def test_position_size_respects_max_pct(self, config: dict) -> None:
        """Position notional should never exceed 10% of account value."""
        from risk.position_sizer import PositionSizer

        sizer = PositionSizer(config["risk"])
        account_value = 100_000.0
        entry_price = 100.0
        size = sizer.calculate_size(
            account_value=account_value, entry_price=entry_price, atr=0.01,
        )
        assert size * entry_price <= account_value * 0.10

    def test_zero_atr_returns_safe_default(self, config: dict) -> None:
        """ATR=0 should not cause division by zero; return a safe value."""
        from risk.position_sizer import PositionSizer

        sizer = PositionSizer(config["risk"])
        size = sizer.calculate_size(
            account_value=100_000.0, entry_price=100.0, atr=0.0,
        )
        assert size > 0


# ===========================================================================
# Kill Switch
# ===========================================================================

class TestKillSwitch:
    """Tests for daily drawdown kill switch."""

    @pytest.fixture
    def config(self) -> dict:
        return {"daily_loss_limit_pct": 0.03}

    def test_trading_allowed_above_limit(self, config: dict) -> None:
        """Trading continues when loss is below threshold."""
        from risk.kill_switch import KillSwitch

        ks = KillSwitch(config)
        assert ks.is_trading_allowed(daily_pnl=-200.0, account_value=100_000.0) is True

    def test_trading_halted_at_limit(self, config: dict) -> None:
        """Trading halts when daily loss equals the 3% threshold."""
        from risk.kill_switch import KillSwitch

        ks = KillSwitch(config)
        assert ks.is_trading_allowed(daily_pnl=-3000.0, account_value=100_000.0) is False

    def test_trading_halted_beyond_limit(self, config: dict) -> None:
        """Trading halts when daily loss exceeds the 3% threshold."""
        from risk.kill_switch import KillSwitch

        ks = KillSwitch(config)
        assert ks.is_trading_allowed(daily_pnl=-5000.0, account_value=100_000.0) is False

    def test_positive_pnl_allowed(self, config: dict) -> None:
        """Positive PnL should always allow trading."""
        from risk.kill_switch import KillSwitch

        ks = KillSwitch(config)
        assert ks.is_trading_allowed(daily_pnl=500.0, account_value=100_000.0) is True


# ===========================================================================
# Reward Ratio Gate
# ===========================================================================

class TestRewardRatio:
    """Tests for the minimum risk:reward ratio checker."""

    def test_acceptable_ratio(self) -> None:
        """2:1 reward:risk should pass with min_ratio=2.0."""
        from risk.reward_ratio import RewardRatioGate

        gate = RewardRatioGate(min_ratio=2.0)
        assert gate.check(
            entry_price=100.0, stop_loss=99.0, take_profit=102.0
        ) is True

    def test_unacceptable_ratio(self) -> None:
        """1:1 reward:risk should fail with min_ratio=2.0."""
        from risk.reward_ratio import RewardRatioGate

        gate = RewardRatioGate(min_ratio=2.0)
        assert gate.check(
            entry_price=100.0, stop_loss=99.0, take_profit=101.0
        ) is False

    def test_short_trade_ratio(self) -> None:
        """Short trade reward:risk calculation should be correct."""
        from risk.reward_ratio import RewardRatioGate

        gate = RewardRatioGate(min_ratio=2.0)
        # Short: entry=100, stop=101 (risk=1), target=98 (reward=2)
        assert gate.check(
            entry_price=100.0, stop_loss=101.0, take_profit=98.0
        ) is True

    def test_zero_risk_rejected(self) -> None:
        """Stop loss equal to entry (zero risk) should be rejected."""
        from risk.reward_ratio import RewardRatioGate

        gate = RewardRatioGate(min_ratio=2.0)
        assert gate.check(
            entry_price=100.0, stop_loss=100.0, take_profit=102.0
        ) is False
