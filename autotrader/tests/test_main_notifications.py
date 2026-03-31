"""Tests for Telegram-facing main-loop notifications."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import AutoTrader
from signals.base_signal import SignalDirection, SignalResult


def test_send_heartbeat_includes_daily_trade_count() -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader._start_time = time.time() - 3700
    trader.paper_trade = False
    trader.paper_executor = None
    trader.db = MagicMock()
    trader.db.get_daily_pnl.return_value = 123.45
    trader.db.get_daily_trade_count.return_value = 7
    trader.telegram = MagicMock()

    AutoTrader.send_heartbeat(trader)

    sent_text = trader.telegram.send_message.call_args.args[0]
    assert "Trades Today: 7" in sent_text
    assert "Daily P&L: $+123" in sent_text


def test_check_signal_starvation_triggers_overseer_once_per_episode() -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader._last_signal_time = time.time() - (5 * 3600)
    trader._starvation_alerted = False
    trader._starvation_overseer_triggered = False
    trader.telegram = MagicMock()
    trader.trigger_overseer_async = MagicMock(return_value="running")

    AutoTrader.check_signal_starvation(trader)
    AutoTrader.check_signal_starvation(trader)

    trader.trigger_overseer_async.assert_called_once_with("signal_starvation")
    assert trader.telegram.send_message.call_count == 2
    assert trader._starvation_alerted is True
    assert trader._starvation_overseer_triggered is True


def test_trigger_overseer_async_queues_when_claude_missing(tmp_path: Path) -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader._overseer_run_in_progress = False
    trigger_path = tmp_path / "overseer_trigger.json"

    with patch("main.OVERSEER_TRIGGER_PATH", trigger_path), patch("main.shutil.which", return_value=None):
        status = AutoTrader.trigger_overseer_async(trader, "signal_starvation")

    assert status == "queued"
    payload = json.loads(trigger_path.read_text(encoding="utf-8"))
    assert payload["trigger_reason"] == "signal_starvation"
    assert payload["model"] == "claude-sonnet-4-6"


def test_run_nightly_analysis_sends_change_summary() -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader.config = {
        "analysis": {
            "lookback_days": 30,
            "min_trades_for_suggestion": 15,
            "auto_apply_changes": True,
        },
        "risk": {
            "stop_loss_pct": 0.015,
        },
    }
    trader.analyzer = MagicMock()
    trader.analyzer.build_full_report.return_value = {"total_trades": 24}
    trader.llm_advisor = MagicMock()
    trader.llm_advisor.get_recommendations.return_value = {"stop_loss_pct": 0.02}
    trader.config_updater = MagicMock()
    trader.config_updater.validate_changes.return_value = (
        {"stop_loss_pct": 0.02},
        [{"param": "rsi_overbought", "value": 85, "reason": "above maximum 80"}],
    )
    trader.report_builder = MagicMock()
    trader.report_builder.build_daily_report.return_value = "daily report"
    trader.telegram = MagicMock()
    trader.db = MagicMock()

    with patch("main.load_config", return_value={"risk": {"stop_loss_pct": 0.02}}), patch(
        "main._resolve_claude_auth_mode", return_value="api"
    ):
        AutoTrader.run_nightly_analysis(trader)

    trader.telegram.send_daily_report.assert_called_once_with("daily report")
    summary_text = trader.telegram.send_message.call_args.args[0]
    assert "NIGHTLY CONFIG UPDATE" in summary_text
    assert "Applied: 1" in summary_text
    assert "stop_loss_pct: 0.015 -> 0.02" in summary_text
    assert "Rejected: 1" in summary_text


def test_run_nightly_analysis_queues_host_when_login_mode_needs_host(tmp_path: Path) -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader.config = {"analysis": {"model": "claude-sonnet-4-6"}}
    trader.telegram = MagicMock()
    trigger_path = tmp_path / "nightly_analysis_trigger.json"

    with patch("main.NIGHTLY_ANALYSIS_TRIGGER_PATH", trigger_path), patch(
        "main._resolve_claude_auth_mode", return_value="login"
    ), patch("main.shutil.which", return_value=None):
        AutoTrader.run_nightly_analysis(trader)

    payload = json.loads(trigger_path.read_text(encoding="utf-8"))
    assert payload["trigger_reason"] == "nightly_analysis"
    sent_text = trader.telegram.send_message.call_args.args[0]
    assert "NIGHTLY ANALYSIS QUEUED" in sent_text


def test_poll_telegram_commands_executes_plain_text_manual_buy() -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader.config = {
        "watchlist": {"stocks": ["NVDA"], "crypto": []},
        "risk": {"position_size_pct": 0.005},
    }
    trader.paper_trade = True
    trader.paper_portfolio = SimpleNamespace(equity=100_000.0, cash=100_000.0)
    trader.paper_executor = MagicMock()
    trader.paper_executor.submit_market_order.return_value = {
        "qty": "1",
        "fill_price": 900.0,
    }
    trader.stock_feed = MagicMock()
    trader.stock_feed.get_latest_price.return_value = 900.0
    trader.crypto_feed = MagicMock()
    trader.db = MagicMock()
    trader.db.get_open_trades.return_value = []
    trader.db.get_daily_trade_count.return_value = 0
    trader.order_validator = MagicMock()
    trader.order_validator.validate.return_value = (True, "")
    trader.kill_switch = SimpleNamespace(halted=False)
    trader._daily_trade_count = 0
    trader.telegram = MagicMock()
    trader.telegram._chat_id = "42"
    trader.telegram.get_updates.return_value = [
        {"message": {"text": "buy nvdia", "chat": {"id": "42"}}}
    ]

    AutoTrader.poll_telegram_commands(trader)

    trader.paper_executor.submit_market_order.assert_called_once()
    order_call = trader.paper_executor.submit_market_order.call_args.kwargs
    assert order_call["symbol"] == "NVDA"
    assert order_call["strategy"] == "manual_telegram"
    sent_text = trader.telegram.send_message.call_args.args[0]
    assert "MANUAL BUY EXECUTED" in sent_text
    assert "NVDA" in sent_text


def test_poll_telegram_commands_executes_manual_sell() -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader.config = {
        "watchlist": {"stocks": ["NVDA"], "crypto": []},
        "risk": {"position_size_pct": 0.005},
    }
    trader.paper_trade = True
    trader.paper_portfolio = SimpleNamespace(equity=100_000.0, cash=99_000.0)
    trader.paper_executor = MagicMock()
    trader.paper_executor.get_positions.return_value = [{"symbol": "NVDA"}]
    trader.paper_executor.close_position.return_value = {
        "fill_price": 910.0,
        "pnl": 10.0,
    }
    trader.stock_feed = MagicMock()
    trader.stock_feed.get_latest_price.return_value = 910.0
    trader.crypto_feed = MagicMock()
    trader.kill_switch = SimpleNamespace(halted=False)
    trader.telegram = MagicMock()
    trader.telegram._chat_id = "42"
    trader.telegram.get_updates.return_value = [
        {"message": {"text": "/sell nvda", "chat": {"id": "42"}}}
    ]

    AutoTrader.poll_telegram_commands(trader)

    trader.paper_executor.close_position.assert_called_once_with(
        symbol="NVDA",
        market_price=910.0,
        market="stock",
    )
    sent_text = trader.telegram.send_message.call_args.args[0]
    assert "MANUAL SELL EXECUTED" in sent_text
    assert "PnL: $+10.00" in sent_text


def test_directional_vwap_filter_only_applies_to_trend_following_strategies() -> None:
    assert AutoTrader._uses_directional_vwap_filter("ema_cross") is False
    assert AutoTrader._uses_directional_vwap_filter("ema_pullback") is True
    assert AutoTrader._uses_directional_vwap_filter("first_candle") is True
    assert AutoTrader._uses_directional_vwap_filter("rsi_momentum") is False
    assert AutoTrader._uses_directional_vwap_filter("bollinger_fade") is False
    assert AutoTrader._uses_directional_vwap_filter("vwap_reversion") is False


def test_duplicate_signal_attempt_is_suppressed() -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader._recent_signal_attempts = {}
    result = SignalResult(
        triggered=True,
        strategy_name="ema_pullback",
        direction=SignalDirection.SHORT,
        entry_price=100.0,
        stop_loss=101.0,
        take_profit=98.0,
    )

    AutoTrader._remember_signal_attempt(trader, "NVDA", result)

    assert AutoTrader._is_duplicate_signal_attempt(trader, "NVDA", result) is True


def test_loss_cooldown_remaining_seconds_uses_recent_losses() -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader.paper_trade = True
    trader.db = MagicMock()
    trader.db.get_recent_closed_trades.return_value = [
        {"outcome": "loss", "exit_timestamp": "2026-03-25T18:00:00+00:00"},
        {"outcome": "loss", "exit_timestamp": "2026-03-25T17:30:00+00:00"},
    ]

    fake_now = datetime(2026, 3, 25, 18, 30, tzinfo=timezone.utc)
    with patch("main.datetime") as mock_datetime:
        mock_datetime.now.return_value = fake_now
        mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
        remaining = AutoTrader._loss_cooldown_remaining_seconds(
            trader,
            symbol="TSLA",
            strategy_name="bollinger_fade",
        )

    assert remaining > 0


def test_stock_signal_ignores_session_bias_when_disabled() -> None:
    trader = AutoTrader.__new__(AutoTrader)
    trader.config = {
        "session_bias": {"enabled": False},
    }
    trader.stock_feed = MagicMock()
    trader.stock_feed.get_historical_bars.side_effect = [
        [SimpleNamespace(close=100.0)],
        [],
    ]
    trader.stock_feed.get_average_volume.return_value = 1_000_000.0
    trader.stock_feed.get_first_candle.return_value = None
    trader.regime_detector = MagicMock()
    trader.regime_detector.detect.return_value = "trending_up"
    trader.session_bias = MagicMock()
    trader.session_bias.should_block.return_value = True
    trader.block_detector = MagicMock()
    trader.block_detector.has_sell_flag.return_value = False
    trader.signals = [MagicMock(name="signal")]
    trader.signals[0].is_enabled.return_value = True
    trader.signals[0].name = "rsi_momentum"
    trader._process_signal = MagicMock()
    trader._last_signal_time = 0.0
    trader._starvation_alerted = True
    trader._starvation_overseer_triggered = True
    trader._run_signal = MagicMock(
        return_value=SignalResult(
            triggered=True,
            strategy_name="rsi_momentum",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=102.0,
        )
    )

    AutoTrader._evaluate_stock_symbol(
        trader,
        symbol="NVDA",
        account_value=100_000.0,
        current_positions=0,
    )

    trader._process_signal.assert_called_once()
    trader.session_bias.should_block.assert_not_called()


def test_reload_config_rebuilds_runtime_components_when_changed() -> None:
    old_config = {
        "watchlist": {"stocks": ["NVDA"], "crypto": ["BTC/USDT"]},
        "strategies": {
            "first_candle": {"enabled": True, "orb_window_minutes": 60, "volume_multiplier": 1.5, "valid_until_hour": 13},
            "ema_cross": {"enabled": True, "fast_ema": 20, "slow_ema": 50, "volume_confirmation": True},
            "ema_pullback": {"enabled": True, "fast_ema": 20, "slow_ema": 50, "pullback_pct": 0.002, "rsi_max_long": 55, "rsi_min_short": 45, "stop_loss_pct": 0.015, "volume_confirmation": True},
            "vwap_reversion": {"enabled": False, "vwap_deviation_pct": 0.3},
            "rsi_momentum": {"enabled": False, "rsi_oversold": 25, "rsi_overbought": 75, "rsi_neutral_low": 45, "rsi_neutral_high": 55},
            "bollinger_fade": {"enabled": False, "bb_period": 20, "bb_std": 2.0, "rsi_threshold_low": 35, "rsi_threshold_high": 65},
        },
        "risk": {
            "position_size_pct": 0.005,
            "stop_loss_pct": 0.015,
            "daily_loss_limit_pct": 0.03,
            "min_reward_ratio": 2.0,
            "atr_period": 14,
            "atr_multiplier": 1.5,
            "max_positions": 10,
            "max_daily_trades": 20,
        },
        "whale": {
            "enabled": True,
            "stock_block_trades_enabled": False,
            "stock_block_shares": 50000,
            "stock_block_usd": 2_000_000,
            "crypto_transfer_usd": 1_000_000,
            "flag_ttl_minutes": 15,
            "sell_pressure_confirmations": 4,
            "max_sell_pressure_block_minutes": 10,
        },
        "regime": {"adx_threshold": 15.0},
        "polymarket": {"block_short_threshold": 0.65},
    }
    new_config = json.loads(json.dumps(old_config))
    new_config["strategies"]["bollinger_fade"]["enabled"] = True
    new_config["strategies"]["rsi_momentum"]["enabled"] = True
    new_config["regime"]["adx_threshold"] = 30.0
    new_config["risk"]["max_positions"] = 6

    trader = AutoTrader.__new__(AutoTrader)
    trader.config = old_config
    trader.db = MagicMock()
    trader.analyzer = MagicMock()
    trader.kill_switch = SimpleNamespace(halted=True)
    trader.paper_executor = None
    trader.stock_feed = MagicMock()
    trader.crypto_feed = MagicMock()

    with patch("main.load_config", return_value=new_config), patch("main.dashboard_init") as mock_dashboard:
        AutoTrader.reload_config(trader)

    assert trader.config["strategies"]["bollinger_fade"]["enabled"] is True
    assert trader.config["strategies"]["rsi_momentum"]["enabled"] is True
    assert trader.regime_detector._adx_threshold == 30.0
    assert trader.kill_switch.halted is True
    assert any(sig.name == "bollinger_fade" and sig.is_enabled() for sig in trader.signals)
    assert any(sig.name == "rsi_momentum" and sig.is_enabled() for sig in trader.signals)
    mock_dashboard.assert_called_once()
