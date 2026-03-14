"""Tests for the new dashboard router with HTML pages and WebSocket endpoints."""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

# Ensure the autotrader package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_config(tmp_path):
    """Write a minimal config.json and weekly_bias.json to a temp directory."""
    cfg = {
        "watchlist": {
            "stocks": ["AAPL", "TSLA"],
            "crypto": ["BTC/USDT"],
        },
        "strategies": {
            "ema_cross": {"enabled": True, "fast_ema": 50, "slow_ema": 200},
            "rsi_momentum": {"enabled": True, "rsi_oversold": 32, "rsi_overbought": 68},
        },
        "risk": {
            "position_size_pct": 0.01,
            "stop_loss_pct": 0.015,
            "daily_loss_limit_pct": 0.03,
            "min_reward_ratio": 2.0,
            "atr_period": 14,
        },
        "whale": {
            "enabled": True,
            "stock_block_shares": 50000,
        },
        "swing_advisor": {"enabled": True, "min_confidence_to_block": 60},
        "analysis": {
            "nightly_run_time": "23:30",
            "auto_apply_changes": False,
        },
        "go_live_thresholds": {
            "orb_min_win_rate": 0.56,
        },
        "bounds": {
            "stop_loss_pct": {"min": 0.005, "max": 0.05},
            "position_size_pct": {"min": 0.005, "max": 0.03},
            "rsi_oversold": {"min": 20, "max": 40},
            "rsi_overbought": {"min": 60, "max": 80},
        },
        "dashboard": {"host": "0.0.0.0", "port": 8000},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # weekly_bias.json
    bias = {"week_start": "2026-03-09", "generated_at": "2026-03-09T10:00:00", "biases": {}}
    bias_path = tmp_path / "weekly_bias.json"
    bias_path.write_text(json.dumps(bias, indent=2), encoding="utf-8")

    return tmp_path


@pytest.fixture()
def sample_db(tmp_path):
    """Create a temp TradeDatabase with sample trade data."""
    from journal.db import TradeDatabase
    from journal.models import Trade, AnalysisRun

    db_path = str(tmp_path / "test_trades.db")
    db = TradeDatabase(db_path)

    now = datetime.now(timezone.utc)

    # Insert some closed trades
    trades = [
        Trade(
            symbol="AAPL",
            market="stock",
            strategy="ema_cross",
            direction="long",
            entry_price=150.0,
            quantity=10,
            stop_loss=147.0,
            take_profit=156.0,
            timestamp=(now - timedelta(days=2)).isoformat(),
            exit_price=155.0,
            pnl=50.0,
            pnl_pct=0.0333,
            outcome="win",
            whale_flag=1,
            paper_trade=1,
        ),
        Trade(
            symbol="MSFT",
            market="stock",
            strategy="rsi_momentum",
            direction="long",
            entry_price=400.0,
            quantity=5,
            stop_loss=395.0,
            take_profit=410.0,
            timestamp=(now - timedelta(days=1)).isoformat(),
            exit_price=394.0,
            pnl=-30.0,
            pnl_pct=-0.015,
            outcome="loss",
            whale_flag=0,
            paper_trade=1,
        ),
        Trade(
            symbol="TSLA",
            market="stock",
            strategy="ema_cross",
            direction="short",
            entry_price=250.0,
            quantity=8,
            stop_loss=255.0,
            take_profit=240.0,
            timestamp=(now - timedelta(days=1)).isoformat(),
            exit_price=242.0,
            pnl=64.0,
            pnl_pct=0.032,
            outcome="win",
            whale_flag=0,
            paper_trade=1,
        ),
    ]

    for t in trades:
        db.insert_trade(t)

    # Insert an open trade (today)
    open_trade = Trade(
        symbol="NVDA",
        market="stock",
        strategy="vwap_reversion",
        direction="long",
        entry_price=800.0,
        quantity=3,
        stop_loss=790.0,
        take_profit=820.0,
        timestamp=now.isoformat(),
        outcome="open",
        whale_flag=1,
        paper_trade=1,
    )
    db.insert_trade(open_trade)

    # Insert a pending (unapproved) analysis run with config changes
    pending_run = AnalysisRun(
        run_timestamp=now.isoformat(),
        trades_analyzed=10,
        report_markdown="## Test Report\nSome analysis results here.",
        config_changes_json=json.dumps({"stop_loss_pct": 0.02}),
        approved=0,
    )
    db.insert_analysis_run(pending_run)

    return db


@pytest.fixture()
def client(sample_config, sample_db):
    """Create a TestClient with the dashboard router mounted on a test app."""
    from analysis.analyzer import PerformanceAnalyzer
    import dashboard.router as router_module

    analyzer = PerformanceAnalyzer(sample_db)

    # Inject dependencies into the router module
    router_module.init(sample_db, analyzer, {})

    # Patch config/bias file loading to use temp directory
    original_load_config = router_module._load_config_from_disk
    original_load_bias = router_module._load_weekly_bias

    config_path = sample_config / "config.json"
    bias_path = sample_config / "weekly_bias.json"

    def patched_load_config():
        with open(config_path, encoding="utf-8") as fh:
            return json.load(fh)

    def patched_load_bias():
        with open(bias_path, encoding="utf-8") as fh:
            return json.load(fh)

    router_module._load_config_from_disk = patched_load_config
    router_module._load_weekly_bias = patched_load_bias

    # Create test FastAPI app and mount router
    app = FastAPI()

    # Mount static files
    static_dir = Path(__file__).resolve().parent.parent / "dashboard" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(router_module.router)

    test_client = TestClient(app)

    yield test_client

    # Restore originals
    router_module._load_config_from_disk = original_load_config
    router_module._load_weekly_bias = original_load_bias


# ---------------------------------------------------------------------------
# Tests: HTML Pages
# ---------------------------------------------------------------------------

class TestDashboardPage:
    def test_dashboard_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "AutoTrader Dashboard" in body

    def test_dashboard_contains_sections(self, client):
        resp = client.get("/")
        body = resp.text
        assert "Open Positions" in body
        assert "Recent Trades" in body
        assert "Equity Curve" in body

    def test_dashboard_has_chart(self, client):
        resp = client.get("/")
        body = resp.text
        assert "<canvas" in body
        assert "equityChart" in body


class TestTradesPage:
    def test_trades_returns_html(self, client):
        resp = client.get("/trades")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "Trade Journal" in body

    def test_trades_with_filters(self, client):
        resp = client.get("/trades?strategy=ema_cross&days=30")
        assert resp.status_code == 200
        body = resp.text
        assert "Trade Journal" in body

    def test_trades_with_outcome_filter(self, client):
        resp = client.get("/trades?outcome=win")
        assert resp.status_code == 200


class TestAnalysisPage:
    def test_analysis_returns_html(self, client):
        resp = client.get("/analysis")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "Strategy Win Rates" in body
        assert "Whale Signal Correlation" in body


class TestConfigPage:
    def test_config_returns_html(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert "Parameter Bounds" in body


# ---------------------------------------------------------------------------
# Tests: JSON APIs
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_json(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "daily_pnl" in data
        assert "open_positions" in data
        assert "kill_switch_active" in data

    def test_health_timestamp_is_iso(self, client):
        resp = client.get("/health")
        data = resp.json()
        dt = datetime.fromisoformat(data["timestamp"])
        assert dt.year >= 2024


class TestWhaleEndpoint:
    def test_whale_returns_json(self, client):
        resp = client.get("/whale")
        assert resp.status_code == 200
        data = resp.json()
        # Initially empty dict
        assert isinstance(data, dict)


class TestSignalsEndpoint:
    def test_signals_returns_json_array(self, client):
        resp = client.get("/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# Tests: Config Approve endpoint
# ---------------------------------------------------------------------------

class TestConfigApprove:
    def test_approve_without_api_key_returns_403(self, client):
        resp = client.post("/config/approve/1")
        assert resp.status_code == 403

    def test_approve_with_wrong_api_key_returns_403(self, client):
        with patch.dict(os.environ, {"DASHBOARD_API_KEY": "correct-key"}):
            resp = client.post(
                "/config/approve/1",
                headers={"X-API-Key": "wrong-key"},
            )
            assert resp.status_code == 403

    def test_approve_with_correct_key_succeeds(self, client):
        with patch.dict(os.environ, {"DASHBOARD_API_KEY": "test-secret-key"}):
            resp = client.post(
                "/config/approve/1",
                headers={"X-API-Key": "test-secret-key"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "approved"
            assert data["run_id"] == 1

    def test_approve_already_approved(self, client):
        with patch.dict(os.environ, {"DASHBOARD_API_KEY": "test-secret-key"}):
            # First approval
            client.post(
                "/config/approve/1",
                headers={"X-API-Key": "test-secret-key"},
            )
            # Second approval
            resp = client.post(
                "/config/approve/1",
                headers={"X-API-Key": "test-secret-key"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "already_approved"

    def test_approve_nonexistent_run_returns_404(self, client):
        with patch.dict(os.environ, {"DASHBOARD_API_KEY": "test-secret-key"}):
            resp = client.post(
                "/config/approve/9999",
                headers={"X-API-Key": "test-secret-key"},
            )
            assert resp.status_code == 404
