"""Tests for the trade journal database module."""

import sys
from datetime import datetime

import pytest

sys.path.insert(0, ".")

from journal.db import TradeDatabase
from journal.models import Trade, AnalysisRun


@pytest.fixture
def db(tmp_path) -> TradeDatabase:
    """Create a fresh in-memory-like database in a temp directory."""
    db_path = str(tmp_path / "test_trades.db")
    database = TradeDatabase(db_path)
    yield database
    database.close()


@pytest.fixture
def sample_trade() -> Trade:
    """Create a sample trade for testing."""
    return Trade(
        symbol="AAPL",
        market="stock",
        strategy="ema_cross",
        direction="long",
        entry_price=150.0,
        quantity=10,
        stop_loss=148.5,
        take_profit=153.0,
        timestamp="2025-01-15T10:30:00",
        whale_flag=0,
        paper_trade=1,
    )


class TestTradeInsertAndRetrieve:
    """Tests for inserting and retrieving trades."""

    def test_insert_trade_returns_id(self, db: TradeDatabase, sample_trade: Trade) -> None:
        """Inserting a trade should return a positive integer ID."""
        trade_id = db.insert_trade(sample_trade)
        assert trade_id is not None
        assert trade_id > 0

    def test_get_trade_by_id(self, db: TradeDatabase, sample_trade: Trade) -> None:
        """Should retrieve the exact trade by ID."""
        trade_id = db.insert_trade(sample_trade)
        result = db.get_trade_by_id(trade_id)
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["strategy"] == "ema_cross"
        assert result["entry_price"] == 150.0

    def test_get_nonexistent_trade(self, db: TradeDatabase) -> None:
        """Should return None for a non-existent trade ID."""
        result = db.get_trade_by_id(9999)
        assert result is None

    def test_get_open_trades(self, db: TradeDatabase, sample_trade: Trade) -> None:
        """Should return only trades with outcome='open'."""
        db.insert_trade(sample_trade)
        open_trades = db.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["outcome"] == "open"


class TestTradeUpdate:
    """Tests for updating trade exits."""

    def test_update_trade_exit(self, db: TradeDatabase, sample_trade: Trade) -> None:
        """Updating exit should set exit_price, pnl, and outcome."""
        trade_id = db.insert_trade(sample_trade)
        db.update_trade_exit(trade_id, exit_price=152.0, pnl=20.0, pnl_pct=0.0133, outcome="win")
        result = db.get_trade_by_id(trade_id)
        assert result["exit_price"] == 152.0
        assert result["pnl"] == 20.0
        assert result["outcome"] == "win"

    def test_closed_trade_not_in_open_list(self, db: TradeDatabase, sample_trade: Trade) -> None:
        """After closing, trade should not appear in open trades."""
        trade_id = db.insert_trade(sample_trade)
        db.update_trade_exit(trade_id, exit_price=152.0, pnl=20.0, pnl_pct=0.0133, outcome="win")
        open_trades = db.get_open_trades()
        assert len(open_trades) == 0


class TestDailyPnL:
    """Tests for daily PnL calculation."""

    def test_daily_pnl_sum(self, db: TradeDatabase) -> None:
        """Daily PnL should sum all trades for that date."""
        t1 = Trade(
            symbol="AAPL", market="stock", strategy="ema_cross", direction="long",
            entry_price=150.0, quantity=10, stop_loss=148.0, take_profit=154.0,
            timestamp="2025-01-15T10:00:00", pnl=50.0, outcome="win",
        )
        t2 = Trade(
            symbol="MSFT", market="stock", strategy="vwap_reversion", direction="long",
            entry_price=300.0, quantity=5, stop_loss=297.0, take_profit=306.0,
            timestamp="2025-01-15T14:00:00", pnl=-30.0, outcome="loss",
        )
        db.insert_trade(t1)
        db.insert_trade(t2)
        total = db.get_daily_pnl("2025-01-15")
        assert total == pytest.approx(20.0)

    def test_daily_pnl_no_trades(self, db: TradeDatabase) -> None:
        """Should return 0.0 for a day with no trades."""
        total = db.get_daily_pnl("2025-12-31")
        assert total == 0.0


class TestAnalysisRun:
    """Tests for analysis run tracking."""

    def test_insert_analysis_run(self, db: TradeDatabase) -> None:
        """Should insert and return an analysis run ID."""
        run = AnalysisRun(
            trades_analyzed=50,
            report_markdown="# Report\nGood day",
            config_changes_json='{"stop_loss_pct": 0.02}',
            approved=1,
        )
        run_id = db.insert_analysis_run(run)
        assert run_id is not None
        assert run_id > 0


class TestParameterizedQueries:
    """Ensure no SQL injection is possible."""

    def test_symbol_with_sql_injection_attempt(self, db: TradeDatabase) -> None:
        """Symbol containing SQL should be safely parameterized."""
        trade = Trade(
            symbol="'; DROP TABLE trades; --",
            market="stock", strategy="test", direction="long",
            entry_price=100.0, quantity=1, stop_loss=99.0, take_profit=102.0,
        )
        trade_id = db.insert_trade(trade)
        result = db.get_trade_by_id(trade_id)
        assert result is not None
        assert result["symbol"] == "'; DROP TABLE trades; --"
        # Table still exists
        open_trades = db.get_open_trades()
        assert isinstance(open_trades, list)
