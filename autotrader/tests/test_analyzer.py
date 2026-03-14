"""Tests for the analysis module."""

import json
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, ".")

from journal.db import TradeDatabase
from journal.models import Trade


@pytest.fixture
def db(tmp_path) -> TradeDatabase:
    """Create a test database with sample trades."""
    db_path = str(tmp_path / "test_analysis.db")
    database = TradeDatabase(db_path)

    strategies = ["first_candle", "ema_cross", "vwap_reversion", "rsi_momentum", "bollinger_fade"]
    outcomes = ["win", "win", "loss", "win", "loss", "win", "win", "loss", "win", "win"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    for i in range(20):
        ts = (datetime(2025, 1, 1) + timedelta(hours=i * 3)).isoformat()
        pnl_val = 50.0 if outcomes[i % len(outcomes)] == "win" else -30.0
        trade = Trade(
            symbol="AAPL",
            market="stock",
            strategy=strategies[i % len(strategies)],
            direction="long",
            entry_price=150.0,
            quantity=10,
            stop_loss=148.5,
            take_profit=153.0,
            timestamp=ts,
            pnl=pnl_val,
            pnl_pct=pnl_val / 1500.0,
            outcome=outcomes[i % len(outcomes)],
            whale_flag=1 if i % 4 == 0 else 0,
            day_of_week=days[i % len(days)],
            hour_of_day=(9 + i) % 24,
            market_condition="trending" if i % 3 == 0 else "ranging",
        )
        database.insert_trade(trade)

    yield database
    database.close()


class TestAnalyzer:
    """Tests for the performance analyzer."""

    def test_win_rate_per_strategy(self, db: TradeDatabase) -> None:
        """Analyzer should compute win rate for each strategy."""
        from analysis.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer(db)
        stats = analyzer.compute_strategy_win_rates(lookback_days=3650)
        assert isinstance(stats, dict)
        assert len(stats) > 0
        for strategy, rate in stats.items():
            assert 0.0 <= rate <= 1.0

    def test_win_rate_by_day(self, db: TradeDatabase) -> None:
        """Analyzer should break down win rate by day of week."""
        from analysis.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer(db)
        stats = analyzer.compute_win_rate_by_day()
        assert isinstance(stats, dict)

    def test_win_rate_by_hour(self, db: TradeDatabase) -> None:
        """Analyzer should break down win rate by hour."""
        from analysis.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer(db)
        stats = analyzer.compute_win_rate_by_hour()
        assert isinstance(stats, dict)

    def test_average_pnl_per_strategy(self, db: TradeDatabase) -> None:
        """Analyzer should compute average PnL per strategy."""
        from analysis.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer(db)
        stats = analyzer.compute_avg_pnl_per_strategy()
        assert isinstance(stats, dict)
        assert len(stats) > 0

    def test_whale_correlation(self, db: TradeDatabase) -> None:
        """Analyzer should compute whale flag correlation."""
        from analysis.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer(db)
        stats = analyzer.compute_whale_correlation()
        assert "with_whale" in stats
        assert "without_whale" in stats

    def test_worst_strategy(self, db: TradeDatabase) -> None:
        """Analyzer should identify the worst performing strategy."""
        from analysis.analyzer import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer(db)
        worst = analyzer.get_worst_strategy(lookback_days=3650)
        assert worst is not None
        assert isinstance(worst, str)

    def test_empty_database_handled(self, tmp_path) -> None:
        """Analyzer should handle empty database gracefully."""
        from analysis.analyzer import PerformanceAnalyzer

        empty_db = TradeDatabase(str(tmp_path / "empty.db"))
        analyzer = PerformanceAnalyzer(empty_db)
        stats = analyzer.compute_strategy_win_rates()
        assert stats == {}
        empty_db.close()
