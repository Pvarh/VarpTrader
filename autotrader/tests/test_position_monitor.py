"""Tests for the position monitor module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from journal.db import TradeDatabase
from journal.models import Trade
from execution.position_monitor import PositionMonitor


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def db(tmp_path) -> TradeDatabase:
    """Create a fresh SQLite database in a temp directory."""
    db_path = str(tmp_path / "test_monitor.db")
    database = TradeDatabase(db_path)
    yield database
    database.close()


@pytest.fixture
def mock_stock_feed() -> MagicMock:
    """Mock StockFeed with a controllable get_latest_price."""
    feed = MagicMock()
    feed.get_latest_price = MagicMock(return_value=0.0)
    return feed


@pytest.fixture
def mock_crypto_feed() -> MagicMock:
    """Mock CryptoFeed with a controllable get_latest_price."""
    feed = MagicMock()
    feed.get_latest_price = MagicMock(return_value=0.0)
    return feed


@pytest.fixture
def mock_stock_executor() -> MagicMock:
    """Mock AlpacaExecutor with a controllable close_position."""
    executor = MagicMock()
    executor.close_position = MagicMock(return_value={"id": "order-123", "status": "filled"})
    return executor


@pytest.fixture
def mock_crypto_executor() -> MagicMock:
    """Mock CryptoExecutor with a controllable close_position."""
    executor = MagicMock()
    executor.close_position = MagicMock(return_value={"id": "order-456", "status": "closed"})
    return executor


@pytest.fixture
def mock_telegram() -> MagicMock:
    """Mock TelegramAlert with a controllable send_trade_alert."""
    telegram = MagicMock()
    telegram.send_trade_alert = MagicMock(return_value=True)
    return telegram


@pytest.fixture
def monitor(
    db: TradeDatabase,
    mock_stock_feed: MagicMock,
    mock_crypto_feed: MagicMock,
    mock_stock_executor: MagicMock,
    mock_crypto_executor: MagicMock,
    mock_telegram: MagicMock,
) -> PositionMonitor:
    """Create a PositionMonitor with all mocked dependencies (paper_trade=True)."""
    return PositionMonitor(
        db=db,
        stock_feed=mock_stock_feed,
        crypto_feed=mock_crypto_feed,
        stock_executor=mock_stock_executor,
        crypto_executor=mock_crypto_executor,
        telegram=mock_telegram,
        paper_trade=True,
    )


@pytest.fixture
def live_monitor(
    db: TradeDatabase,
    mock_stock_feed: MagicMock,
    mock_crypto_feed: MagicMock,
    mock_stock_executor: MagicMock,
    mock_crypto_executor: MagicMock,
    mock_telegram: MagicMock,
) -> PositionMonitor:
    """Create a PositionMonitor with paper_trade=False for live-execution tests."""
    return PositionMonitor(
        db=db,
        stock_feed=mock_stock_feed,
        crypto_feed=mock_crypto_feed,
        stock_executor=mock_stock_executor,
        crypto_executor=mock_crypto_executor,
        telegram=mock_telegram,
        paper_trade=False,
    )


# ======================================================================
# Helper: insert an open trade and return its ID
# ======================================================================


def _insert_open_trade(
    db: TradeDatabase,
    symbol: str = "AAPL",
    market: str = "stock",
    direction: str = "long",
    entry_price: float = 100.0,
    quantity: float = 10.0,
    stop_loss: float = 95.0,
    take_profit: float = 110.0,
    strategy: str = "ema_cross",
) -> int:
    """Insert an open trade into the database and return its ID."""
    trade = Trade(
        symbol=symbol,
        market=market,
        strategy=strategy,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        timestamp="2025-06-15T10:30:00",
        outcome="open",
        paper_trade=1,
    )
    return db.insert_trade(trade)


# ======================================================================
# Stop-loss tests
# ======================================================================


class TestStopLossLong:
    """Stop-loss triggered for long trades."""

    def test_stop_loss_hit_exact(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """Price exactly at stop-loss should trigger exit for a long trade."""
        trade_id = _insert_open_trade(db, stop_loss=95.0, take_profit=110.0, entry_price=100.0)
        mock_stock_feed.get_latest_price.return_value = 95.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "loss"
        assert result["exit_price"] == 95.0
        mock_telegram.send_trade_alert.assert_called_once()

    def test_stop_loss_hit_below(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
    ) -> None:
        """Price below stop-loss should trigger exit for a long trade."""
        trade_id = _insert_open_trade(db, stop_loss=95.0, take_profit=110.0, entry_price=100.0)
        mock_stock_feed.get_latest_price.return_value = 90.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "loss"
        assert result["exit_price"] == 90.0


class TestStopLossShort:
    """Stop-loss triggered for short trades."""

    def test_stop_loss_hit_exact(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """Price exactly at stop-loss should trigger exit for a short trade."""
        trade_id = _insert_open_trade(
            db, direction="short", entry_price=100.0, stop_loss=105.0, take_profit=90.0,
        )
        mock_stock_feed.get_latest_price.return_value = 105.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "loss"
        assert result["exit_price"] == 105.0
        mock_telegram.send_trade_alert.assert_called_once()

    def test_stop_loss_hit_above(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
    ) -> None:
        """Price above stop-loss should trigger exit for a short trade."""
        trade_id = _insert_open_trade(
            db, direction="short", entry_price=100.0, stop_loss=105.0, take_profit=90.0,
        )
        mock_stock_feed.get_latest_price.return_value = 108.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "loss"
        assert result["exit_price"] == 108.0


# ======================================================================
# Take-profit tests
# ======================================================================


class TestTakeProfitLong:
    """Take-profit triggered for long trades."""

    def test_take_profit_hit_exact(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """Price exactly at take-profit should trigger exit for a long trade."""
        trade_id = _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        mock_stock_feed.get_latest_price.return_value = 110.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "win"
        assert result["exit_price"] == 110.0
        mock_telegram.send_trade_alert.assert_called_once()

    def test_take_profit_hit_above(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
    ) -> None:
        """Price above take-profit should trigger exit for a long trade."""
        trade_id = _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        mock_stock_feed.get_latest_price.return_value = 115.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "win"
        assert result["exit_price"] == 115.0


class TestTakeProfitShort:
    """Take-profit triggered for short trades."""

    def test_take_profit_hit_exact(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """Price exactly at take-profit should trigger exit for a short trade."""
        trade_id = _insert_open_trade(
            db, direction="short", entry_price=100.0, stop_loss=105.0, take_profit=90.0,
        )
        mock_stock_feed.get_latest_price.return_value = 90.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "win"
        assert result["exit_price"] == 90.0
        mock_telegram.send_trade_alert.assert_called_once()

    def test_take_profit_hit_below(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
    ) -> None:
        """Price below take-profit should trigger exit for a short trade."""
        trade_id = _insert_open_trade(
            db, direction="short", entry_price=100.0, stop_loss=105.0, take_profit=90.0,
        )
        mock_stock_feed.get_latest_price.return_value = 85.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "win"
        assert result["exit_price"] == 85.0


# ======================================================================
# No-action tests
# ======================================================================


class TestNoActionWithinBounds:
    """Price between SL and TP should not trigger any exit."""

    def test_long_trade_no_exit(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """Long trade with price between SL and TP remains open."""
        trade_id = _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        mock_stock_feed.get_latest_price.return_value = 102.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "open"
        assert result["exit_price"] is None
        mock_telegram.send_trade_alert.assert_not_called()

    def test_short_trade_no_exit(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """Short trade with price between SL and TP remains open."""
        trade_id = _insert_open_trade(
            db, direction="short", entry_price=100.0, stop_loss=105.0, take_profit=90.0,
        )
        mock_stock_feed.get_latest_price.return_value = 98.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "open"
        assert result["exit_price"] is None
        mock_telegram.send_trade_alert.assert_not_called()

    def test_no_open_trades(
        self, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """When there are no open trades, nothing should happen."""
        monitor.check_open_positions()

        mock_stock_feed.get_latest_price.assert_not_called()
        mock_telegram.send_trade_alert.assert_not_called()

    def test_price_unavailable_skips_trade(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """When the feed returns 0.0, the trade should be skipped."""
        _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        mock_stock_feed.get_latest_price.return_value = 0.0

        monitor.check_open_positions()

        mock_telegram.send_trade_alert.assert_not_called()


# ======================================================================
# Trailing stop tests
# ======================================================================


class TestTrailingStop:
    """Trailing stop tightens to breakeven when > 50 % to target."""

    def test_trailing_stop_tightens_long(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """Long trade > 50% to target should have stop tightened to entry.

        Entry=100, SL=95, TP=110. Target range=10. 50% threshold=105.
        Price at 106 (> 105) triggers trailing stop to 100 (breakeven).
        Price is still below TP (110), so trade stays open.
        """
        trade_id = _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        mock_stock_feed.get_latest_price.return_value = 106.0

        monitor.check_open_positions()

        # Trade should still be open since 106 < 110 (TP)
        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "open"

        # Verify trailing stop was recorded
        assert trade_id in monitor._trailing_stop_adjusted

    def test_trailing_stop_triggers_breakeven_exit_long(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """After trailing stop tightening, price dropping to entry triggers SL.

        Step 1: Price at 106 -- trailing stop tightens to 100 (breakeven).
        Step 2: Price drops to 100 -- now at the tightened stop, exit triggered.
        """
        trade_id = _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)

        # Step 1: Price above 50% threshold, trailing stop tightened
        mock_stock_feed.get_latest_price.return_value = 106.0
        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "open"

        # Step 2: Price drops to breakeven, tightened stop triggered
        mock_stock_feed.get_latest_price.return_value = 100.0
        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "breakeven"
        assert result["exit_price"] == 100.0

    def test_trailing_stop_tightens_short(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """Short trade > 50% to target should have stop tightened to entry.

        Entry=100, SL=105, TP=90. Target range=10. 50% threshold=95.
        Price at 94 (below 95) triggers trailing stop to 100 (breakeven).
        Price is still above TP (90), so trade stays open.
        """
        trade_id = _insert_open_trade(
            db, direction="short", entry_price=100.0, stop_loss=105.0, take_profit=90.0,
        )
        mock_stock_feed.get_latest_price.return_value = 94.0

        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "open"
        assert trade_id in monitor._trailing_stop_adjusted

    def test_trailing_stop_triggers_breakeven_exit_short(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
    ) -> None:
        """After trailing stop tightening on short, price rising to entry triggers SL.

        Step 1: Price at 94 -- trailing stop tightens to 100 (breakeven).
        Step 2: Price rises to 100 -- now at the tightened stop, exit triggered.
        """
        trade_id = _insert_open_trade(
            db, direction="short", entry_price=100.0, stop_loss=105.0, take_profit=90.0,
        )

        # Step 1: Price below 50% threshold, trailing stop tightened
        mock_stock_feed.get_latest_price.return_value = 94.0
        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "open"

        # Step 2: Price rises back to breakeven, tightened stop triggered
        mock_stock_feed.get_latest_price.return_value = 100.0
        monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "breakeven"
        assert result["exit_price"] == 100.0

    def test_no_trailing_stop_below_threshold(
        self, db: TradeDatabase, monitor: PositionMonitor, mock_stock_feed: MagicMock,
    ) -> None:
        """Trade under 50% to target should not have trailing stop activated.

        Entry=100, SL=95, TP=110. 50% threshold=105.
        Price at 103 (below threshold) -- no trailing stop adjustment.
        """
        trade_id = _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        mock_stock_feed.get_latest_price.return_value = 103.0

        monitor.check_open_positions()

        assert trade_id not in monitor._trailing_stop_adjusted


# ======================================================================
# PnL computation tests
# ======================================================================


class TestPnLComputation:
    """Verify PnL calculation correctness."""

    def test_long_win_pnl(self, monitor: PositionMonitor) -> None:
        """Long trade closed above entry should yield positive PnL."""
        trade: dict = {
            "entry_price": 100.0,
            "quantity": 10.0,
            "direction": "long",
        }
        pnl, pnl_pct, outcome = monitor._compute_trade_pnl(trade, exit_price=110.0)

        assert pnl == pytest.approx(100.0)   # (110 - 100) * 10
        assert pnl_pct == pytest.approx(0.1)  # (110 - 100) / 100
        assert outcome == "win"

    def test_long_loss_pnl(self, monitor: PositionMonitor) -> None:
        """Long trade closed below entry should yield negative PnL."""
        trade: dict = {
            "entry_price": 100.0,
            "quantity": 10.0,
            "direction": "long",
        }
        pnl, pnl_pct, outcome = monitor._compute_trade_pnl(trade, exit_price=95.0)

        assert pnl == pytest.approx(-50.0)     # (95 - 100) * 10
        assert pnl_pct == pytest.approx(-0.05)  # (95 - 100) / 100
        assert outcome == "loss"

    def test_short_win_pnl(self, monitor: PositionMonitor) -> None:
        """Short trade closed below entry should yield positive PnL."""
        trade: dict = {
            "entry_price": 100.0,
            "quantity": 10.0,
            "direction": "short",
        }
        pnl, pnl_pct, outcome = monitor._compute_trade_pnl(trade, exit_price=90.0)

        assert pnl == pytest.approx(100.0)   # (100 - 90) * 10
        assert pnl_pct == pytest.approx(0.1)  # (100 - 90) / 100
        assert outcome == "win"

    def test_short_loss_pnl(self, monitor: PositionMonitor) -> None:
        """Short trade closed above entry should yield negative PnL."""
        trade: dict = {
            "entry_price": 100.0,
            "quantity": 10.0,
            "direction": "short",
        }
        pnl, pnl_pct, outcome = monitor._compute_trade_pnl(trade, exit_price=108.0)

        assert pnl == pytest.approx(-80.0)     # (100 - 108) * 10
        assert pnl_pct == pytest.approx(-0.08)  # (100 - 108) / 100
        assert outcome == "loss"

    def test_breakeven_pnl(self, monitor: PositionMonitor) -> None:
        """Trade closed at entry price should be breakeven."""
        trade: dict = {
            "entry_price": 100.0,
            "quantity": 10.0,
            "direction": "long",
        }
        pnl, pnl_pct, outcome = monitor._compute_trade_pnl(trade, exit_price=100.0)

        assert pnl == pytest.approx(0.0)
        assert pnl_pct == pytest.approx(0.0)
        assert outcome == "breakeven"

    def test_fractional_quantity_pnl(self, monitor: PositionMonitor) -> None:
        """PnL should work correctly with fractional quantities (crypto)."""
        trade: dict = {
            "entry_price": 50000.0,
            "quantity": 0.5,
            "direction": "long",
        }
        pnl, pnl_pct, outcome = monitor._compute_trade_pnl(trade, exit_price=52000.0)

        assert pnl == pytest.approx(1000.0)    # (52000 - 50000) * 0.5
        assert pnl_pct == pytest.approx(0.04)   # (52000 - 50000) / 50000
        assert outcome == "win"


# ======================================================================
# Crypto market tests
# ======================================================================


class TestCryptoMarket:
    """Ensure the monitor routes crypto trades to the correct feed/executor."""

    def test_crypto_stop_loss_uses_crypto_feed(
        self, db: TradeDatabase, monitor: PositionMonitor,
        mock_crypto_feed: MagicMock, mock_stock_feed: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """Crypto trade should query CryptoFeed, not StockFeed."""
        trade_id = _insert_open_trade(
            db, symbol="BTC/USDT", market="crypto",
            entry_price=50000.0, stop_loss=48000.0, take_profit=55000.0,
        )
        mock_crypto_feed.get_latest_price.return_value = 47000.0

        monitor.check_open_positions()

        mock_crypto_feed.get_latest_price.assert_called_once_with("BTC/USDT")
        mock_stock_feed.get_latest_price.assert_not_called()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "loss"

    def test_crypto_live_close_calls_crypto_executor(
        self, db: TradeDatabase, live_monitor: PositionMonitor,
        mock_crypto_feed: MagicMock, mock_crypto_executor: MagicMock,
        mock_stock_executor: MagicMock,
    ) -> None:
        """Live crypto close should call CryptoExecutor.close_position."""
        _insert_open_trade(
            db, symbol="ETH/USDT", market="crypto",
            direction="long", entry_price=3000.0,
            stop_loss=2800.0, take_profit=3500.0, quantity=2.0,
        )
        mock_crypto_feed.get_latest_price.return_value = 2700.0

        live_monitor.check_open_positions()

        mock_crypto_executor.close_position.assert_called_once_with(
            symbol="ETH/USDT", side="sell", amount=2.0,
        )
        mock_stock_executor.close_position.assert_not_called()


# ======================================================================
# Live execution tests
# ======================================================================


class TestLiveExecution:
    """Tests with paper_trade=False to verify executor interaction."""

    def test_live_stock_close_calls_executor(
        self, db: TradeDatabase, live_monitor: PositionMonitor,
        mock_stock_feed: MagicMock, mock_stock_executor: MagicMock,
    ) -> None:
        """Live stock close should call AlpacaExecutor.close_position."""
        _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        mock_stock_feed.get_latest_price.return_value = 93.0

        live_monitor.check_open_positions()

        mock_stock_executor.close_position.assert_called_once_with("AAPL")

    def test_live_close_failure_does_not_update_db(
        self, db: TradeDatabase, live_monitor: PositionMonitor,
        mock_stock_feed: MagicMock, mock_stock_executor: MagicMock,
        mock_telegram: MagicMock,
    ) -> None:
        """If executor returns None (failure), trade should remain open."""
        trade_id = _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        mock_stock_feed.get_latest_price.return_value = 93.0
        mock_stock_executor.close_position.return_value = None

        live_monitor.check_open_positions()

        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "open"
        assert result["exit_price"] is None
        mock_telegram.send_trade_alert.assert_not_called()

    def test_paper_trade_does_not_call_executor(
        self, db: TradeDatabase, monitor: PositionMonitor,
        mock_stock_feed: MagicMock, mock_stock_executor: MagicMock,
    ) -> None:
        """Paper-trade mode should update DB without calling executor."""
        trade_id = _insert_open_trade(db, entry_price=100.0, stop_loss=95.0, take_profit=110.0)
        mock_stock_feed.get_latest_price.return_value = 93.0

        monitor.check_open_positions()

        mock_stock_executor.close_position.assert_not_called()
        result = db.get_trade_by_id(trade_id)
        assert result["outcome"] == "loss"


# ======================================================================
# Multiple trades in single check
# ======================================================================


class TestMultipleTrades:
    """Ensure the monitor handles multiple open trades correctly."""

    def test_mixed_outcomes(
        self, db: TradeDatabase, monitor: PositionMonitor,
        mock_stock_feed: MagicMock, mock_telegram: MagicMock,
    ) -> None:
        """Multiple trades: one hits SL, one hits TP, one stays open."""
        # Trade 1: long, will hit SL (price=93 < SL=95)
        t1_id = _insert_open_trade(
            db, symbol="AAPL", entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        )
        # Trade 2: long, will hit TP (price=115 > TP=110)
        t2_id = _insert_open_trade(
            db, symbol="MSFT", entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        )
        # Trade 3: long, stays open (price=102 between SL and TP)
        t3_id = _insert_open_trade(
            db, symbol="GOOG", entry_price=100.0, stop_loss=95.0, take_profit=110.0,
        )

        # Return different prices per symbol
        def price_for_symbol(symbol: str) -> float:
            prices = {"AAPL": 93.0, "MSFT": 115.0, "GOOG": 102.0}
            return prices.get(symbol, 0.0)

        mock_stock_feed.get_latest_price.side_effect = price_for_symbol

        monitor.check_open_positions()

        r1 = db.get_trade_by_id(t1_id)
        r2 = db.get_trade_by_id(t2_id)
        r3 = db.get_trade_by_id(t3_id)

        assert r1["outcome"] == "loss"
        assert r2["outcome"] == "win"
        assert r3["outcome"] == "open"

        # Two alerts sent (SL + TP), not three
        assert mock_telegram.send_trade_alert.call_count == 2


# ======================================================================
# Alert content tests
# ======================================================================


class TestAlertContent:
    """Verify that Telegram alerts contain the correct trade details."""

    def test_exit_alert_has_pnl(
        self, db: TradeDatabase, monitor: PositionMonitor,
        mock_stock_feed: MagicMock, mock_telegram: MagicMock,
    ) -> None:
        """Exit alert should include PnL and correct action/direction."""
        _insert_open_trade(
            db, symbol="AAPL", entry_price=100.0, quantity=10.0,
            stop_loss=95.0, take_profit=110.0, strategy="ema_cross",
        )
        mock_stock_feed.get_latest_price.return_value = 110.0

        monitor.check_open_positions()

        alert_call = mock_telegram.send_trade_alert.call_args
        alert_dict = alert_call[0][0]  # first positional argument

        assert alert_dict["action"] == "EXIT"
        assert alert_dict["symbol"] == "AAPL"
        assert alert_dict["direction"] == "LONG"
        assert alert_dict["price"] == 110.0
        assert alert_dict["quantity"] == 10.0
        assert alert_dict["strategy"] == "ema_cross"
        assert alert_dict["pnl"] == pytest.approx(100.0)
