"""Tests for the paper trading engine."""

import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, ".")

from execution.paper_engine import PaperExecutor, PaperPortfolio, PaperPosition
from journal.db import TradeDatabase


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def db(tmp_path) -> TradeDatabase:
    """Create a fresh database in a temp directory."""
    db_path = str(tmp_path / "test_paper.db")
    database = TradeDatabase(db_path)
    yield database
    database.close()


@pytest.fixture
def portfolio() -> PaperPortfolio:
    """Create a portfolio with 100k starting capital."""
    return PaperPortfolio(initial_capital=100_000.0)


@pytest.fixture
def executor(portfolio: PaperPortfolio, db: TradeDatabase) -> PaperExecutor:
    """Create a paper executor with default 0.1% slippage."""
    return PaperExecutor(portfolio=portfolio, db=db, slippage_pct=0.001)


@pytest.fixture
def zero_slip_executor(portfolio: PaperPortfolio, db: TradeDatabase) -> PaperExecutor:
    """Create a paper executor with zero slippage for exact PnL tests."""
    return PaperExecutor(portfolio=portfolio, db=db, slippage_pct=0.0)


# =====================================================================
# PaperPosition unit tests
# =====================================================================

class TestPaperPosition:
    """Tests for the PaperPosition dataclass properties."""

    def test_long_position_market_value(self) -> None:
        """Market value equals quantity * current price."""
        pos = PaperPosition(
            symbol="AAPL", market="stock", side="long",
            quantity=10, entry_price=150.0, current_price=155.0,
        )
        assert pos.market_value == pytest.approx(1550.0)

    def test_long_position_unrealized_pnl_profit(self) -> None:
        """Long PnL is positive when price increases."""
        pos = PaperPosition(
            symbol="AAPL", market="stock", side="long",
            quantity=10, entry_price=150.0, current_price=155.0,
        )
        assert pos.unrealized_pnl == pytest.approx(50.0)

    def test_long_position_unrealized_pnl_loss(self) -> None:
        """Long PnL is negative when price decreases."""
        pos = PaperPosition(
            symbol="AAPL", market="stock", side="long",
            quantity=10, entry_price=150.0, current_price=145.0,
        )
        assert pos.unrealized_pnl == pytest.approx(-50.0)

    def test_short_position_unrealized_pnl_profit(self) -> None:
        """Short PnL is positive when price decreases."""
        pos = PaperPosition(
            symbol="TSLA", market="stock", side="short",
            quantity=5, entry_price=200.0, current_price=190.0,
        )
        assert pos.unrealized_pnl == pytest.approx(50.0)

    def test_short_position_unrealized_pnl_loss(self) -> None:
        """Short PnL is negative when price increases."""
        pos = PaperPosition(
            symbol="TSLA", market="stock", side="short",
            quantity=5, entry_price=200.0, current_price=210.0,
        )
        assert pos.unrealized_pnl == pytest.approx(-50.0)

    def test_unrealized_pnl_pct_long(self) -> None:
        """Long PnL percentage is (current - entry) / entry."""
        pos = PaperPosition(
            symbol="AAPL", market="stock", side="long",
            quantity=10, entry_price=100.0, current_price=110.0,
        )
        assert pos.unrealized_pnl_pct == pytest.approx(0.10)

    def test_unrealized_pnl_pct_short(self) -> None:
        """Short PnL percentage is (entry - current) / entry."""
        pos = PaperPosition(
            symbol="TSLA", market="stock", side="short",
            quantity=5, entry_price=200.0, current_price=180.0,
        )
        assert pos.unrealized_pnl_pct == pytest.approx(0.10)

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict() should include all key fields."""
        pos = PaperPosition(
            symbol="MSFT", market="stock", side="long",
            quantity=20, entry_price=300.0, current_price=310.0,
        )
        d = pos.to_dict()
        assert d["symbol"] == "MSFT"
        assert d["market"] == "stock"
        assert d["side"] == "long"
        assert d["quantity"] == 20
        assert d["entry_price"] == 300.0
        assert d["current_price"] == 310.0
        assert "market_value" in d
        assert "unrealized_pnl" in d
        assert "unrealized_pnl_pct" in d


# =====================================================================
# Open long position, verify portfolio updates
# =====================================================================

class TestOpenLongPosition:
    """Opening a long position should update portfolio cash and positions."""

    def test_open_long_reduces_cash(
        self, executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Buying shares should reduce available cash."""
        initial_cash = portfolio.cash
        order = executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        assert order is not None
        assert portfolio.cash < initial_cash

    def test_open_long_creates_position(
        self, executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """A buy order should create a long position."""
        executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        assert portfolio.has_position("AAPL")
        pos = portfolio.get_position("AAPL")
        assert pos is not None
        assert pos["side"] == "long"
        assert pos["quantity"] == 10

    def test_open_long_order_result_fields(
        self, executor: PaperExecutor,
    ) -> None:
        """Order result should contain expected fields."""
        order = executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        assert order is not None
        assert order["status"] == "filled"
        assert order["type"] == "market"
        assert order["symbol"] == "AAPL"
        assert order["direction"] == "long"
        assert "id" in order
        assert "trade_id" in order
        assert "fill_price" in order


# =====================================================================
# Open and close position, verify PnL
# =====================================================================

class TestOpenClosePositionPnL:
    """Opening and closing a position should compute correct PnL."""

    def test_profitable_long_trade(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Buy low, sell high should yield positive PnL."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        close_result = zero_slip_executor.close_position(
            symbol="AAPL", market_price=160.0,
        )
        assert close_result is not None
        assert close_result["pnl"] == pytest.approx(100.0)
        assert close_result["outcome"] == "win"

    def test_losing_long_trade(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Buy high, sell low should yield negative PnL."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        close_result = zero_slip_executor.close_position(
            symbol="AAPL", market_price=140.0,
        )
        assert close_result is not None
        assert close_result["pnl"] == pytest.approx(-100.0)
        assert close_result["outcome"] == "loss"

    def test_breakeven_trade(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Closing at the same price should be breakeven."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        close_result = zero_slip_executor.close_position(
            symbol="AAPL", market_price=150.0,
        )
        assert close_result is not None
        assert close_result["pnl"] == pytest.approx(0.0)
        assert close_result["outcome"] == "breakeven"

    def test_cash_restored_after_close(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Cash should return to initial + PnL after closing."""
        initial_cash = portfolio.cash
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        zero_slip_executor.close_position(symbol="AAPL", market_price=110.0)
        # Started with 100k, made 10*10 = 100 profit
        assert portfolio.cash == pytest.approx(initial_cash + 100.0)

    def test_position_removed_after_close(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Closed position should no longer exist in portfolio."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        zero_slip_executor.close_position(symbol="AAPL", market_price=155.0)
        assert not portfolio.has_position("AAPL")

    def test_realized_pnl_accumulated(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Realized PnL should accumulate across trades."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        zero_slip_executor.close_position(symbol="AAPL", market_price=110.0)

        zero_slip_executor.submit_market_order(
            symbol="MSFT", side="buy", quantity=5,
            market_price=200.0, market="stock", strategy="test",
        )
        zero_slip_executor.close_position(symbol="MSFT", market_price=220.0)

        # 10 * 10 + 5 * 20 = 200
        assert portfolio.realized_pnl == pytest.approx(200.0)


# =====================================================================
# Slippage applied correctly on buy/sell
# =====================================================================

class TestSlippage:
    """Slippage should be applied correctly to fills."""

    def test_buy_slippage_increases_price(
        self, executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Buy fill price should be higher than market price."""
        order = executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        assert order is not None
        expected_fill = 100.0 * 1.001  # 100.1
        assert order["fill_price"] == pytest.approx(expected_fill)
        pos = portfolio.get_position("AAPL")
        assert pos["entry_price"] == pytest.approx(expected_fill)

    def test_sell_slippage_decreases_price(
        self, executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Sell (close) fill price should be lower than market price."""
        executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        close_result = executor.close_position(
            symbol="AAPL", market_price=110.0,
        )
        assert close_result is not None
        expected_sell_fill = 110.0 * (1.0 - 0.001)  # 109.89
        assert close_result["fill_price"] == pytest.approx(expected_sell_fill)

    def test_slippage_reduces_profit(
        self, portfolio: PaperPortfolio, db: TradeDatabase,
    ) -> None:
        """Round-trip slippage should reduce net profit vs zero slippage."""
        # With slippage
        exec_slip = PaperExecutor(portfolio, db, slippage_pct=0.01)
        exec_slip.submit_market_order(
            symbol="AAPL", side="buy", quantity=100,
            market_price=100.0, market="stock", strategy="test",
        )
        close_result = exec_slip.close_position(
            symbol="AAPL", market_price=110.0,
        )
        slippage_pnl = close_result["pnl"]

        # Without slippage the PnL would be 100 * (110 - 100) = 1000
        # Slippage eats into profit: buy at 101, sell at 108.9
        assert slippage_pnl < 1000.0
        assert slippage_pnl > 0.0  # still profitable

    def test_custom_slippage_rate(
        self, portfolio: PaperPortfolio, db: TradeDatabase,
    ) -> None:
        """Custom slippage rate should be applied correctly."""
        exec_high_slip = PaperExecutor(portfolio, db, slippage_pct=0.05)
        order = exec_high_slip.submit_market_order(
            symbol="AAPL", side="buy", quantity=1,
            market_price=100.0, market="stock", strategy="test",
        )
        assert order is not None
        assert order["fill_price"] == pytest.approx(105.0)


# =====================================================================
# Short position PnL computation
# =====================================================================

class TestShortPositionPnL:
    """Short positions should compute PnL correctly."""

    def test_profitable_short(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Short sell high, buy back low should be profitable."""
        zero_slip_executor.submit_market_order(
            symbol="TSLA", side="sell", quantity=5,
            market_price=200.0, market="stock", strategy="test",
        )
        assert portfolio.has_position("TSLA")
        pos = portfolio.get_position("TSLA")
        assert pos["side"] == "short"

        close_result = zero_slip_executor.close_position(
            symbol="TSLA", market_price=180.0,
        )
        assert close_result is not None
        # Short PnL = (entry - exit) * qty = (200 - 180) * 5 = 100
        assert close_result["pnl"] == pytest.approx(100.0)
        assert close_result["outcome"] == "win"

    def test_losing_short(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Short sell low, buy back high should be a loss."""
        zero_slip_executor.submit_market_order(
            symbol="TSLA", side="sell", quantity=5,
            market_price=200.0, market="stock", strategy="test",
        )
        close_result = zero_slip_executor.close_position(
            symbol="TSLA", market_price=220.0,
        )
        assert close_result is not None
        # Short PnL = (200 - 220) * 5 = -100
        assert close_result["pnl"] == pytest.approx(-100.0)
        assert close_result["outcome"] == "loss"

    def test_short_unrealized_pnl_positive(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Unrealized PnL for a short should be positive when price drops."""
        zero_slip_executor.submit_market_order(
            symbol="TSLA", side="sell", quantity=10,
            market_price=200.0, market="stock", strategy="test",
        )
        portfolio.update_price("TSLA", 190.0)
        assert portfolio.unrealized_pnl == pytest.approx(100.0)

    def test_short_unrealized_pnl_negative(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Unrealized PnL for a short should be negative when price rises."""
        zero_slip_executor.submit_market_order(
            symbol="TSLA", side="sell", quantity=10,
            market_price=200.0, market="stock", strategy="test",
        )
        portfolio.update_price("TSLA", 210.0)
        assert portfolio.unrealized_pnl == pytest.approx(-100.0)


# =====================================================================
# Multiple positions tracked simultaneously
# =====================================================================

class TestMultiplePositions:
    """Multiple positions should be tracked independently."""

    def test_multiple_positions_tracked(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Should hold multiple positions simultaneously."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        zero_slip_executor.submit_market_order(
            symbol="MSFT", side="buy", quantity=5,
            market_price=300.0, market="stock", strategy="test",
        )
        zero_slip_executor.submit_market_order(
            symbol="BTC/USDT", side="buy", quantity=0.5,
            market_price=50000.0, market="crypto", strategy="test",
        )
        positions = portfolio.get_positions()
        assert len(positions) == 3
        symbols = {p["symbol"] for p in positions}
        assert symbols == {"AAPL", "MSFT", "BTC/USDT"}

    def test_close_one_keeps_others(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Closing one position should not affect others."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        zero_slip_executor.submit_market_order(
            symbol="MSFT", side="buy", quantity=5,
            market_price=300.0, market="stock", strategy="test",
        )
        zero_slip_executor.close_position(symbol="AAPL", market_price=155.0)

        assert not portfolio.has_position("AAPL")
        assert portfolio.has_position("MSFT")
        positions = portfolio.get_positions()
        assert len(positions) == 1

    def test_mixed_long_short_positions(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Should handle a mix of long and short positions."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        zero_slip_executor.submit_market_order(
            symbol="TSLA", side="sell", quantity=5,
            market_price=200.0, market="stock", strategy="test",
        )
        aapl = portfolio.get_position("AAPL")
        tsla = portfolio.get_position("TSLA")
        assert aapl["side"] == "long"
        assert tsla["side"] == "short"

    def test_update_prices_affects_all_positions(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """update_prices should update all matching positions."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        zero_slip_executor.submit_market_order(
            symbol="MSFT", side="buy", quantity=5,
            market_price=300.0, market="stock", strategy="test",
        )
        zero_slip_executor.update_prices({"AAPL": 160.0, "MSFT": 310.0})

        aapl = portfolio.get_position("AAPL")
        msft = portfolio.get_position("MSFT")
        assert aapl["current_price"] == 160.0
        assert msft["current_price"] == 310.0


# =====================================================================
# Equity calculation with open positions
# =====================================================================

class TestEquityCalculation:
    """Equity should correctly reflect cash + unrealized PnL."""

    def test_equity_equals_capital_at_start(
        self, portfolio: PaperPortfolio,
    ) -> None:
        """Equity should equal initial capital with no positions."""
        assert portfolio.equity == pytest.approx(100_000.0)

    def test_equity_unchanged_immediately_after_buy(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Equity should remain constant right after opening a position
        (no price movement yet)."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        # Cash = 99_000, market_value of position = 1_000
        # equity = cash + market_value = 100_000
        assert portfolio.equity == pytest.approx(100_000.0)

    def test_equity_increases_with_profitable_position(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Equity should increase when positions are profitable."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        portfolio.update_price("AAPL", 110.0)
        # Gained 10 * 10 = 100
        assert portfolio.equity == pytest.approx(100_100.0)

    def test_equity_decreases_with_losing_position(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Equity should decrease when positions are in the red."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        portfolio.update_price("AAPL", 90.0)
        # Lost 10 * 10 = 100
        assert portfolio.equity == pytest.approx(99_900.0)

    def test_equity_with_multiple_positions(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Equity reflects combined PnL of all positions."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        zero_slip_executor.submit_market_order(
            symbol="MSFT", side="buy", quantity=5,
            market_price=200.0, market="stock", strategy="test",
        )
        portfolio.update_prices({"AAPL": 110.0, "MSFT": 190.0})
        # AAPL: +10 * 10 = +100
        # MSFT: -10 * 5  = -50
        # Net: +50
        assert portfolio.equity == pytest.approx(100_050.0)


# =====================================================================
# Cannot open position with insufficient cash
# =====================================================================

class TestInsufficientCash:
    """Orders exceeding available cash should be rejected."""

    def test_reject_order_exceeding_cash(
        self, executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Should return None when position cost exceeds buying power."""
        order = executor.submit_market_order(
            symbol="BRK.A", side="buy", quantity=1,
            market_price=500_000.0, market="stock", strategy="test",
        )
        assert order is None

    def test_cash_unchanged_on_rejected_order(
        self, executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Cash should not change when an order is rejected."""
        initial_cash = portfolio.cash
        executor.submit_market_order(
            symbol="BRK.A", side="buy", quantity=1,
            market_price=500_000.0, market="stock", strategy="test",
        )
        assert portfolio.cash == pytest.approx(initial_cash)

    def test_no_position_on_rejected_order(
        self, executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """No position should be created when an order is rejected."""
        executor.submit_market_order(
            symbol="BRK.A", side="buy", quantity=1,
            market_price=500_000.0, market="stock", strategy="test",
        )
        assert not portfolio.has_position("BRK.A")

    def test_partial_capital_then_insufficient(
        self, executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """After using most capital, next large order should be scaled down."""
        executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=600,
            market_price=150.0, market="stock", strategy="test",
        )
        # Used ~90k, only ~10k left -- order for 100 @ 300 = 30k, scales down
        order = executor.submit_market_order(
            symbol="MSFT", side="buy", quantity=100,
            market_price=300.0, market="stock", strategy="test",
        )
        assert order is not None
        assert portfolio.has_position("AAPL")
        assert portfolio.has_position("MSFT")
        # Quantity should be scaled down to fit available cash
        msft_pos = portfolio.get_position("MSFT")
        assert msft_pos["quantity"] < 100


# =====================================================================
# get_account and get_positions return correct format
# =====================================================================

class TestAccountAndPositionsFormat:
    """get_account() and get_positions() should match AlpacaExecutor format."""

    def test_get_account_format(
        self, executor: PaperExecutor,
    ) -> None:
        """get_account() should return all required fields."""
        account = executor.get_account()
        assert account is not None
        assert "equity" in account
        assert "buying_power" in account
        assert "cash" in account
        assert "portfolio_value" in account
        assert "currency" in account
        assert isinstance(account["equity"], float)
        assert isinstance(account["buying_power"], float)
        assert isinstance(account["cash"], float)
        assert account["currency"] == "USD"

    def test_get_account_initial_values(
        self, executor: PaperExecutor,
    ) -> None:
        """Account should reflect initial capital with no positions."""
        account = executor.get_account()
        assert account["equity"] == pytest.approx(100_000.0)
        assert account["buying_power"] == pytest.approx(100_000.0)
        assert account["cash"] == pytest.approx(100_000.0)

    def test_get_account_after_trade(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Account should update after a trade."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        account = zero_slip_executor.get_account()
        # Cash reduced by cost of shares
        assert account["cash"] == pytest.approx(99_000.0)

    def test_get_positions_empty(
        self, executor: PaperExecutor,
    ) -> None:
        """get_positions() should return empty list with no trades."""
        positions = executor.get_positions()
        assert positions == []

    def test_get_positions_format(
        self, zero_slip_executor: PaperExecutor,
    ) -> None:
        """get_positions() should match AlpacaExecutor format."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        positions = zero_slip_executor.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos["symbol"] == "AAPL"
        assert pos["qty"] == 10
        assert pos["side"] == "long"
        assert "market_value" in pos
        assert "avg_entry_price" in pos
        assert "unrealized_pl" in pos
        assert "unrealized_plpc" in pos
        assert pos["avg_entry_price"] == pytest.approx(150.0)
        assert pos["market_value"] == pytest.approx(1500.0)

    def test_get_positions_unrealized_pnl(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Position unrealized PnL should update with price changes."""
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="test",
        )
        portfolio.update_price("AAPL", 160.0)
        positions = zero_slip_executor.get_positions()
        pos = positions[0]
        assert pos["unrealized_pl"] == pytest.approx(100.0)
        assert pos["unrealized_plpc"] == pytest.approx(100.0 / 1500.0)


# =====================================================================
# Journal integration
# =====================================================================

class TestJournalIntegration:
    """Paper trades should be properly journaled in the database."""

    def test_trade_journaled_on_open(
        self, zero_slip_executor: PaperExecutor, db: TradeDatabase,
    ) -> None:
        """Opening a position should insert a trade record."""
        order = zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="ema_cross",
        )
        trade_id = order["trade_id"]
        record = db.get_trade_by_id(trade_id)
        assert record is not None
        assert record["symbol"] == "AAPL"
        assert record["direction"] == "long"
        assert record["paper_trade"] == 1
        assert record["outcome"] == "open"
        assert record["strategy"] == "ema_cross"

    def test_trade_updated_on_close(
        self, zero_slip_executor: PaperExecutor, db: TradeDatabase,
    ) -> None:
        """Closing a position should update the trade record."""
        order = zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=150.0, market="stock", strategy="ema_cross",
        )
        trade_id = order["trade_id"]
        zero_slip_executor.close_position(symbol="AAPL", market_price=160.0)

        record = db.get_trade_by_id(trade_id)
        assert record is not None
        assert record["exit_price"] == pytest.approx(160.0)
        assert record["pnl"] == pytest.approx(100.0)
        assert record["outcome"] == "win"


# =====================================================================
# Daily PnL
# =====================================================================

class TestDailyPnL:
    """get_daily_pnl should track intra-day performance."""

    def test_daily_pnl_zero_at_start(
        self, executor: PaperExecutor,
    ) -> None:
        """Daily PnL should be zero with no trades."""
        assert executor.get_daily_pnl() == pytest.approx(0.0)

    def test_daily_pnl_with_unrealized(
        self, zero_slip_executor: PaperExecutor, portfolio: PaperPortfolio,
    ) -> None:
        """Daily PnL should include unrealized gains."""
        portfolio.snapshot_day_start()
        zero_slip_executor.submit_market_order(
            symbol="AAPL", side="buy", quantity=10,
            market_price=100.0, market="stock", strategy="test",
        )
        portfolio.update_price("AAPL", 110.0)
        # Unrealized gain = 10 * 10 = 100
        assert zero_slip_executor.get_daily_pnl() == pytest.approx(100.0)


# =====================================================================
# Close non-existent position
# =====================================================================

class TestCloseNonExistent:
    """Closing a non-existent position should return None."""

    def test_close_nonexistent_returns_none(
        self, executor: PaperExecutor,
    ) -> None:
        """Should return None when no position exists."""
        result = executor.close_position(symbol="FAKE", market_price=100.0)
        assert result is None
