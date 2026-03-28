"""Paper trading engine that simulates order execution without real money.

Provides :class:`PaperPortfolio` for virtual position tracking and
:class:`PaperExecutor` as a drop-in replacement for
:class:`~execution.alpaca_executor.AlpacaExecutor` and
:class:`~execution.crypto_executor.CryptoExecutor` during development and
strategy validation.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from journal.db import TradeDatabase
from journal.models import Trade


# ---------------------------------------------------------------------
# Position model
# ---------------------------------------------------------------------
@dataclass
class PaperPosition:
    """A single paper trading position.

    Attributes
    ----------
    symbol:
        Instrument symbol (e.g. ``"AAPL"`` or ``"BTC/USDT"``).
    market:
        ``"stock"`` or ``"crypto"``.
    side:
        ``"long"`` or ``"short"``.
    quantity:
        Number of shares / contracts / coins held.
    entry_price:
        Average entry price.
    current_price:
        Most recent market price used for mark-to-market.
    timestamp:
        ISO-8601 timestamp of when the position was opened.
    trade_id:
        Database row ID from the journal, if recorded.
    """

    symbol: str
    market: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    trade_id: Optional[int] = None

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost at entry."""
        return abs(self.quantity) * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss in dollar terms."""
        if self.side == "long":
            return (self.current_price - self.entry_price) * abs(self.quantity)
        # short
        return (self.entry_price - self.current_price) * abs(self.quantity)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized PnL as a percentage of entry cost."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "long":
            return (self.current_price - self.entry_price) / self.entry_price
        # short
        return (self.entry_price - self.current_price) / self.entry_price

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "symbol": self.symbol,
            "market": self.market,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "timestamp": self.timestamp,
            "trade_id": self.trade_id,
        }


# ---------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------
class PaperPortfolio:
    """Tracks virtual positions, equity, and PnL for paper trading.

    All mutating operations are protected by a :class:`threading.Lock`
    so the portfolio is safe to use from multiple threads (e.g. a
    strategy thread and a monitoring thread).

    Parameters
    ----------
    initial_capital:
        Starting cash balance in the account currency.
    """

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        """Initialize with starting capital."""
        self._initial_capital: float = initial_capital
        self._cash: float = initial_capital
        self._positions: dict[str, PaperPosition] = {}  # symbol -> position
        self._trade_history: list[dict[str, Any]] = []
        self._realized_pnl: float = 0.0
        self._day_start_equity: float = initial_capital
        self._lock: threading.Lock = threading.Lock()

    # ----- properties ------------------------------------------------

    @property
    def cash(self) -> float:
        """Current cash balance."""
        with self._lock:
            return self._cash

    @property
    def equity(self) -> float:
        """Total equity = cash + sum of position market values."""
        with self._lock:
            position_value = sum(
                pos.market_value for pos in self._positions.values()
            )
            return self._cash + position_value

    @property
    def buying_power(self) -> float:
        """Available cash for new trades."""
        with self._lock:
            return self._cash

    @property
    def unrealized_pnl(self) -> float:
        """Sum of unrealized PnL across all open positions."""
        with self._lock:
            return sum(
                pos.unrealized_pnl for pos in self._positions.values()
            )

    @property
    def realized_pnl(self) -> float:
        """Cumulative realized PnL from all closed trades."""
        with self._lock:
            return self._realized_pnl

    @property
    def initial_capital(self) -> float:
        """The initial capital the portfolio was seeded with."""
        return self._initial_capital

    # ----- position queries ------------------------------------------

    def get_positions(self) -> list[dict[str, Any]]:
        """Return all open positions as dicts."""
        with self._lock:
            return [pos.to_dict() for pos in self._positions.values()]

    def get_position(self, symbol: str) -> Optional[dict[str, Any]]:
        """Return a specific position as a dict, or ``None``."""
        with self._lock:
            pos = self._positions.get(symbol)
            return pos.to_dict() if pos else None

    def has_position(self, symbol: str) -> bool:
        """Check if an open position exists for *symbol*."""
        with self._lock:
            return symbol in self._positions

    # ----- mutators (called by PaperExecutor) ------------------------

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        market: str,
        trade_id: Optional[int] = None,
    ) -> PaperPosition:
        """Open a new position and deduct cash.

        Parameters
        ----------
        symbol:
            Instrument symbol.
        side:
            ``"long"`` or ``"short"``.
        quantity:
            Number of units.
        fill_price:
            Simulated fill price (after slippage).
        market:
            ``"stock"`` or ``"crypto"``.
        trade_id:
            Optional journal database row ID.

        Returns
        -------
        PaperPosition
            The newly created position.

        Raises
        ------
        ValueError
            If there is insufficient cash to open the position.
        """
        cost = abs(quantity) * fill_price
        with self._lock:
            if cost > self._cash:
                raise ValueError(
                    f"Insufficient cash: need {cost:.2f}, "
                    f"have {self._cash:.2f}"
                )
            self._cash -= cost
            position = PaperPosition(
                symbol=symbol,
                market=market,
                side=side,
                quantity=abs(quantity),
                entry_price=fill_price,
                current_price=fill_price,
                trade_id=trade_id,
            )
            self._positions[symbol] = position
            logger.info(
                "paper_position_opened | symbol={symbol} side={side} quantity={quantity} fill_price={fill_price} cost={cost} cash_remaining={cash_remaining}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                fill_price=fill_price,
                cost=cost,
                cash_remaining=self._cash,
            )
            return position

    def close_position(
        self,
        symbol: str,
        fill_price: float,
    ) -> dict[str, Any]:
        """Close an existing position and credit cash.

        Parameters
        ----------
        symbol:
            Instrument symbol to close.
        fill_price:
            Simulated fill price (after slippage).

        Returns
        -------
        dict
            Summary including realized PnL.

        Raises
        ------
        KeyError
            If no position exists for *symbol*.
        """
        with self._lock:
            if symbol not in self._positions:
                raise KeyError(f"No open position for '{symbol}'")

            pos = self._positions.pop(symbol)
            pos.current_price = fill_price

            pnl = pos.unrealized_pnl
            proceeds = pos.cost_basis + pnl
            self._cash += proceeds
            self._realized_pnl += pnl

            result: dict[str, Any] = {
                "symbol": symbol,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "exit_price": fill_price,
                "pnl": pnl,
                "pnl_pct": pos.unrealized_pnl_pct,
                "trade_id": pos.trade_id,
            }
            self._trade_history.append(result)

            logger.info(
                "paper_position_closed | symbol={symbol} pnl={pnl} cash_after={cash_after}",
                symbol=symbol,
                pnl=pnl,
                cash_after=self._cash,
            )
            return result

    def update_price(self, symbol: str, price: float) -> None:
        """Update the current price for a single position.

        Parameters
        ----------
        symbol:
            Instrument symbol.
        price:
            Latest market price.
        """
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].current_price = price

    def update_prices(self, prices: dict[str, float]) -> None:
        """Bulk-update current prices for held positions.

        Parameters
        ----------
        prices:
            Mapping of symbol to current market price.
        """
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self._positions:
                    self._positions[symbol].current_price = price

    def snapshot_day_start(self) -> None:
        """Record current equity as the day-start baseline.

        Called at the beginning of each trading day so that
        :meth:`PaperExecutor.get_daily_pnl` can compute intra-day PnL.
        """
        self._day_start_equity = self.equity

    @property
    def day_start_equity(self) -> float:
        """Equity snapshot taken at the start of the current trading day."""
        return self._day_start_equity


# ---------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------
class PaperExecutor:
    """Simulated order execution engine for paper trading.

    Implements the same public interface as
    :class:`~execution.alpaca_executor.AlpacaExecutor` and
    :class:`~execution.crypto_executor.CryptoExecutor` but executes
    orders against a :class:`PaperPortfolio` instead of a real broker.

    Parameters
    ----------
    portfolio:
        Virtual portfolio to track positions and cash.
    db:
        Trade database for journaling fills.
    slippage_pct:
        Simulated slippage as a fraction (``0.001`` = 0.1 %).
    """

    def __init__(
        self,
        portfolio: PaperPortfolio,
        db: TradeDatabase,
        slippage_pct: float = 0.001,
    ) -> None:
        self._portfolio: PaperPortfolio = portfolio
        self._db: TradeDatabase = db
        self._slippage_pct: float = slippage_pct

        self._reload_open_positions()

        logger.info(
            "paper_executor_initialised | initial_capital={initial_capital} slippage_pct={slippage_pct}",
            initial_capital=portfolio.initial_capital,
            slippage_pct=slippage_pct,
        )

    # ----- helpers ---------------------------------------------------

    def _reset_portfolio_runtime_state(self) -> None:
        """Reset runtime-only portfolio state before journal replay."""
        with self._portfolio._lock:
            self._portfolio._cash = self._portfolio.initial_capital
            self._portfolio._positions = {}
            self._portfolio._trade_history = []
            self._portfolio._realized_pnl = 0.0
            self._portfolio._day_start_equity = self._portfolio.initial_capital

    def _reload_open_positions(self) -> None:
        """Rebuild the paper portfolio from the journal after restart."""
        try:
            history = self._db.get_paper_trade_history()
            if not history:
                return
            self._reset_portfolio_runtime_state()

            restored = 0
            duplicate_open = 0
            for trade in history:
                if not trade.get("paper_trade", 0):
                    continue

                symbol = str(trade["symbol"])
                exit_timestamp = trade.get("exit_timestamp")
                pnl = float(trade.get("pnl") or 0.0)

                if exit_timestamp:
                    with self._portfolio._lock:
                        self._portfolio._cash += pnl
                        self._portfolio._realized_pnl += pnl
                    continue

                if self._portfolio.has_position(symbol):
                    duplicate_open += 1
                    logger.warning(
                        "paper_reload_duplicate_open_trade_ignored | symbol={symbol} trade_id={trade_id}",
                        symbol=symbol,
                        trade_id=trade.get("id"),
                    )
                    continue

                cost = abs(float(trade["quantity"])) * float(trade["entry_price"])
                with self._portfolio._lock:
                    self._portfolio._cash -= cost
                    pos = PaperPosition(
                        symbol=symbol,
                        market=trade.get("market", "crypto"),
                        side=trade["direction"],
                        quantity=abs(float(trade["quantity"])),
                        entry_price=float(trade["entry_price"]),
                        current_price=float(trade["entry_price"]),
                        timestamp=trade.get("timestamp", ""),
                        trade_id=trade.get("id"),
                    )
                    self._portfolio._positions[symbol] = pos
                    restored += 1

            logger.info(
                "paper_positions_reloaded | count={count} duplicate_open={duplicate_open} cash_remaining={cash} realized_pnl={realized_pnl}",
                count=restored,
                duplicate_open=duplicate_open,
                cash=self._portfolio.cash,
                realized_pnl=self._portfolio.realized_pnl,
            )
            if self._portfolio.cash < 0:
                logger.warning(
                    "paper_reload_negative_cash | cash={cash} open_positions={count}",
                    cash=self._portfolio.cash,
                    count=restored,
                )
        except Exception:
            logger.exception("paper_reload_error")

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply simulated slippage to a price.

        - Buy orders fill *higher* (unfavorable).
        - Sell orders fill *lower* (unfavorable).

        Parameters
        ----------
        price:
            Raw market price.
        side:
            ``"buy"`` or ``"sell"``.

        Returns
        -------
        float
            Adjusted fill price.
        """
        if side.lower() == "buy":
            return price * (1.0 + self._slippage_pct)
        return price * (1.0 - self._slippage_pct)

    @staticmethod
    def _generate_order_id() -> str:
        """Generate a unique simulated order ID."""
        return f"paper-{uuid.uuid4().hex[:12]}"

    # ----- order submission ------------------------------------------

    def submit_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_price: float,
        market: str = "stock",
        strategy: str = "",
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Simulate a market order fill.

        Applies slippage to *market_price* and updates the portfolio.

        - **Buy**: ``fill_price = market_price * (1 + slippage_pct)``
        - **Sell**: ``fill_price = market_price * (1 - slippage_pct)``

        Parameters
        ----------
        symbol:
            Instrument symbol.
        side:
            ``"buy"`` or ``"sell"``.
        quantity:
            Number of units to trade.
        market_price:
            Current market price before slippage.
        market:
            ``"stock"`` or ``"crypto"``.
        strategy:
            Strategy name for journal tagging.
        **kwargs:
            Additional metadata (``stop_loss``, ``take_profit``, etc.).

        Returns
        -------
        dict | None
            Order result dictionary on success, or ``None`` on failure.
        """
        fill_price = self._apply_slippage(market_price, side)
        order_id = self._generate_order_id()
        now_iso = datetime.now(timezone.utc).isoformat()
        direction = "long" if side.lower() == "buy" else "short"

        stop_loss = kwargs.get("stop_loss", 0.0)
        take_profit = kwargs.get("take_profit", 0.0)
        whale_flag = kwargs.get("whale_flag", 0)

        try:
            # Scale down quantity if it exceeds available cash
            cost = abs(quantity) * fill_price
            available = self._portfolio.cash
            if cost > available:
                max_qty = available / fill_price
                if market == "crypto":
                    # Crypto allows fractional quantities
                    quantity = float(int(max_qty * 1e6) / 1e6)  # 6 decimal places
                else:
                    quantity = int(max_qty)
                if quantity <= 0:
                    raise ValueError(
                        f"Insufficient cash: need {cost:.2f}, "
                        f"have {available:.2f}"
                    )
                logger.info(
                    "paper_order_scaled_down | symbol={symbol} original_cost={original_cost} "
                    "available_cash={available} new_qty={new_qty}",
                    symbol=symbol, original_cost=round(cost, 2),
                    available=round(available, 2), new_qty=quantity,
                )

            # Journal the trade entry
            trade = Trade(
                symbol=symbol,
                market=market,
                strategy=strategy,
                direction=direction,
                entry_price=fill_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                whale_flag=whale_flag,
                timestamp=now_iso,
                paper_trade=1,
            )
            trade_id = self._db.insert_trade(trade)

            # Update the portfolio
            self._portfolio.open_position(
                symbol=symbol,
                side=direction,
                quantity=quantity,
                fill_price=fill_price,
                market=market,
                trade_id=trade_id,
            )

            result: dict[str, Any] = {
                "id": order_id,
                "trade_id": trade_id,
                "symbol": symbol,
                "qty": str(quantity),
                "side": side.lower(),
                "direction": direction,
                "type": "market",
                "status": "filled",
                "fill_price": fill_price,
                "market_price": market_price,
                "slippage": fill_price - market_price,
                "submitted_at": now_iso,
                "filled_at": now_iso,
                "market": market,
                "strategy": strategy,
            }
            logger.info(
                "paper_market_order_filled | symbol={symbol} side={side} quantity={quantity} market_price={market_price} market={market} strategy={strategy} order_id={order_id} fill_price={fill_price} trade_id={trade_id}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                market_price=market_price,
                market=market,
                strategy=strategy,
                order_id=order_id,
                fill_price=fill_price,
                trade_id=trade_id,
            )
            return result

        except ValueError:
            logger.warning(
                "paper_order_insufficient_funds | symbol={symbol} side={side} quantity={quantity} market_price={market_price} market={market} strategy={strategy}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                market_price=market_price,
                market=market,
                strategy=strategy,
            )
            return None
        except Exception:
            logger.exception(
                "paper_market_order_error | symbol={symbol} side={side} quantity={quantity} market_price={market_price} market={market} strategy={strategy}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                market_price=market_price,
                market=market,
                strategy=strategy,
            )
            return None

    # ----- close position --------------------------------------------

    def close_position(
        self,
        symbol: str,
        market_price: float,
        market: str = "stock",
    ) -> dict[str, Any] | None:
        """Close an existing position at market price with slippage.

        Parameters
        ----------
        symbol:
            Instrument symbol to close.
        market_price:
            Current market price before slippage.
        market:
            ``"stock"`` or ``"crypto"``.

        Returns
        -------
        dict | None
            Closing order details including realized PnL, or ``None``
            on failure.
        """
        try:
            pos_dict = self._portfolio.get_position(symbol)
            if pos_dict is None:
                logger.warning(
                    "paper_close_no_position | symbol={symbol} market_price={market_price}",
                    symbol=symbol,
                    market_price=market_price,
                )
                return None

            # Determine close side (opposite of position direction)
            close_side = "sell" if pos_dict["side"] == "long" else "buy"
            fill_price = self._apply_slippage(market_price, close_side)

            # Close in portfolio
            close_result = self._portfolio.close_position(symbol, fill_price)

            # Determine outcome
            pnl = close_result["pnl"]
            pnl_pct = close_result["pnl_pct"]
            if pnl > 0:
                outcome = "win"
            elif pnl < 0:
                outcome = "loss"
            else:
                outcome = "breakeven"

            # Update journal
            trade_id = close_result.get("trade_id")
            if trade_id is not None:
                self._db.update_trade_exit(
                    trade_id=trade_id,
                    exit_price=fill_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    outcome=outcome,
                )

            order_id = self._generate_order_id()
            now_iso = datetime.now(timezone.utc).isoformat()

            result: dict[str, Any] = {
                "id": order_id,
                "trade_id": trade_id,
                "symbol": symbol,
                "qty": str(close_result["quantity"]),
                "side": close_side,
                "type": "market",
                "status": "filled",
                "fill_price": fill_price,
                "market_price": market_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "outcome": outcome,
                "submitted_at": now_iso,
                "filled_at": now_iso,
            }
            logger.info(
                "paper_position_closed | symbol={symbol} market_price={market_price} order_id={order_id} fill_price={fill_price} pnl={pnl} outcome={outcome}",
                symbol=symbol,
                market_price=market_price,
                order_id=order_id,
                fill_price=fill_price,
                pnl=pnl,
                outcome=outcome,
            )
            return result

        except Exception:
            logger.exception(
                "paper_close_position_error | symbol={symbol} market_price={market_price}",
                symbol=symbol,
                market_price=market_price,
            )
            return None

    # ----- price updates ---------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all held positions.

        Parameters
        ----------
        prices:
            Mapping of symbol to current market price.
        """
        self._portfolio.update_prices(prices)
        logger.debug("paper_prices_updated | count={count}", count=len(prices))

    # ----- account & position queries --------------------------------

    def get_account(self) -> dict[str, Any]:
        """Get account summary matching ``AlpacaExecutor.get_account()`` format.

        Returns
        -------
        dict
            Dictionary with ``equity``, ``buying_power``, ``cash``,
            ``portfolio_value``, and ``currency``.
        """
        equity = self._portfolio.equity
        result: dict[str, Any] = {
            "equity": equity,
            "buying_power": self._portfolio.buying_power,
            "cash": self._portfolio.cash,
            "portfolio_value": equity,
            "currency": "USD",
            "initial_capital": self._portfolio.initial_capital,
            "unrealized_pnl": self._portfolio.unrealized_pnl,
            "realized_pnl": self._portfolio.realized_pnl,
        }
        logger.info(
            "paper_account_fetched | equity={equity} buying_power={buying_power}",
            equity=result["equity"],
            buying_power=result["buying_power"],
        )
        return result

    def get_positions(self) -> list[dict[str, Any]]:
        """Get open positions matching the ``AlpacaExecutor`` interface.

        Returns
        -------
        list[dict]
            List of position dicts, each containing ``symbol``, ``qty``,
            ``side``, ``market_value``, ``avg_entry_price``,
            ``unrealized_pl``, and ``unrealized_plpc``.
        """
        raw_positions = self._portfolio.get_positions()
        result: list[dict[str, Any]] = []
        for pos in raw_positions:
            result.append({
                "symbol": pos["symbol"],
                "qty": pos["quantity"],
                "side": pos["side"],
                "market_value": pos["market_value"],
                "avg_entry_price": pos["entry_price"],
                "unrealized_pl": pos["unrealized_pnl"],
                "unrealized_plpc": pos["unrealized_pnl_pct"],
            })
        logger.info("paper_positions_fetched | count={count}", count=len(result))
        return result

    def get_daily_pnl(self) -> float:
        """Calculate today's realized + unrealized PnL.

        Uses the day-start equity snapshot (set via
        :meth:`PaperPortfolio.snapshot_day_start`) as the baseline.

        Returns
        -------
        float
            Combined realized and unrealized PnL since day start.
        """
        current_equity = self._portfolio.equity
        daily_pnl = current_equity - self._portfolio.day_start_equity
        logger.debug(
            "paper_daily_pnl | daily_pnl={daily_pnl} current_equity={current_equity} day_start_equity={day_start_equity}",
            daily_pnl=daily_pnl,
            current_equity=current_equity,
            day_start_equity=self._portfolio.day_start_equity,
        )
        return daily_pnl
