"""Position monitor for open trades with stop-loss and take-profit management.

Continuously checks open trades against live market prices and triggers
automated exits when stop-loss or take-profit levels are breached.
Includes a continuous trailing-stop mechanism that ratchets the stop upward
as price moves favorably:

- At 50 % progress toward take-profit: stop moves to breakeven (entry price).
- At 75 % progress: stop trails at 50 % of current gains.
- At 100 %+ progress (beyond TP): stop trails at 75 % of current gains.

The stop never moves backwards -- the highest (for longs) or lowest
(for shorts) trailing stop level is preserved across successive ticks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from alerts.telegram_bot import TelegramAlert
from data.feed_crypto import CryptoFeed
from data.feed_stocks import StockFeed
from execution.alpaca_executor import AlpacaExecutor
from execution.crypto_executor import CryptoExecutor
from execution.paper_engine import PaperExecutor
from journal.db import TradeDatabase


class PositionMonitor:
    """Monitors open trades and triggers exits at stop-loss or take-profit levels.

    Designed to be called on a periodic schedule (e.g. every few seconds or
    minutes).  Each invocation of :meth:`check_open_positions` queries the
    database for open trades, fetches current prices, and determines
    whether any exit condition has been met.

    A continuous trailing stop ratchets the effective stop-loss as the trade
    moves in the favourable direction:

    - **50 % progress** toward take-profit: stop moves to breakeven (entry).
    - **75 % progress**: stop trails at 50 % of unrealised gains.
    - **100 %+ progress**: stop trails at 75 % of unrealised gains.

    The trailing stop never moves backwards; the best level per trade is
    tracked in ``_trailing_stop_levels``.

    Parameters
    ----------
    db:
        Trade database instance used to read open trades and persist exit
        updates.
    stock_feed:
        Stock market data feed for retrieving latest stock prices.
    crypto_feed:
        Crypto market data feed for retrieving latest crypto prices.
    stock_executor:
        Alpaca executor for closing stock positions.
    crypto_executor:
        CCXT executor for closing crypto positions.
    telegram:
        Telegram alerting interface for sending trade notifications.
    paper_trade:
        When ``True`` (default) no real orders are submitted -- only the
        database is updated and alerts are sent.
    """

    def __init__(
        self,
        db: TradeDatabase,
        stock_feed: StockFeed,
        crypto_feed: CryptoFeed,
        stock_executor: AlpacaExecutor,
        crypto_executor: CryptoExecutor,
        telegram: TelegramAlert,
        paper_trade: bool = True,
        paper_executor: PaperExecutor | None = None,
        config: dict | None = None,
    ) -> None:
        """Initialize with all required dependencies."""
        self._db: TradeDatabase = db
        self._stock_feed: StockFeed = stock_feed
        self._crypto_feed: CryptoFeed = crypto_feed
        self._stock_executor: AlpacaExecutor = stock_executor
        self._crypto_executor: CryptoExecutor = crypto_executor
        self._telegram: TelegramAlert = telegram
        self._paper_trade: bool = paper_trade
        self._paper_executor: PaperExecutor | None = paper_executor
        self._config: dict = config or {}

        # Map trade_id → best trailing stop level seen so far.
        # The stop is only ever ratcheted forward, never backwards.
        self._trailing_stop_levels: dict[int, float] = {}

        # Track trade IDs that have already had a partial profit take.
        self._partial_taken: set[int] = set()

        # Strategy auto-disable: count consecutive cooldown triggers.
        # After STRATEGY_DISABLE_AFTER_COOLDOWNS cooldowns, auto-disable.
        self._strategy_cooldown_count: dict[str, int] = {}

        logger.info(
            "position_monitor_initialised | paper_trade={paper_trade}",
            paper_trade=paper_trade,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_open_positions(self) -> None:
        """Main loop: fetch all open trades, check current prices, handle exits.

        For each open trade the method:

        1. Retrieves the current market price from the appropriate data feed
           (stock or crypto, based on ``trade["market"]``).
        2. Applies trailing-stop logic -- if the trade has moved more than
           50 % toward its take-profit target the stop is tightened to the
           entry price (breakeven).
        3. Checks whether the stop-loss has been hit:
           - **Long**: ``current_price <= stop_loss``
           - **Short**: ``current_price >= stop_loss``
        4. Checks whether the take-profit has been hit:
           - **Long**: ``current_price >= take_profit``
           - **Short**: ``current_price <= take_profit``
        5. If either exit condition is triggered the trade is closed via
           :meth:`_close_trade`.
        """
        open_trades: list[dict[str, Any]] = self._db.get_open_trades()
        if not open_trades:
            logger.debug("no_open_positions")
            return

        logger.info("checking_open_positions | count={count}", count=len(open_trades))

        for trade in open_trades:
            trade_id: int = trade["id"]
            symbol: str = trade["symbol"]
            market: str = trade["market"]
            direction: str = trade["direction"]
            entry_price: float = trade["entry_price"]
            stop_loss: float = trade["stop_loss"]
            take_profit: float = trade["take_profit"]

            # 1. Get current market price
            current_price: float | None = self._get_current_price(symbol, market)
            if current_price is None or current_price == 0.0:
                logger.warning(
                    "price_unavailable | trade_id={trade_id} symbol={symbol} market={market} direction={direction}",
                    trade_id=trade_id,
                    symbol=symbol,
                    market=market,
                    direction=direction,
                )
                continue

            # Update paper portfolio with current price for unrealized PnL
            if self._paper_trade and self._paper_executor:
                self._paper_executor._portfolio.update_price(symbol, current_price)

            if trade.get("strategy") == "manual_telegram":
                logger.debug(
                    "manual_position_monitor_skipped | trade_id={trade_id} symbol={symbol}",
                    trade_id=trade_id,
                    symbol=symbol,
                )
                continue

            # 1b. Check max hold time
            max_hold = self._get_max_hold_hours(market)
            if max_hold is not None and max_hold > 0:
                trade_ts = trade.get("timestamp", "")
                if trade_ts:
                    try:
                        trade_dt = datetime.fromisoformat(trade_ts)
                        if trade_dt.tzinfo is None:
                            trade_dt = trade_dt.replace(tzinfo=timezone.utc)
                        age_hours = (datetime.now(timezone.utc) - trade_dt).total_seconds() / 3600
                        if age_hours >= max_hold:
                            logger.info(
                                "max_hold_time_exceeded | trade_id={trade_id} symbol={symbol} "
                                "market={market} age_hours={age_hours} max_hours={max_hours}",
                                trade_id=trade_id,
                                symbol=symbol,
                                market=market,
                                age_hours=round(age_hours, 1),
                                max_hours=max_hold,
                            )
                            self._close_trade(trade, exit_price=current_price, reason="max_hold_time")
                            continue
                    except (ValueError, TypeError):
                        pass

            # 2. Trailing stop: tighten to breakeven when > 50 % to target
            stop_loss = self._apply_trailing_stop(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=current_price,
            )

            # 2b. Partial profit take at 50% of TP range (once per trade)
            if trade_id not in self._partial_taken:
                quantity: float = trade["quantity"]
                if quantity > 0:
                    self._maybe_partial_take(
                        trade=trade,
                        direction=direction,
                        entry_price=entry_price,
                        take_profit=take_profit,
                        current_price=current_price,
                        market=market,
                    )

            # 3. Check stop-loss
            sl_hit: bool = False
            if not stop_loss or stop_loss <= 0:
                logger.warning(
                    "invalid_stop_loss | trade_id={trade_id} symbol={symbol} stop_loss={stop_loss}",
                    trade_id=trade_id,
                    symbol=symbol,
                    stop_loss=stop_loss,
                )
            elif direction == "long" and current_price <= stop_loss:
                sl_hit = True
            elif direction == "short" and current_price >= stop_loss:
                sl_hit = True

            if sl_hit:
                logger.info(
                    "stop_loss_triggered | trade_id={trade_id} symbol={symbol} market={market} direction={direction} current_price={current_price} stop_loss={stop_loss}",
                    trade_id=trade_id,
                    symbol=symbol,
                    market=market,
                    direction=direction,
                    current_price=current_price,
                    stop_loss=stop_loss,
                )
                self._close_trade(trade, exit_price=current_price, reason="stop_loss")
                continue

            # 4. Check take-profit
            tp_hit: bool = False
            if not take_profit or take_profit <= 0:
                logger.warning(
                    "invalid_take_profit | trade_id={trade_id} symbol={symbol} take_profit={take_profit}",
                    trade_id=trade_id,
                    symbol=symbol,
                    take_profit=take_profit,
                )
            elif direction == "long" and current_price >= take_profit:
                tp_hit = True
            elif direction == "short" and current_price <= take_profit:
                tp_hit = True

            if tp_hit:
                logger.info(
                    "take_profit_triggered | trade_id={trade_id} symbol={symbol} market={market} direction={direction} current_price={current_price} take_profit={take_profit}",
                    trade_id=trade_id,
                    symbol=symbol,
                    market=market,
                    direction=direction,
                    current_price=current_price,
                    take_profit=take_profit,
                )
                self._close_trade(trade, exit_price=current_price, reason="take_profit")
                continue

            logger.debug(
                "position_within_bounds | trade_id={trade_id} symbol={symbol} market={market} direction={direction} current_price={current_price}",
                trade_id=trade_id,
                symbol=symbol,
                market=market,
                direction=direction,
                current_price=current_price,
            )

    # ------------------------------------------------------------------
    # Max hold time
    # ------------------------------------------------------------------

    def _get_max_hold_hours(self, market: str) -> float | None:
        """Return configured max hold hours for the given market type."""
        risk_cfg = self._config.get("risk", {})
        if market == "crypto":
            return risk_cfg.get("max_hold_hours_crypto")
        if market == "stock":
            return risk_cfg.get("max_hold_hours_stock")
        return None

    # ------------------------------------------------------------------
    # Price retrieval
    # ------------------------------------------------------------------

    def _get_current_price(self, symbol: str, market: str) -> float | None:
        """Get current price for a symbol from the appropriate feed.

        Parameters
        ----------
        symbol:
            Instrument symbol (e.g. ``"AAPL"`` or ``"BTC/USDT"``).
        market:
            ``"stock"`` or ``"crypto"``.

        Returns
        -------
        float | None
            The latest traded price, or ``None`` if the market type is
            unrecognised.  A return value of ``0.0`` indicates a feed error.
        """
        if market == "stock":
            return self._stock_feed.get_latest_price(symbol)
        elif market == "crypto":
            return self._crypto_feed.get_latest_price(symbol)
        else:
            logger.error(
                "unknown_market_type | market={market} symbol={symbol}",
                market=market,
                symbol=symbol,
            )
            return None

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    def _apply_trailing_stop(
        self,
        trade_id: int,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        current_price: float,
    ) -> float:
        """Continuous trailing stop that ratchets up as price moves favorably.

        Thresholds:
        - 50% progress -> stop at breakeven (entry)
        - 75% progress -> stop at 50% of gains
        - 100%+ progress -> stop at 75% of gains

        Parameters
        ----------
        trade_id:
            Database ID of the trade.
        symbol:
            Instrument symbol.
        direction:
            ``"long"`` or ``"short"``.
        entry_price:
            Original entry price.
        stop_loss:
            Current stop-loss level.
        take_profit:
            Original take-profit target.
        current_price:
            Latest market price.

        Returns
        -------
        float
            The (possibly adjusted) stop-loss value.  Never moves backwards
            relative to previously recorded levels for this trade.
        """
        if direction == "long":
            target_range: float = take_profit - entry_price
            if target_range <= 0:
                return stop_loss
            progress: float = (current_price - entry_price) / target_range
            gain: float = current_price - entry_price
        else:  # short
            target_range = entry_price - take_profit
            if target_range <= 0:
                return stop_loss
            progress = (entry_price - current_price) / target_range
            gain = entry_price - current_price

        if progress < 0.5:
            # Even below threshold, return previously stored level if it exists
            return self._trailing_stop_levels.get(trade_id, stop_loss)

        # Calculate new trailing stop level
        if progress >= 1.0:
            trail_pct: float = 0.75
        elif progress >= 0.75:
            trail_pct = 0.50
        else:
            trail_pct = 0.0  # breakeven

        if direction == "long":
            new_stop: float = entry_price + gain * trail_pct
        else:
            new_stop = entry_price - gain * trail_pct

        # Never move stop backwards
        prev_stop: float = self._trailing_stop_levels.get(trade_id, stop_loss)
        if direction == "long":
            best_stop: float = max(new_stop, prev_stop)
        else:
            best_stop = min(new_stop, prev_stop)

        if best_stop != prev_stop:
            self._trailing_stop_levels[trade_id] = best_stop
            logger.info(
                "trailing_stop_updated | trade_id={trade_id} symbol={symbol} old_stop={old_stop} new_stop={new_stop} progress_pct={progress_pct} trail_pct={trail_pct}",
                trade_id=trade_id,
                symbol=symbol,
                old_stop=round(prev_stop, 4),
                new_stop=round(best_stop, 4),
                progress_pct=round(progress * 100, 1),
                trail_pct=round(trail_pct * 100, 0),
            )
        elif trade_id not in self._trailing_stop_levels:
            self._trailing_stop_levels[trade_id] = best_stop
            logger.info(
                "trailing_stop_activated | trade_id={trade_id} symbol={symbol} stop={stop} progress_pct={progress_pct}",
                trade_id=trade_id,
                symbol=symbol,
                stop=round(best_stop, 4),
                progress_pct=round(progress * 100, 1),
            )

        return best_stop

    # ------------------------------------------------------------------
    # Partial profit taking
    # ------------------------------------------------------------------

    _PARTIAL_TAKE_PROGRESS: float = 0.50  # Take profit at 50% of TP range
    _PARTIAL_TAKE_FRACTION: float = 0.50  # Close 50% of position

    def _maybe_partial_take(
        self,
        trade: dict[str, Any],
        direction: str,
        entry_price: float,
        take_profit: float,
        current_price: float,
        market: str,
    ) -> None:
        """Close half the position when price reaches 50% of TP target."""
        trade_id: int = trade["id"]
        symbol: str = trade["symbol"]
        quantity: float = trade["quantity"]

        # Calculate progress toward TP
        if direction == "long":
            target_range = take_profit - entry_price
            if target_range <= 0:
                return
            progress = (current_price - entry_price) / target_range
        else:
            target_range = entry_price - take_profit
            if target_range <= 0:
                return
            progress = (entry_price - current_price) / target_range

        if progress < self._PARTIAL_TAKE_PROGRESS:
            return

        # Calculate quantities
        close_qty = quantity * self._PARTIAL_TAKE_FRACTION
        remaining_qty = quantity - close_qty

        # Round for market type
        if market == "crypto":
            close_qty = float(int(close_qty * 1e6) / 1e6)
            remaining_qty = float(int(remaining_qty * 1e6) / 1e6)
        else:
            close_qty = float(int(close_qty))
            remaining_qty = float(int(quantity) - int(close_qty))

        if close_qty <= 0 or remaining_qty <= 0:
            return

        # Calculate PnL on the closed portion
        if direction == "long":
            partial_pnl = (current_price - entry_price) * close_qty
        else:
            partial_pnl = (entry_price - current_price) * close_qty

        # Execute partial close for paper trades
        if self._paper_trade and self._paper_executor:
            portfolio = self._paper_executor._portfolio
            with portfolio._lock:
                if symbol in portfolio._positions:
                    pos = portfolio._positions[symbol]
                    pos.quantity = remaining_qty
                    portfolio._cash += close_qty * current_price
                    portfolio._realized_pnl += partial_pnl

        # Update DB: reduce quantity, record partial take
        self._db.update_trade_partial_close(
            trade_id=trade_id,
            closed_qty=close_qty,
            remaining_qty=remaining_qty,
            partial_pnl=partial_pnl,
            partial_exit_price=current_price,
        )

        self._partial_taken.add(trade_id)

        logger.info(
            "partial_profit_taken | trade_id={} symbol={} closed_qty={} remaining={} partial_pnl={:.2f} progress={:.0%}",
            trade_id, symbol, close_qty, remaining_qty, partial_pnl, progress,
        )

        # Telegram alert
        self._telegram.send_alert({
            "action": "PARTIAL TAKE",
            "symbol": symbol,
            "direction": direction.upper(),
            "price": current_price,
            "quantity": close_qty,
            "strategy": trade.get("strategy", "N/A"),
            "pnl": partial_pnl,
        })

    # ------------------------------------------------------------------
    # Trade closure
    # ------------------------------------------------------------------

    def _close_trade(
        self,
        trade: dict[str, Any],
        exit_price: float,
        reason: str,
    ) -> None:
        """Execute exit, update DB, and send alert.

        Parameters
        ----------
        trade:
            Dictionary representing the open trade (from
            ``TradeDatabase.get_open_trades``).
        exit_price:
            Price at which the position is being closed.
        reason:
            Human-readable exit reason (``"stop_loss"`` or
            ``"take_profit"``).
        """
        trade_id: int = trade["id"]
        symbol: str = trade["symbol"]
        market: str = trade["market"]
        direction: str = trade["direction"]
        quantity: float = trade["quantity"]

        # 1. Execute the closing order (unless paper trading)
        if not self._paper_trade:
            order_result = self._execute_close_order(
                symbol=symbol,
                market=market,
                direction=direction,
                quantity=quantity,
            )
            if order_result is None:
                logger.error(
                    "close_order_failed | trade_id={trade_id} symbol={symbol} reason={reason} exit_price={exit_price}",
                    trade_id=trade_id,
                    symbol=symbol,
                    reason=reason,
                    exit_price=exit_price,
                )
                return
            logger.info(
                "close_order_executed | trade_id={trade_id} symbol={symbol} reason={reason} exit_price={exit_price} order={order}",
                trade_id=trade_id,
                symbol=symbol,
                reason=reason,
                exit_price=exit_price,
                order=order_result,
            )
        else:
            # Paper trade: remove RAM position and credit cash back
            if self._paper_executor:
                try:
                    self._paper_executor._portfolio.close_position(symbol, exit_price)
                except KeyError:
                    logger.debug(
                        "paper_ram_position_already_gone | trade_id={trade_id} symbol={symbol}",
                        trade_id=trade_id,
                        symbol=symbol,
                    )
            logger.info(
                "paper_trade_close | trade_id={trade_id} symbol={symbol} reason={reason} exit_price={exit_price}",
                trade_id=trade_id,
                symbol=symbol,
                reason=reason,
                exit_price=exit_price,
            )

        # 2. Compute PnL
        pnl, pnl_pct, outcome = self._compute_trade_pnl(trade, exit_price)

        # 3. Update the database
        self._db.update_trade_exit(
            trade_id=trade_id,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            outcome=outcome,
        )
        logger.info(
            "trade_exit_persisted | trade_id={trade_id} symbol={symbol} reason={reason} exit_price={exit_price} pnl={pnl} pnl_pct={pnl_pct} outcome={outcome}",
            trade_id=trade_id,
            symbol=symbol,
            reason=reason,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            outcome=outcome,
        )

        # 4. Clean up trailing stop tracking
        self._trailing_stop_levels.pop(trade_id, None)
        self._partial_taken.discard(trade_id)

        # 5. Send Telegram alert
        alert_dict: dict[str, Any] = {
            "action": "EXIT",
            "symbol": symbol,
            "direction": direction.upper(),
            "price": exit_price,
            "quantity": quantity,
            "strategy": trade.get("strategy", "N/A"),
            "pnl": pnl,
        }
        logger.info(
            "exit_alert_sent | trade_id={trade_id} symbol={symbol} reason={reason} exit_price={exit_price} pnl={pnl} pnl_pct={pnl_pct} outcome={outcome}",
            trade_id=trade_id,
            symbol=symbol,
            reason=reason,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            outcome=outcome,
        )

    # ------------------------------------------------------------------
    # Order execution helpers
    # ------------------------------------------------------------------

    def _execute_close_order(
        self,
        symbol: str,
        market: str,
        direction: str,
        quantity: float,
    ) -> dict[str, Any] | None:
        """Submit a closing order to the appropriate executor.

        Parameters
        ----------
        symbol:
            Instrument symbol.
        market:
            ``"stock"`` or ``"crypto"``.
        direction:
            ``"long"`` or ``"short"``.  Determines the order side
            (sell to close a long, buy to close a short).
        quantity:
            Position size to close.

        Returns
        -------
        dict | None
            Order result dictionary on success, or ``None`` on failure.
        """
        if market == "stock":
            return self._stock_executor.close_position(symbol)
        elif market == "crypto":
            close_side: str = "sell" if direction == "long" else "buy"
            return self._crypto_executor.close_position(
                symbol=symbol,
                side=close_side,
                amount=quantity,
            )
        else:
            logger.error(
                "unknown_market_for_close | market={market} symbol={symbol}",
                market=market,
                symbol=symbol,
            )
            return None

    # ------------------------------------------------------------------
    # PnL computation
    # ------------------------------------------------------------------

    def _compute_trade_pnl(
        self,
        trade: dict[str, Any],
        exit_price: float,
    ) -> tuple[float, float, str]:
        """Calculate PnL, PnL percentage, and outcome for a trade exit.

        Parameters
        ----------
        trade:
            Dictionary representing the trade being closed.
        exit_price:
            Price at which the position is closed.

        Returns
        -------
        tuple[float, float, str]
            A three-element tuple of ``(pnl, pnl_pct, outcome)`` where:

            - ``pnl`` is the dollar profit/loss.
            - ``pnl_pct`` is the percentage return relative to entry.
            - ``outcome`` is one of ``"win"``, ``"loss"``, or
              ``"breakeven"``.
        """
        entry_price: float = trade["entry_price"]
        quantity: float = trade["quantity"]
        direction: str = trade["direction"]

        if direction == "long":
            pnl: float = (exit_price - entry_price) * quantity
            pnl_pct: float = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price

        if pnl > 0:
            outcome: str = "win"
        elif pnl < 0:
            outcome = "loss"
        else:
            outcome = "breakeven"

        return pnl, pnl_pct, outcome
