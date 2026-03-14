"""Position monitor for open trades with stop-loss and take-profit management.

Continuously checks open trades against live market prices and triggers
automated exits when stop-loss or take-profit levels are breached.
Includes a trailing-stop mechanism that tightens the stop to breakeven
once the trade is more than 50 % of the way to its profit target.
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
from journal.db import TradeDatabase


class PositionMonitor:
    """Monitors open trades and triggers exits at stop-loss or take-profit levels.

    Designed to be called on a periodic schedule (e.g. every few seconds or
    minutes).  Each invocation of :meth:`check_open_positions` queries the
    database for open trades, fetches current prices, and determines
    whether any exit condition has been met.

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
    ) -> None:
        """Initialize with all required dependencies."""
        self._db: TradeDatabase = db
        self._stock_feed: StockFeed = stock_feed
        self._crypto_feed: CryptoFeed = crypto_feed
        self._stock_executor: AlpacaExecutor = stock_executor
        self._crypto_executor: CryptoExecutor = crypto_executor
        self._telegram: TelegramAlert = telegram
        self._paper_trade: bool = paper_trade

        # Track trade IDs whose stops have already been tightened to
        # breakeven so the adjustment only happens once per trade.
        self._trailing_stop_adjusted: set[int] = set()

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

            # 2. Trailing stop: tighten to breakeven when > 50 % to target
            stop_loss = self._apply_trailing_stop(
                trade_id=trade_id,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=current_price,
            )

            # 3. Check stop-loss
            sl_hit: bool = False
            if direction == "long" and current_price <= stop_loss:
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
            if direction == "long" and current_price >= take_profit:
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
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        current_price: float,
    ) -> float:
        """Tighten stop to breakeven when the trade is > 50 % to target.

        The adjustment is applied at most once per trade (tracked via
        ``_trailing_stop_adjusted``).

        Parameters
        ----------
        trade_id:
            Database ID of the trade.
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
            The (possibly adjusted) stop-loss value.
        """
        if trade_id in self._trailing_stop_adjusted:
            # Already tightened -- use breakeven (entry_price) as stop.
            return entry_price

        if direction == "long":
            target_range: float = take_profit - entry_price
            if target_range <= 0:
                return stop_loss
            progress: float = (current_price - entry_price) / target_range
        else:  # short
            target_range = entry_price - take_profit
            if target_range <= 0:
                return stop_loss
            progress = (entry_price - current_price) / target_range

        if progress > 0.5:
            self._trailing_stop_adjusted.add(trade_id)
            logger.info(
                "trailing_stop_tightened | trade_id={trade_id} old_stop={old_stop} new_stop={new_stop} progress_pct={progress_pct}",
                trade_id=trade_id,
                old_stop=stop_loss,
                new_stop=entry_price,
                progress_pct=round(progress * 100, 1),
            )
            return entry_price

        return stop_loss

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
        self._trailing_stop_adjusted.discard(trade_id)

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
        self._telegram.send_trade_alert(alert_dict)
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
