"""Pre-flight order validation checks.

Enforces risk and operational constraints before any order is submitted
to a broker or exchange, acting as the last line of defence.
"""

from __future__ import annotations

from loguru import logger

_VALID_DIRECTIONS: frozenset[str] = frozenset({"buy", "sell"})


class OrderValidator:
    """Pre-flight checks before placing any order.

    Every call to :meth:`validate` runs the full checklist and returns
    a pass/fail verdict with a human-readable reason when the order is
    rejected.

    Parameters
    ----------
    config:
        Dictionary that must contain:

        - ``max_positions`` (int): Maximum number of simultaneous open
          positions allowed.
        - ``max_daily_trades`` (int): Maximum number of round-trip
          trades allowed per trading day.
    """

    def __init__(self, config: dict) -> None:
        self._max_positions: int = int(config["max_positions"])
        self._max_daily_trades: int = int(config["max_daily_trades"])

        logger.info(
            "order_validator_initialised | max_positions={max_positions} max_daily_trades={max_daily_trades}",
            max_positions=self._max_positions,
            max_daily_trades=self._max_daily_trades,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(
        self,
        symbol: str,
        direction: str,
        quantity: int | float,
        current_positions: int,
        daily_trade_count: int,
        kill_switch_active: bool,
        open_symbols: set[str] | None = None,
    ) -> tuple[bool, str]:
        """Run all pre-flight checks on an intended order.

        Parameters
        ----------
        symbol:
            Instrument to trade (e.g. ``"AAPL"`` or ``"BTC/USDT"``).
        direction:
            ``"buy"`` or ``"sell"``.
        quantity:
            Number of shares or units to trade.
        current_positions:
            How many positions are currently open.
        daily_trade_count:
            How many trades have already been executed today.
        kill_switch_active:
            Whether the daily-loss kill switch is engaged.
        open_symbols:
            Set of symbols that already have open positions.
            When provided, duplicate entries for the same symbol
            are rejected.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` when the order passes all checks.
            ``(False, reason)`` when a check fails, where *reason* is a
            short human-readable explanation of the rejection.
        """
        # 1. Kill switch
        if kill_switch_active:
            reason = "Kill switch is active -- all trading halted"
            logger.warning(
                "order_rejected_kill_switch | symbol={symbol} direction={direction} quantity={quantity} current_positions={current_positions} daily_trade_count={daily_trade_count} kill_switch_active={kill_switch_active}",
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                current_positions=current_positions,
                daily_trade_count=daily_trade_count,
                kill_switch_active=kill_switch_active,
            )
            return False, reason

        # 2. Valid direction
        if direction.lower() not in _VALID_DIRECTIONS:
            reason = (
                f"Invalid direction '{direction}'; "
                f"must be one of {sorted(_VALID_DIRECTIONS)}"
            )
            logger.warning(
                "order_rejected_invalid_direction | symbol={symbol} direction={direction} quantity={quantity} current_positions={current_positions} daily_trade_count={daily_trade_count} kill_switch_active={kill_switch_active}",
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                current_positions=current_positions,
                daily_trade_count=daily_trade_count,
                kill_switch_active=kill_switch_active,
            )
            return False, reason

        # 3. Positive quantity
        if quantity <= 0:
            reason = f"Quantity must be > 0, got {quantity}"
            logger.warning(
                "order_rejected_non_positive_quantity | symbol={symbol} direction={direction} quantity={quantity} current_positions={current_positions} daily_trade_count={daily_trade_count} kill_switch_active={kill_switch_active}",
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                current_positions=current_positions,
                daily_trade_count=daily_trade_count,
                kill_switch_active=kill_switch_active,
            )
            return False, reason

        # 4. Max positions (applies to all new entries)
        if current_positions >= self._max_positions:
            reason = (
                f"Maximum positions reached ({current_positions}/"
                f"{self._max_positions})"
            )
            logger.warning(
                "order_rejected_max_positions | symbol={symbol} direction={direction} quantity={quantity} current_positions={current_positions} daily_trade_count={daily_trade_count} kill_switch_active={kill_switch_active}",
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                current_positions=current_positions,
                daily_trade_count=daily_trade_count,
                kill_switch_active=kill_switch_active,
            )
            return False, reason

        # 5. Duplicate symbol (don't open a second position for same symbol)
        if open_symbols and symbol in open_symbols:
            reason = f"Duplicate position -- {symbol} already has an open trade"
            logger.warning(
                "order_rejected_duplicate_position | symbol={symbol}",
                symbol=symbol,
            )
            return False, reason

        # 5b. Duplicate base asset (e.g. reject ETH/USDC when ETH/USDT is open)
        if open_symbols:
            base = symbol.split("/")[0] if "/" in symbol else None
            if base:
                for os_sym in open_symbols:
                    os_base = os_sym.split("/")[0] if "/" in os_sym else None
                    if os_base and os_base == base:
                        reason = f"Duplicate base asset -- {os_sym} already open for {base}"
                        logger.warning(
                            "order_rejected_duplicate_base | symbol={symbol} existing={existing}",
                            symbol=symbol,
                            existing=os_sym,
                        )
                        return False, reason

        # 6. Max daily trades
        if daily_trade_count >= self._max_daily_trades:
            reason = (
                f"Maximum daily trades reached ({daily_trade_count}/"
                f"{self._max_daily_trades})"
            )
            logger.warning(
                "order_rejected_max_daily_trades | symbol={symbol} direction={direction} quantity={quantity} current_positions={current_positions} daily_trade_count={daily_trade_count} kill_switch_active={kill_switch_active}",
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                current_positions=current_positions,
                daily_trade_count=daily_trade_count,
                kill_switch_active=kill_switch_active,
            )
            return False, reason

        logger.info(
            "order_validated | symbol={symbol} direction={direction} quantity={quantity} current_positions={current_positions} daily_trade_count={daily_trade_count} kill_switch_active={kill_switch_active}",
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            current_positions=current_positions,
            daily_trade_count=daily_trade_count,
            kill_switch_active=kill_switch_active,
        )
        return True, ""
