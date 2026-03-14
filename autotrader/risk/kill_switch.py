"""Daily drawdown kill-switch module.

Monitors realised daily profit-and-loss and halts all trading activity
when the loss exceeds a configurable threshold.
"""

from loguru import logger


class KillSwitch:
    """Circuit-breaker that disables trading when the daily loss limit is hit.

    Once triggered the switch stays in the *halted* state until
    :meth:`reset` is called (typically at the start of a new trading
    day).

    Parameters
    ----------
    config : dict
        Must contain:

        - ``daily_loss_limit_pct`` (float): Maximum tolerable daily loss
          expressed as a fraction of account value (e.g. ``0.03`` for
          3 %).
    """

    def __init__(self, config: dict) -> None:
        self._daily_loss_limit_pct: float = config["daily_loss_limit_pct"]
        self._halted: bool = False

        logger.info(
            "kill_switch_initialised | daily_loss_limit_pct={daily_loss_limit_pct}",
            daily_loss_limit_pct=self._daily_loss_limit_pct,
        )

    @property
    def halted(self) -> bool:
        """Return ``True`` if the kill switch is currently engaged."""
        return self._halted

    def is_trading_allowed(
        self, daily_pnl: float, account_value: float
    ) -> bool:
        """Determine whether trading should continue.

        Parameters
        ----------
        daily_pnl : float
            Cumulative profit or loss for the current trading day.  A
            negative value indicates a loss.
        account_value : float
            Current total account equity.

        Returns
        -------
        bool
            ``True`` if trading may continue, ``False`` if the daily
            loss limit has been breached (or was breached earlier today).
        """
        # Once halted, remain halted until explicitly reset
        if self._halted:
            logger.debug("kill_switch_already_halted")
            return False

        # Profitable or break-even day -- always allow
        if daily_pnl >= 0:
            return True

        # Evaluate the loss as a fraction of account value
        loss_pct: float = abs(daily_pnl) / account_value

        if loss_pct >= self._daily_loss_limit_pct:
            self._halted = True
            logger.warning(
                "kill_switch_triggered | daily_pnl={daily_pnl} account_value={account_value} loss_pct={loss_pct} daily_loss_limit_pct={daily_loss_limit_pct}",
                daily_pnl=daily_pnl,
                account_value=account_value,
                loss_pct=round(loss_pct, 6),
                daily_loss_limit_pct=self._daily_loss_limit_pct,
            )
            return False

        logger.debug(
            "kill_switch_within_limits | daily_pnl={daily_pnl} loss_pct={loss_pct} daily_loss_limit_pct={daily_loss_limit_pct}",
            daily_pnl=daily_pnl,
            loss_pct=round(loss_pct, 6),
            daily_loss_limit_pct=self._daily_loss_limit_pct,
        )
        return True

    def reset(self) -> None:
        """Reset the kill switch for a new trading day.

        This clears the *halted* flag so that trading can resume.
        Typically called at the start of each session or trading day.
        """
        was_halted: bool = self._halted
        self._halted = False

        logger.info("kill_switch_reset | was_halted={was_halted}", was_halted=was_halted)
