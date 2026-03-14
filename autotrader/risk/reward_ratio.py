"""Risk-to-reward ratio gate-check module.

Provides a pre-trade filter that rejects setups whose potential reward
does not justify the risk being taken.
"""

from loguru import logger


class RewardRatioGate:
    """Gate that approves or rejects trades based on their risk:reward ratio.

    A trade is approved only when::

        reward / risk >= min_ratio

    where *risk* is the distance from entry to stop-loss and *reward*
    is the distance from entry to take-profit.

    Parameters
    ----------
    min_ratio : float, optional
        Minimum acceptable reward-to-risk ratio.  Defaults to ``2.0``
        (i.e. the potential reward must be at least twice the risk).
    """

    def __init__(self, min_ratio: float = 2.0) -> None:
        self._min_ratio: float = min_ratio

        logger.info(
            "reward_ratio_gate_initialised | min_ratio={min_ratio}",
            min_ratio=self._min_ratio,
        )

    def check(
        self, entry_price: float, stop_loss: float, take_profit: float
    ) -> bool:
        """Verify that a proposed trade meets the minimum risk:reward ratio.

        The method works for both long and short trades because it uses
        absolute distances.

        Parameters
        ----------
        entry_price : float
            Planned entry price.
        stop_loss : float
            Planned stop-loss price.
        take_profit : float
            Planned take-profit price.

        Returns
        -------
        bool
            ``True`` if the trade's reward-to-risk ratio meets or
            exceeds ``min_ratio``, ``False`` otherwise.  Also returns
            ``False`` when *risk* is zero (stop-loss equals entry
            price), since the ratio is undefined.
        """
        risk: float = abs(entry_price - stop_loss)
        reward: float = abs(take_profit - entry_price)

        if risk == 0:
            logger.warning(
                "reward_ratio_zero_risk | entry_price={entry_price} stop_loss={stop_loss} take_profit={take_profit}",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            return False

        ratio: float = reward / risk
        approved: bool = ratio >= self._min_ratio

        logger.info(
            "reward_ratio_checked | entry_price={entry_price} stop_loss={stop_loss} take_profit={take_profit} risk={risk} reward={reward} ratio={ratio} min_ratio={min_ratio} approved={approved}",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk=round(risk, 6),
            reward=round(reward, 6),
            ratio=round(ratio, 4),
            min_ratio=self._min_ratio,
            approved=approved,
        )

        return approved
