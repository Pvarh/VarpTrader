"""ATR-based position sizing module.

Calculates the number of shares/units to trade based on account risk
tolerance and the Average True Range (ATR) of the instrument.
"""

from loguru import logger


class PositionSizer:
    """Determine position size using ATR-based risk management.

    The sizer ensures that no single trade risks more than a configured
    percentage of the account value.  When ATR data is unavailable it
    falls back to a fixed stop-loss percentage of the entry price.

    Parameters
    ----------
    config : dict
        Must contain the following keys:

        - ``position_size_pct`` (float): Maximum fraction of the account
          value to risk per trade (e.g. 0.02 for 2 %).
        - ``atr_period`` (int): Look-back period used to compute ATR
          (informational; the actual ATR value is passed in at call time).
        - ``atr_multiplier`` (float): Multiplier applied to the ATR to
          derive the per-unit risk distance.
        - ``stop_loss_pct`` (float): Fallback stop-loss expressed as a
          fraction of the entry price, used when ATR is zero or missing.
    """

    def __init__(self, config: dict) -> None:
        self._position_size_pct: float = config["position_size_pct"]
        self._atr_period: int = config["atr_period"]
        self._atr_multiplier: float = config["atr_multiplier"]
        self._stop_loss_pct: float = config["stop_loss_pct"]

        logger.info(
            "position_sizer_initialised | position_size_pct={position_size_pct} atr_period={atr_period} atr_multiplier={atr_multiplier} stop_loss_pct={stop_loss_pct}",
            position_size_pct=self._position_size_pct,
            atr_period=self._atr_period,
            atr_multiplier=self._atr_multiplier,
            stop_loss_pct=self._stop_loss_pct,
        )

    def calculate_size(
        self,
        account_value: float,
        entry_price: float,
        atr: float,
        available_cash: float = 0.0,
        market: str = "stock",
    ) -> float:
        """Calculate the number of shares/units to trade.

        Parameters
        ----------
        account_value : float
            Current total account equity.
        entry_price : float
            Expected entry price for the trade.
        atr : float
            Current Average True Range of the instrument.  Pass ``0``
            (or a negative value) to use the stop-loss-percentage
            fallback.
        available_cash : float
            Available cash/buying power.  When positive the quantity is
            capped so that ``qty * entry_price <= available_cash``.
        market : str
            ``"crypto"`` for fractional sizing, ``"stock"`` for whole
            shares.

        Returns
        -------
        float
            Number of units.  Whole shares for stocks (min 1), fractional
            for crypto (6 decimal places, min 0.000001).
        """
        max_risk_amount: float = account_value * self._position_size_pct

        if atr > 0:
            risk_per_unit: float = atr * self._atr_multiplier
            logger.debug(
                "using_atr_risk | atr={atr} atr_multiplier={atr_multiplier} risk_per_unit={risk_per_unit}",
                atr=atr,
                atr_multiplier=self._atr_multiplier,
                risk_per_unit=risk_per_unit,
            )
        else:
            risk_per_unit = self._stop_loss_pct * entry_price
            logger.debug(
                "using_stop_loss_fallback | stop_loss_pct={stop_loss_pct} entry_price={entry_price} risk_per_unit={risk_per_unit}",
                stop_loss_pct=self._stop_loss_pct,
                entry_price=entry_price,
                risk_per_unit=risk_per_unit,
            )

        position_size: float = max_risk_amount / risk_per_unit

        # Cap so total position value does not exceed 10% of account
        max_position_value: float = account_value * 0.10
        max_qty_by_value: float = max_position_value / entry_price
        if position_size > max_qty_by_value:
            logger.debug(
                "position_capped_by_value | raw_size={raw_size} max_value={max_value} capped_size={capped_size}",
                raw_size=position_size,
                max_value=max_position_value,
                capped_size=max_qty_by_value,
            )
            position_size = max_qty_by_value

        # Cap by available cash so we never size beyond buying power
        if available_cash > 0:
            max_qty_by_cash: float = available_cash / entry_price
            if position_size > max_qty_by_cash:
                logger.info(
                    "position_capped_by_cash | raw_size={raw} cash={cash} max_qty={max_qty}",
                    raw=round(position_size, 6),
                    cash=round(available_cash, 2),
                    max_qty=round(max_qty_by_cash, 6),
                )
                position_size = max_qty_by_cash

        # Final rounding: fractional for crypto, whole shares for stocks
        if market == "crypto":
            final_size = float(int(position_size * 1e6) / 1e6)
            if final_size <= 0:
                final_size = 0.0
        else:
            final_size = float(max(1, int(position_size)))

        logger.info(
            "position_size_calculated | account_value={account_value} entry_price={entry_price} atr={atr} max_risk_amount={max_risk_amount} position_size={position_size} market={market}",
            account_value=account_value,
            entry_price=entry_price,
            atr=atr,
            max_risk_amount=max_risk_amount,
            position_size=final_size,
            market=market,
        )

        return final_size
