"""Funding Rate Fade signal strategy.

Generates contrarian signals based on extreme perpetual futures funding rates.
When funding is very positive (longs paying shorts), the crowd is overleveraged
long — go SHORT.  When very negative, shorts are crowded — go LONG.
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class FundingRateSignal(BaseSignal):
    """Contrarian funding-rate fade signal (crypto only)."""

    @property
    def name(self) -> str:
        return "funding_rate"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic entry-point — real logic is in evaluate_from_funding."""
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_from_funding(
        self,
        symbol: str,
        funding_rate: float,
        current_price: float,
        atr: float = 0.0,
    ) -> SignalResult:
        """Evaluate a funding rate fade signal.

        Parameters
        ----------
        symbol:
            Trading pair (e.g. ``"BTC/USDT"``).
        funding_rate:
            Current 8-hour funding rate as a decimal (e.g. 0.0003 = 0.03%).
        current_price:
            Latest price.
        """
        if not self.is_enabled():
            return SignalResult(triggered=False, strategy_name=self.name)

        threshold: float = self.config.get("threshold", 0.0001)
        extreme: float = self.config.get("extreme_threshold", 0.0005)
        stop_pct: float = self.config.get("stop_loss_pct", 0.015)
        atr_stop_mult: float = self.config.get("atr_stop_multiplier", 1.5)

        abs_rate = abs(funding_rate)
        if abs_rate < threshold:
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason=f"Funding rate {funding_rate:.6f} within threshold",
            )

        # Positive funding => longs paying => crowd is long => fade SHORT
        # Negative funding => shorts paying => crowd is short => fade LONG
        direction = SignalDirection.SHORT if funding_rate > 0 else SignalDirection.LONG

        if not self.check_swing_bias(symbol, direction):
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="Blocked by swing bias",
            )

        # Confidence: scale from 0.50 at threshold to 0.70 at extreme+
        if abs_rate >= extreme:
            confidence = 0.70
        else:
            confidence = 0.50 + 0.20 * (abs_rate - threshold) / max(extreme - threshold, 1e-9)

        entry_price = current_price
        stop_distance = atr * atr_stop_mult if atr > 0 else entry_price * stop_pct
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * 2.0
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * 2.0

        reason = (
            f"Funding rate {funding_rate:.6f} "
            f"({'extreme' if abs_rate >= extreme else 'elevated'}) "
            f"=> fade {'longs' if funding_rate > 0 else 'shorts'}"
        )

        logger.info(
            "signal_triggered | signal={} symbol={} direction={} rate={:.6f} confidence={:.2f}",
            self.name, symbol, direction.value, funding_rate, confidence,
        )

        return SignalResult(
            triggered=True,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            strategy_name=self.name,
            reason=reason,
        )
