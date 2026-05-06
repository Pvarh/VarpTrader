"""Open Interest Divergence signal — crypto only.

Detects divergence between price movement and open interest changes
on perpetual futures markets. Available via CCXT's futures exchange.

Key insight: Open Interest tracks the total number of outstanding
derivative contracts. When price and OI move in opposite directions,
it signals a potential reversal:

- Price up + OI down = longs are closing (weak rally) → SHORT
- Price down + OI down = shorts are closing (weak selloff) → LONG
- Price up + OI up = new longs entering (confirms trend) → no signal
- Price down + OI up = new shorts entering (confirms trend) → no signal

The divergence (price one way, OI the other) is the tradeable signal.
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class OIDivergenceSignal(BaseSignal):
    """Open Interest divergence signal (crypto perpetual futures only)."""

    @property
    def name(self) -> str:
        return "oi_divergence"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate — not used. See evaluate_oi_divergence()."""
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_oi_divergence(
        self,
        symbol: str,
        current_price: float,
        price_change_pct: float,
        oi_change_pct: float,
        atr: float = 0.0,
    ) -> SignalResult:
        """Evaluate OI divergence signal.

        Parameters
        ----------
        symbol:
            Trading pair.
        current_price:
            Latest price.
        price_change_pct:
            Price change over lookback as decimal (0.02 = 2%).
        oi_change_pct:
            Open Interest change over lookback as decimal (0.05 = 5%).
        atr:
            Current ATR for stop calculation.
        """
        no_signal = SignalResult(triggered=False, strategy_name=self.name)

        if not self.is_enabled():
            return no_signal

        min_price_move = self.config.get("price_change_pct", 0.01)
        min_oi_move = self.config.get("oi_change_pct", 0.03)
        sl_pct = self.config.get("stop_loss_pct", 0.015)
        atr_stop_mult = self.config.get("atr_stop_multiplier", 1.5)
        rr_ratio = self.config.get("rr_ratio", 2.0)

        # Need meaningful moves in both to trigger
        if abs(price_change_pct) < min_price_move:
            return no_signal
        if abs(oi_change_pct) < min_oi_move:
            return no_signal

        direction = None
        reason = ""

        # Price up + OI down = weak rally, longs exiting → SHORT
        if price_change_pct > min_price_move and oi_change_pct < -min_oi_move:
            direction = SignalDirection.SHORT
            reason = (
                f"OI divergence bearish: price +{price_change_pct:.1%} "
                f"but OI {oi_change_pct:.1%} — longs exiting"
            )

        # Price down + OI down = weak selloff, shorts covering → LONG
        elif price_change_pct < -min_price_move and oi_change_pct < -min_oi_move:
            direction = SignalDirection.LONG
            reason = (
                f"OI divergence bullish: price {price_change_pct:.1%} "
                f"but OI {oi_change_pct:.1%} — shorts covering"
            )

        if direction is None:
            return no_signal

        # Swing bias check
        if not self.check_swing_bias(symbol, direction):
            return SignalResult(
                triggered=False, strategy_name=self.name,
                reason=f"Swing bias blocks {direction.value}",
            )

        # Price levels
        entry_price = current_price
        stop_distance = atr * atr_stop_mult if atr > 0 else entry_price * sl_pct

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * rr_ratio
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * rr_ratio

        # Scale confidence by OI divergence magnitude
        oi_magnitude = abs(oi_change_pct)
        confidence = min(0.75, 0.55 + oi_magnitude * 2.0)

        logger.info(
            "oi_divergence_triggered | symbol={} dir={} entry={:.2f} sl={:.2f} "
            "tp={:.2f} price_chg={:.2%} oi_chg={:.2%}",
            symbol, direction.value, entry_price, stop_loss, take_profit,
            price_change_pct, oi_change_pct,
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
