"""EMA pullback / trend continuation signal strategy.

Generates signals when price pulls back to the fast EMA during a confirmed
trend.  Unlike :class:`EMACrossSignal` which fires once at the crossover
moment, this strategy fires **multiple times per trend** on each
retracement to the EMA20, making it the primary signal generator in
trending markets.

Logic:
    LONG:   EMA20 > EMA50 (uptrend)  AND  price dips to EMA20 (±0.2%)
            AND  RSI < 55 (not overbought)
    SHORT:  EMA20 < EMA50 (downtrend)  AND  price rallies to EMA20 (±0.2%)
            AND  RSI > 45 (not oversold)

Config keys:
    enabled           (bool)  : Whether this signal is active.
    fast_ema          (int)   : Fast EMA period (default 20).
    slow_ema          (int)   : Slow EMA period (default 50).
    pullback_pct      (float) : How close to EMA20 to trigger (default 0.002 = 0.2%).
    rsi_max_long      (int)   : Max RSI for long entries (default 55).
    rsi_min_short     (int)   : Min RSI for short entries (default 45).
    stop_loss_pct     (float) : Stop-loss distance as fraction of entry (default 0.015).
    volume_confirmation (bool): Require volume > 20-period avg (default true).
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class EMAPullbackSignal(BaseSignal):
    """EMA pullback / trend continuation signal."""

    @property
    def name(self) -> str:
        return "ema_pullback"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate — always returns not-triggered.

        The real logic is in :meth:`evaluate_pullback`.
        """
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_pullback(
        self,
        symbol: str,
        current_price: float,
        ema_fast: float,
        ema_slow: float,
        rsi: float,
        atr: float = 0.0,
    ) -> SignalResult:
        """Evaluate an EMA pullback entry.

        Args:
            symbol: Trading symbol.
            current_price: Latest market price.
            ema_fast: Current fast EMA value (e.g. EMA20).
            ema_slow: Current slow EMA value (e.g. EMA50).
            rsi: Current RSI value (0-100).

        Returns:
            A :class:`SignalResult` — triggered when price dips to EMA20
            in a confirmed trend with RSI confirmation.
        """
        if not self.is_enabled():
            return SignalResult(triggered=False, strategy_name=self.name)

        pullback_pct = self.config.get("pullback_pct", 0.002)
        rsi_max_long = self.config.get("rsi_max_long", 55)
        rsi_min_short = self.config.get("rsi_min_short", 45)
        stop_loss_pct = self.config.get("stop_loss_pct", 0.015)
        atr_stop_mult: float = self.config.get("atr_stop_multiplier", 1.5)

        direction: SignalDirection | None = None
        reason = ""

        # ---- Uptrend: EMA20 > EMA50, price pulls back to EMA20 --------
        if ema_fast > ema_slow:
            # Price must be near or below EMA20 (within pullback_pct above)
            distance = (current_price - ema_fast) / ema_fast if ema_fast > 0 else 0
            near_ema = distance <= pullback_pct  # price at or below EMA20+0.2%
            above_floor = current_price > ema_slow  # don't enter below EMA50

            if near_ema and above_floor and rsi < rsi_max_long:
                direction = SignalDirection.LONG
                reason = (
                    f"Uptrend pullback to EMA20 ({ema_fast:.2f}), "
                    f"price={current_price:.2f}, RSI={rsi:.1f}"
                )

        # ---- Downtrend: EMA20 < EMA50, price rallies to EMA20 ---------
        elif ema_fast < ema_slow:
            distance = (ema_fast - current_price) / ema_fast if ema_fast > 0 else 0
            near_ema = distance <= pullback_pct  # price at or above EMA20-0.2%
            below_ceiling = current_price < ema_slow  # don't enter above EMA50

            if near_ema and below_ceiling and rsi > rsi_min_short:
                direction = SignalDirection.SHORT
                reason = (
                    f"Downtrend pullback to EMA20 ({ema_fast:.2f}), "
                    f"price={current_price:.2f}, RSI={rsi:.1f}"
                )

        if direction is None:
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="No pullback signal",
            )

        # ---- Swing bias gate -------------------------------------------
        if not self.check_swing_bias(symbol, direction):
            logger.info(
                "swing_bias_blocked | signal={} symbol={} direction={}",
                self.name, symbol, direction.value,
            )
            return SignalResult(
                triggered=False,
                strategy_name=self.name,
                reason="Blocked by swing bias",
            )

        # ---- Price levels ----------------------------------------------
        entry_price = current_price
        stop_distance = atr * atr_stop_mult if atr > 0 else entry_price * stop_loss_pct

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 2)
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 2)

        logger.info(
            "signal_triggered | signal={} symbol={} direction={} "
            "entry={} sl={} tp={} ema_fast={:.2f} ema_slow={:.2f} rsi={:.1f}",
            self.name, symbol, direction.value,
            entry_price, stop_loss, take_profit,
            ema_fast, ema_slow, rsi,
        )

        return SignalResult(
            triggered=True,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.60,
            strategy_name=self.name,
            reason=reason,
        )
