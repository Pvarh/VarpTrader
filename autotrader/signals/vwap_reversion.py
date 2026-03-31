"""VWAP Reversion signal with ATR-dynamic bands, volume confirmation, and slope filter."""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class VWAPReversionSignal(BaseSignal):
    """Mean-reversion signal: fade price when it deviates from VWAP.

    Uses ATR-based dynamic bands instead of fixed percentage, volume
    confirmation to filter noise, and VWAP slope to avoid fading trends.
    """

    @property
    def name(self) -> str:
        return "vwap_reversion"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate — not used directly. See evaluate_from_vwap()."""
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_from_vwap(
        self,
        symbol: str,
        current_price: float,
        vwap: float,
        atr: float,
        vwap_slope: float,
        current_volume: float,
        avg_volume: float,
        market: str,
    ) -> SignalResult:
        """Evaluate VWAP reversion with dynamic bands and filters."""
        no_signal = SignalResult(triggered=False, strategy_name=self.name)

        if not self.is_enabled():
            return no_signal

        band_mult = self.config.get("atr_band_multiplier", 1.5)
        vol_mult = self.config.get("volume_confirmation_mult", 1.2)
        slope_max = self.config.get("slope_max", 0.001)
        sl_atr_mult = self.config.get("stop_loss_atr_mult", 1.0)
        rr_ratio = self.config.get("rr_ratio", 2.0)

        # Guard: slope too steep
        if abs(vwap_slope) > slope_max:
            logger.debug(
                "vwap_reversion_slope_blocked | symbol={} slope={:.6f} max={}",
                symbol, vwap_slope, slope_max,
            )
            return SignalResult(
                triggered=False, strategy_name=self.name,
                reason=f"VWAP slope {vwap_slope:.6f} exceeds max {slope_max}",
            )

        # Guard: volume too low
        vol_threshold = avg_volume * vol_mult
        if current_volume < vol_threshold:
            logger.debug(
                "vwap_reversion_volume_blocked | symbol={} vol={} threshold={}",
                symbol, current_volume, vol_threshold,
            )
            return SignalResult(
                triggered=False, strategy_name=self.name,
                reason=f"Volume {current_volume:.0f} below threshold {vol_threshold:.0f}",
            )

        # Dynamic bands
        band_distance = atr * band_mult
        lower_band = vwap - band_distance
        upper_band = vwap + band_distance

        direction = None
        if current_price <= lower_band:
            direction = SignalDirection.LONG
        elif current_price >= upper_band:
            direction = SignalDirection.SHORT
        else:
            return no_signal

        # Swing bias check
        if self.check_swing_bias(symbol, direction):
            return SignalResult(
                triggered=False, strategy_name=self.name,
                reason=f"Swing bias blocks {direction.value}",
            )

        # Price levels
        entry_price = current_price
        sl_distance = atr * sl_atr_mult

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - sl_distance
            take_profit = vwap
        else:
            stop_loss = entry_price + sl_distance
            take_profit = vwap

        logger.info(
            "vwap_reversion_triggered | symbol={} dir={} entry={:.2f} sl={:.2f} tp={:.2f} "
            "vwap={:.2f} atr={:.4f} slope={:.6f}",
            symbol, direction.value, entry_price, stop_loss, take_profit,
            vwap, atr, vwap_slope,
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
            reason=f"VWAP reversion {direction.value}: price {entry_price:.2f} "
                   f"beyond {'lower' if direction == SignalDirection.LONG else 'upper'} "
                   f"band, slope flat ({vwap_slope:.6f}), volume confirmed",
        )
