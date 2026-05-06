"""Squeeze Momentum signal — John Carter TTM Squeeze adaptation.

Detects low-volatility compression (Bollinger Bands inside Keltner Channels)
and trades the breakout direction using MACD histogram momentum.

When BB contracts inside KC, volatility is compressed ("squeeze on").
When BB expands outside KC, the squeeze releases ("squeeze off").
The direction of the breakout is determined by the MACD histogram slope
at the moment of release.

This strategy catches explosive moves after consolidation periods.
"""

from __future__ import annotations

import math

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult
from signals.indicators import Indicators


class SqueezeMomentumSignal(BaseSignal):
    """Squeeze momentum breakout signal."""

    @property
    def name(self) -> str:
        return "squeeze_momentum"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate — not used directly. See evaluate_squeeze()."""
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_squeeze(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
        atr: float = 0.0,
    ) -> SignalResult:
        """Evaluate squeeze momentum breakout."""
        no_signal = SignalResult(triggered=False, strategy_name=self.name)

        if not self.is_enabled():
            return no_signal

        bb_period = self.config.get("bb_period", 20)
        bb_std = self.config.get("bb_std", 2.0)
        kc_period = self.config.get("kc_period", 20)
        kc_atr_mult = self.config.get("kc_atr_mult", 1.5)
        min_squeeze_bars = self.config.get("min_squeeze_bars", 6)
        sl_pct = self.config.get("stop_loss_pct", 0.015)
        atr_stop_mult = self.config.get("atr_stop_multiplier", 1.5)
        rr_ratio = self.config.get("rr_ratio", 2.0)

        min_bars = max(bb_period, kc_period) + min_squeeze_bars + 5
        if len(candles) < min_bars:
            return no_signal

        closes = [c.close for c in candles]

        # Compute indicators
        bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(
            closes, period=bb_period, std_dev=bb_std,
        )
        kc_upper, kc_middle, kc_lower = Indicators.keltner_channels(
            candles, ema_period=kc_period, atr_mult=kc_atr_mult,
        )

        # MACD histogram for momentum direction
        _, _, histogram = Indicators.macd(closes, 12, 26, 9)

        # Detect squeeze state for recent bars
        # squeeze_on = BB inside KC (low volatility)
        squeeze_states: list[bool] = []
        lookback = min_squeeze_bars + 5
        for i in range(-lookback, 0):
            try:
                bb_u = bb_upper[i]
                bb_l = bb_lower[i]
                kc_u = kc_upper[i]
                kc_l = kc_lower[i]
                if any(math.isnan(v) for v in (bb_u, bb_l, kc_u, kc_l)):
                    squeeze_states.append(False)
                    continue
                squeeze_on = bb_l > kc_l and bb_u < kc_u
                squeeze_states.append(squeeze_on)
            except (IndexError, TypeError):
                squeeze_states.append(False)

        if len(squeeze_states) < min_squeeze_bars + 1:
            return no_signal

        # Check: was in squeeze for at least min_squeeze_bars, now released
        # Recent bars should show squeeze, current bar should show release
        recent_squeeze = squeeze_states[-(min_squeeze_bars + 1):-1]
        current_released = not squeeze_states[-1]

        if not all(recent_squeeze) or not current_released:
            return no_signal

        # Momentum direction from MACD histogram
        hist_curr = histogram[-1] if histogram else 0.0
        hist_prev = histogram[-2] if len(histogram) >= 2 else 0.0

        if math.isnan(hist_curr) or math.isnan(hist_prev):
            return no_signal

        # Histogram rising = bullish momentum, falling = bearish
        if hist_curr > 0 and hist_curr > hist_prev:
            direction = SignalDirection.LONG
        elif hist_curr < 0 and hist_curr < hist_prev:
            direction = SignalDirection.SHORT
        else:
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

        logger.info(
            "squeeze_momentum_triggered | symbol={} dir={} entry={:.2f} sl={:.2f} "
            "tp={:.2f} squeeze_bars={} hist={:.6f}",
            symbol, direction.value, entry_price, stop_loss, take_profit,
            min_squeeze_bars, hist_curr,
        )

        return SignalResult(
            triggered=True,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.70,
            strategy_name=self.name,
            reason=f"Squeeze release {direction.value}: {min_squeeze_bars} bars "
                   f"compressed, momentum {'rising' if direction == SignalDirection.LONG else 'falling'}",
        )
