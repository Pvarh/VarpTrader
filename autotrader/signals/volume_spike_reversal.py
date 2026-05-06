"""Volume Spike Reversal signal — exhaustion/capitulation detector.

Catches extreme volume events combined with RSI extremes and rejection
wicks that signal capitulation (selling climax) or euphoria (buying climax).

Entry criteria for LONG (capitulation buy):
- Volume > 3x 20-bar average (panic selling / liquidation cascade)
- RSI < 25 (deeply oversold)
- Lower wick > 50% of candle range (buyers stepping in, price rejected lows)

Entry criteria for SHORT (euphoria sell):
- Volume > 3x 20-bar average (FOMO buying)
- RSI > 75 (deeply overbought)
- Upper wick > 50% of candle range (sellers stepping in, price rejected highs)

The idea: extreme volume + RSI extreme + rejection wick = the move is
exhausted and a reversal is imminent. Higher confidence than regular
mean-reversion because volume confirms real participation.
"""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult


class VolumeSpikeReversalSignal(BaseSignal):
    """Volume spike exhaustion reversal signal."""

    @property
    def name(self) -> str:
        return "volume_spike_reversal"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate — not used. See evaluate_spike()."""
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_spike(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
        atr: float = 0.0,
    ) -> SignalResult:
        """Evaluate volume spike reversal."""
        no_signal = SignalResult(triggered=False, strategy_name=self.name)

        if not self.is_enabled():
            return no_signal

        vol_mult = self.config.get("volume_spike_mult", 3.0)
        rsi_oversold = self.config.get("rsi_oversold", 25)
        rsi_overbought = self.config.get("rsi_overbought", 75)
        min_wick_ratio = self.config.get("min_wick_ratio", 0.5)
        sl_pct = self.config.get("stop_loss_pct", 0.012)
        atr_stop_mult = self.config.get("atr_stop_multiplier", 1.5)
        rr_ratio = self.config.get("rr_ratio", 2.0)

        if len(candles) < 21:
            return no_signal

        # Current candle
        c = candles[-1]
        candle_range = c.high - c.low
        if candle_range <= 0:
            return no_signal

        # Volume spike check
        volumes = [bar.volume for bar in candles[-21:-1]]
        avg_vol = sum(volumes) / len(volumes) if volumes else 1.0
        if c.volume < avg_vol * vol_mult:
            return no_signal

        # RSI
        from signals.indicators import Indicators
        closes = [bar.close for bar in candles]
        rsi_series = Indicators.rsi(closes, 14)
        rsi = rsi_series[-1] if rsi_series else 50.0

        # Wick analysis
        body_top = max(c.open, c.close)
        body_bottom = min(c.open, c.close)
        lower_wick = body_bottom - c.low
        upper_wick = c.high - body_top
        lower_wick_ratio = lower_wick / candle_range
        upper_wick_ratio = upper_wick / candle_range

        direction = None
        reason = ""

        # Capitulation buy: oversold + volume spike + lower rejection wick
        if rsi < rsi_oversold and lower_wick_ratio >= min_wick_ratio:
            direction = SignalDirection.LONG
            reason = (
                f"Capitulation reversal: vol {c.volume:.0f} "
                f"({c.volume / avg_vol:.1f}x avg), RSI={rsi:.1f}, "
                f"lower wick {lower_wick_ratio:.0%} of range"
            )

        # Euphoria sell: overbought + volume spike + upper rejection wick
        elif rsi > rsi_overbought and upper_wick_ratio >= min_wick_ratio:
            direction = SignalDirection.SHORT
            reason = (
                f"Euphoria reversal: vol {c.volume:.0f} "
                f"({c.volume / avg_vol:.1f}x avg), RSI={rsi:.1f}, "
                f"upper wick {upper_wick_ratio:.0%} of range"
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

        # Higher confidence when volume spike is extreme
        vol_ratio = c.volume / avg_vol
        confidence = min(0.80, 0.60 + (vol_ratio - vol_mult) * 0.05)

        logger.info(
            "volume_spike_reversal_triggered | symbol={} dir={} entry={:.2f} "
            "sl={:.2f} tp={:.2f} vol_ratio={:.1f}x rsi={:.1f}",
            symbol, direction.value, entry_price, stop_loss, take_profit,
            vol_ratio, rsi,
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
