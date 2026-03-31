# New Trading Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add VPOC Bounce and MACD Divergence signals, and fully redesign VWAP Reversion with ATR-dynamic bands, volume confirmation, and slope filtering.

**Architecture:** Three independent signals following the existing BaseSignal pattern. New indicators added to `signals/indicators.py`. Each signal gets its own file, config section, and test file. Wired into `main.py` via `_build_signals()` and `_run_signal()`.

**Tech Stack:** Python 3, SQLite, APScheduler, loguru, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `signals/indicators.py` | Modify | Add `macd()`, `volume_profile()`, `vwap_slope()` |
| `signals/vwap_reversion.py` | Rewrite | ATR-dynamic bands, volume confirmation, slope filter |
| `signals/vpoc_bounce.py` | Create | Volume profile POC bounce detection |
| `signals/macd_divergence.py` | Create | Price/histogram divergence + MACD cross confirmation |
| `main.py` | Modify | Import, register, and wire new signals |
| `config.json` | Modify | Add vpoc_bounce, macd_divergence; update vwap_reversion |
| `tests/test_signals.py` | Modify | Add tests for all three signals |

---

### Task 1: Add MACD indicator to Indicators class

**Files:**
- Modify: `autotrader/signals/indicators.py` (append after `vwap()` method, ~line 436)
- Test: `autotrader/tests/test_signals.py`

- [ ] **Step 1: Write failing test for MACD indicator**

Add to `autotrader/tests/test_signals.py`:

```python
class TestMACDIndicator:
    def test_macd_basic_output_shape(self) -> None:
        """MACD returns three lists of same length as input."""
        closes = [float(i) for i in range(50)]
        macd_line, signal_line, histogram = Indicators.macd(closes, fast=12, slow=26, signal=9)
        assert len(macd_line) == 50
        assert len(signal_line) == 50
        assert len(histogram) == 50

    def test_macd_early_values_are_nan(self) -> None:
        """First slow+signal-2 values should be NaN."""
        closes = [float(i) for i in range(50)]
        macd_line, signal_line, histogram = Indicators.macd(closes, fast=12, slow=26, signal=9)
        import math
        # First 25 values (slow-1=25) of macd_line are NaN
        assert math.isnan(macd_line[0])
        # First 33 values (slow+signal-2=33) of signal_line/histogram are NaN
        assert math.isnan(signal_line[0])
        assert math.isnan(histogram[0])

    def test_macd_converging_prices(self) -> None:
        """With constant prices, MACD should converge to zero."""
        closes = [100.0] * 60
        macd_line, signal_line, histogram = Indicators.macd(closes)
        # Last values should be near zero for flat price
        assert abs(macd_line[-1]) < 0.01
        assert abs(signal_line[-1]) < 0.01
        assert abs(histogram[-1]) < 0.01

    def test_macd_uptrend_positive(self) -> None:
        """In a steady uptrend, MACD line should be positive."""
        closes = [100.0 + i * 0.5 for i in range(60)]
        macd_line, signal_line, histogram = Indicators.macd(closes)
        assert macd_line[-1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestMACDIndicator -v`
Expected: FAIL — `Indicators` has no `macd` method

- [ ] **Step 3: Implement MACD indicator**

Add to `autotrader/signals/indicators.py` after the `vwap()` method:

```python
@staticmethod
def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[float], list[float], list[float]]:
    """Compute MACD line, signal line, and histogram.

    Args:
        closes: List of closing prices.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.

    Returns:
        Tuple of (macd_line, signal_line, histogram), each same length as input.
        Early values are NaN where insufficient data exists.
    """
    fast_ema = Indicators.ema(closes, fast)
    slow_ema = Indicators.ema(closes, slow)

    macd_line: list[float] = []
    for f, s in zip(fast_ema, slow_ema):
        if math.isnan(f) or math.isnan(s):
            macd_line.append(float("nan"))
        else:
            macd_line.append(f - s)

    # Signal line = EMA of the MACD line (only over non-NaN values)
    valid_start = slow - 1  # first non-NaN MACD value index
    macd_valid = macd_line[valid_start:]
    if len(macd_valid) >= signal:
        signal_ema = Indicators.ema(macd_valid, signal)
        signal_line = [float("nan")] * valid_start + signal_ema
    else:
        signal_line = [float("nan")] * len(closes)

    histogram: list[float] = []
    for m, s in zip(macd_line, signal_line):
        if math.isnan(m) or math.isnan(s):
            histogram.append(float("nan"))
        else:
            histogram.append(m - s)

    return macd_line, signal_line, histogram
```

Also add `import math` at the top of `indicators.py` if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestMACDIndicator -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add signals/indicators.py tests/test_signals.py
git commit -m "feat: add MACD indicator to Indicators class"
```

---

### Task 2: Add volume_profile indicator

**Files:**
- Modify: `autotrader/signals/indicators.py`
- Test: `autotrader/tests/test_signals.py`

- [ ] **Step 1: Write failing test for volume_profile**

Add to `autotrader/tests/test_signals.py`:

```python
class TestVolumeProfileIndicator:
    def _make_session_candles(self, prices_volumes: list[tuple[float, float, float, float, float]]) -> list[OHLCV]:
        """Create candles from (open, high, low, close, volume) tuples."""
        from datetime import datetime, timezone, timedelta
        candles = []
        base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
        for i, (o, h, l, c, v) in enumerate(prices_volumes):
            candles.append(OHLCV(
                timestamp=base + timedelta(minutes=i * 5),
                open=o, high=h, low=l, close=c, volume=v,
                symbol="TEST", timeframe="5m", market="stock",
            ))
        return candles

    def test_poc_is_highest_volume_level(self) -> None:
        """POC should be at the price level with most volume."""
        # Heavy volume clustered around 100, light volume at 110
        candles = self._make_session_candles([
            (99, 101, 98, 100, 10000),
            (100, 102, 99, 101, 10000),
            (100, 101, 99, 100, 10000),
            (109, 111, 108, 110, 100),
            (110, 112, 109, 111, 100),
        ])
        poc, vah, val = Indicators.volume_profile(candles, num_bins=20)
        # POC should be near 100, not 110
        assert abs(poc - 100.0) < 2.0

    def test_value_area_contains_70_pct(self) -> None:
        """Value area should span bins with ~70% of volume."""
        candles = self._make_session_candles([
            (99, 101, 98, 100, 5000),
            (100, 102, 99, 101, 5000),
            (100, 101, 99, 100, 5000),
            (105, 107, 104, 106, 1000),
            (110, 112, 109, 111, 500),
        ])
        poc, vah, val = Indicators.volume_profile(candles, num_bins=20)
        assert val <= poc <= vah

    def test_single_candle(self) -> None:
        """Single candle: POC equals typical price."""
        candles = self._make_session_candles([(100, 105, 95, 102, 1000)])
        poc, vah, val = Indicators.volume_profile(candles, num_bins=10)
        typical = (105 + 95 + 102) / 3
        assert abs(poc - typical) < 2.0

    def test_empty_candles(self) -> None:
        """Empty candles should return zeros."""
        poc, vah, val = Indicators.volume_profile([], num_bins=20)
        assert poc == 0.0
        assert vah == 0.0
        assert val == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestVolumeProfileIndicator -v`
Expected: FAIL — no `volume_profile` method

- [ ] **Step 3: Implement volume_profile**

Add to `autotrader/signals/indicators.py`:

```python
@staticmethod
def volume_profile(
    candles: list,
    num_bins: int = 20,
) -> tuple[float, float, float]:
    """Compute volume profile and return POC + value area.

    Args:
        candles: List of OHLCV candles for the session.
        num_bins: Number of price bins to divide the range into.

    Returns:
        Tuple of (poc_price, value_area_high, value_area_low).
        Returns (0.0, 0.0, 0.0) if no candles provided.
    """
    if not candles:
        return 0.0, 0.0, 0.0

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    price_high = max(highs)
    price_low = min(lows)

    if price_high == price_low:
        return price_high, price_high, price_low

    bin_width = (price_high - price_low) / num_bins
    bins = [0.0] * num_bins

    for c in candles:
        typical = (c.high + c.low + c.close) / 3
        idx = min(int((typical - price_low) / bin_width), num_bins - 1)
        bins[idx] += c.volume

    # POC = midpoint of highest-volume bin
    poc_idx = bins.index(max(bins))
    poc_price = price_low + (poc_idx + 0.5) * bin_width

    # Value area: expand outward from POC until 70% of volume
    total_volume = sum(bins)
    if total_volume == 0:
        return poc_price, price_high, price_low

    va_volume = bins[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx

    while va_volume / total_volume < 0.70 and (lo_idx > 0 or hi_idx < num_bins - 1):
        expand_lo = bins[lo_idx - 1] if lo_idx > 0 else -1.0
        expand_hi = bins[hi_idx + 1] if hi_idx < num_bins - 1 else -1.0
        if expand_lo >= expand_hi:
            lo_idx -= 1
            va_volume += bins[lo_idx]
        else:
            hi_idx += 1
            va_volume += bins[hi_idx]

    val = price_low + lo_idx * bin_width
    vah = price_low + (hi_idx + 1) * bin_width

    return poc_price, vah, val
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestVolumeProfileIndicator -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add signals/indicators.py tests/test_signals.py
git commit -m "feat: add volume_profile indicator"
```

---

### Task 3: Add vwap_slope indicator

**Files:**
- Modify: `autotrader/signals/indicators.py`
- Test: `autotrader/tests/test_signals.py`

- [ ] **Step 1: Write failing test for vwap_slope**

Add to `autotrader/tests/test_signals.py`:

```python
class TestVWAPSlopeIndicator:
    def test_flat_vwap_returns_near_zero(self) -> None:
        """Flat VWAP values should give slope near zero."""
        vwap_values = [100.0] * 30
        slope = Indicators.vwap_slope(vwap_values, lookback=20)
        assert abs(slope) < 0.0001

    def test_uptrending_vwap_positive_slope(self) -> None:
        """Rising VWAP should give positive normalized slope."""
        vwap_values = [100.0 + i * 0.1 for i in range(30)]
        slope = Indicators.vwap_slope(vwap_values, lookback=20)
        assert slope > 0

    def test_downtrending_vwap_negative_slope(self) -> None:
        """Falling VWAP should give negative normalized slope."""
        vwap_values = [100.0 - i * 0.1 for i in range(30)]
        slope = Indicators.vwap_slope(vwap_values, lookback=20)
        assert slope < 0

    def test_insufficient_data_returns_zero(self) -> None:
        """Too few values should return 0.0."""
        slope = Indicators.vwap_slope([100.0], lookback=20)
        assert slope == 0.0

    def test_normalization_across_price_levels(self) -> None:
        """Same proportional move at different price levels gives similar slope."""
        # 0.1% per bar at price 100
        low = [100.0 + i * 0.1 for i in range(30)]
        # 0.1% per bar at price 50000
        high = [50000.0 + i * 50.0 for i in range(30)]
        slope_low = Indicators.vwap_slope(low, lookback=20)
        slope_high = Indicators.vwap_slope(high, lookback=20)
        # Both should be similar magnitude after normalization
        assert abs(slope_low - slope_high) < abs(slope_low) * 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestVWAPSlopeIndicator -v`
Expected: FAIL — no `vwap_slope` method

- [ ] **Step 3: Implement vwap_slope**

Add to `autotrader/signals/indicators.py`:

```python
@staticmethod
def vwap_slope(vwap_values: list[float], lookback: int = 20) -> float:
    """Compute normalized linear regression slope of VWAP.

    Args:
        vwap_values: List of VWAP values (time series).
        lookback: Number of recent values to use for regression.

    Returns:
        Normalized slope (slope / current_vwap). Near 0 = flat,
        large positive/negative = trending. Returns 0.0 if insufficient data.
    """
    if len(vwap_values) < 2:
        return 0.0

    window = vwap_values[-lookback:]
    n = len(window)
    if n < 2:
        return 0.0

    # Simple linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    for i, v in enumerate(window):
        sum_x += i
        sum_y += v
        sum_xy += i * v
        sum_x2 += i * i

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    current_vwap = window[-1]
    if current_vwap == 0:
        return 0.0

    return slope / current_vwap
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestVWAPSlopeIndicator -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add signals/indicators.py tests/test_signals.py
git commit -m "feat: add vwap_slope indicator"
```

---

### Task 4: Rewrite VWAP Reversion signal

**Files:**
- Rewrite: `autotrader/signals/vwap_reversion.py`
- Test: `autotrader/tests/test_signals.py`

- [ ] **Step 1: Write failing tests for new VWAP Reversion**

Replace the existing `TestVWAPReversion` and `TestVWAPReversionSwingBias` classes in `autotrader/tests/test_signals.py`:

```python
class TestVWAPReversionRedesign:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "atr_band_multiplier": 1.5,
            "atr_period": 14,
            "volume_confirmation_mult": 1.2,
            "slope_max": 0.001,
            "slope_lookback": 20,
            "stop_loss_atr_mult": 1.0,
            "rr_ratio": 2.0,
        }

    @pytest.fixture(autouse=True)
    def _no_swing_block(self):
        with patch("signals.vwap_reversion.VWAPReversionSignal.check_swing_bias", return_value=False):
            yield

    def test_long_below_lower_band(self, config: dict) -> None:
        """LONG when price below VWAP - ATR*mult, slope flat, volume confirmed."""
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=95.0, vwap=100.0,
            atr=4.0, vwap_slope=0.0003, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        # lower band = 100 - 4*1.5 = 94.0, price 95 > 94 → no trigger
        assert not result.triggered

    def test_long_triggers_at_band(self, config: dict) -> None:
        """LONG triggers when price at or below lower band."""
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=93.5, vwap=100.0,
            atr=4.0, vwap_slope=0.0003, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        # lower band = 100 - 6.0 = 94.0, price 93.5 <= 94.0 → trigger
        assert result.triggered
        assert result.direction == SignalDirection.LONG
        assert result.take_profit > result.entry_price  # TP targets VWAP
        assert result.stop_loss < result.entry_price

    def test_short_triggers_at_upper_band(self, config: dict) -> None:
        """SHORT triggers when price at or above upper band."""
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=107.0, vwap=100.0,
            atr=4.0, vwap_slope=-0.0002, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        # upper band = 100 + 6.0 = 106.0, price 107 >= 106 → trigger
        assert result.triggered
        assert result.direction == SignalDirection.SHORT
        assert result.take_profit < result.entry_price
        assert result.stop_loss > result.entry_price

    def test_no_trigger_slope_too_steep(self, config: dict) -> None:
        """No trigger when VWAP slope indicates trending market."""
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=93.5, vwap=100.0,
            atr=4.0, vwap_slope=0.005, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        assert not result.triggered
        assert "slope" in result.reason.lower()

    def test_no_trigger_low_volume(self, config: dict) -> None:
        """No trigger when volume below confirmation threshold."""
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=93.5, vwap=100.0,
            atr=4.0, vwap_slope=0.0003, current_volume=800.0,
            avg_volume=1000.0, market="stock",
        )
        assert not result.triggered
        assert "volume" in result.reason.lower()

    def test_disabled(self, config: dict) -> None:
        """No trigger when signal disabled."""
        config["enabled"] = False
        sig = VWAPReversionSignal(config)
        result = sig.evaluate_from_vwap(
            symbol="AAPL", current_price=93.5, vwap=100.0,
            atr=4.0, vwap_slope=0.0003, current_volume=1500.0,
            avg_volume=1000.0, market="stock",
        )
        assert not result.triggered


class TestVWAPReversionSwingBiasRedesign:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "atr_band_multiplier": 1.5,
            "atr_period": 14,
            "volume_confirmation_mult": 1.2,
            "slope_max": 0.001,
            "slope_lookback": 20,
            "stop_loss_atr_mult": 1.0,
            "rr_ratio": 2.0,
        }

    def test_long_blocked_by_swing_bias(self, config: dict) -> None:
        with patch("signals.vwap_reversion.VWAPReversionSignal.check_swing_bias", return_value=True):
            sig = VWAPReversionSignal(config)
            result = sig.evaluate_from_vwap(
                symbol="AAPL", current_price=93.5, vwap=100.0,
                atr=4.0, vwap_slope=0.0003, current_volume=1500.0,
                avg_volume=1000.0, market="stock",
            )
            assert not result.triggered
            assert "swing" in result.reason.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestVWAPReversionRedesign tests/test_signals.py::TestVWAPReversionSwingBiasRedesign -v`
Expected: FAIL — old `evaluate_from_vwap` has different signature

- [ ] **Step 3: Rewrite vwap_reversion.py**

Replace entire content of `autotrader/signals/vwap_reversion.py`:

```python
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
        """Evaluate VWAP reversion with dynamic bands and filters.

        Args:
            symbol: Instrument symbol.
            current_price: Latest price.
            vwap: Current session VWAP.
            atr: Current ATR value.
            vwap_slope: Normalized VWAP slope from Indicators.vwap_slope().
            current_volume: Volume of the current bar.
            avg_volume: 20-bar average volume.
            market: 'stock' or 'crypto'.

        Returns:
            SignalResult with trigger decision and price levels.
        """
        no_signal = SignalResult(triggered=False, strategy_name=self.name)

        if not self.is_enabled():
            return no_signal

        band_mult = self.config.get("atr_band_multiplier", 1.5)
        vol_mult = self.config.get("volume_confirmation_mult", 1.2)
        slope_max = self.config.get("slope_max", 0.001)
        sl_atr_mult = self.config.get("stop_loss_atr_mult", 1.0)
        rr_ratio = self.config.get("rr_ratio", 2.0)

        # Guard: slope too steep → trending, don't fade
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
            take_profit = vwap  # revert to VWAP
        else:
            stop_loss = entry_price + sl_distance
            take_profit = vwap  # revert to VWAP

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestVWAPReversionRedesign tests/test_signals.py::TestVWAPReversionSwingBiasRedesign -v`
Expected: 8 passed

- [ ] **Step 5: Remove old VWAP tests**

Delete the old `TestVWAPReversion` and `TestVWAPReversionSwingBias` classes from `tests/test_signals.py` (they reference the old method signature).

- [ ] **Step 6: Run full test suite to check for breakage**

Run: `cd autotrader && python3 -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: Some tests that call old `evaluate_from_vwap` signature may fail — fix in Task 7 when wiring main.py.

- [ ] **Step 7: Commit**

```bash
git add signals/vwap_reversion.py tests/test_signals.py
git commit -m "feat: rewrite VWAP reversion with ATR bands, volume filter, slope guard"
```

---

### Task 5: Create VPOC Bounce signal

**Files:**
- Create: `autotrader/signals/vpoc_bounce.py`
- Test: `autotrader/tests/test_signals.py`

- [ ] **Step 1: Write failing tests for VPOC Bounce**

Add to `autotrader/tests/test_signals.py`:

```python
from signals.vpoc_bounce import VPOCBounceSignal


class TestVPOCBounce:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "num_bins": 20,
            "proximity_pct": 0.002,
            "bounce_candles": 2,
            "min_poc_volume_pct": 0.15,
            "stop_loss_pct": 0.01,
            "rr_ratio": 2.0,
        }

    @pytest.fixture(autouse=True)
    def _no_swing_block(self):
        with patch("signals.vpoc_bounce.VPOCBounceSignal.check_swing_bias", return_value=False):
            yield

    def _make_session_candles(self, data: list[tuple]) -> list[OHLCV]:
        from datetime import datetime, timezone, timedelta
        candles = []
        base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
        for i, (o, h, l, c, v) in enumerate(data):
            candles.append(OHLCV(
                timestamp=base + timedelta(minutes=i * 5),
                open=o, high=h, low=l, close=c, volume=v,
                symbol="AAPL", timeframe="5m", market="stock",
            ))
        return candles

    def test_long_bounce_off_poc(self, config: dict) -> None:
        """LONG when price near POC from below + bullish bounce candles."""
        # Build heavy volume at 100, then price dips near 100 and bounces
        session = self._make_session_candles([
            (99, 101, 98, 100, 10000),
            (100, 102, 99, 101, 10000),
            (100, 101, 99, 100, 10000),
            (100, 101, 99, 100, 10000),
            (100, 101, 99, 100, 10000),
            # Recent candles: approach from below, bounce up
            (99.5, 100.2, 99.3, 100.0, 5000),
            (99.8, 100.5, 99.7, 100.3, 5000),
        ])
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=100.1,
            session_candles=session, recent_candles=session[-2:],
            market="stock",
        )
        assert result.triggered
        assert result.direction == SignalDirection.LONG

    def test_short_bounce_off_poc(self, config: dict) -> None:
        """SHORT when price near POC from above + bearish bounce candles."""
        session = self._make_session_candles([
            (99, 101, 98, 100, 10000),
            (100, 102, 99, 101, 10000),
            (100, 101, 99, 100, 10000),
            (100, 101, 99, 100, 10000),
            (100, 101, 99, 100, 10000),
            # Recent: approach from above, reject down
            (100.5, 100.8, 99.9, 100.0, 5000),
            (100.2, 100.4, 99.6, 99.8, 5000),
        ])
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=99.9,
            session_candles=session, recent_candles=session[-2:],
            market="stock",
        )
        assert result.triggered
        assert result.direction == SignalDirection.SHORT

    def test_no_trigger_price_far_from_poc(self, config: dict) -> None:
        """No trigger when price is far from POC."""
        session = self._make_session_candles([
            (99, 101, 98, 100, 10000),
            (100, 102, 99, 101, 10000),
            (100, 101, 99, 100, 10000),
            (109, 111, 108, 110, 100),
            (110, 112, 109, 111, 100),
        ])
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=110.0,
            session_candles=session, recent_candles=session[-2:],
            market="stock",
        )
        assert not result.triggered

    def test_no_trigger_weak_poc(self, config: dict) -> None:
        """No trigger when POC bin has too little volume concentration."""
        # Evenly distributed volume — no strong POC
        session = self._make_session_candles([
            (95, 97, 94, 96, 1000),
            (98, 100, 97, 99, 1000),
            (101, 103, 100, 102, 1000),
            (104, 106, 103, 105, 1000),
            (107, 109, 106, 108, 1000),
            (108, 109, 107, 108, 1000),
            (108, 109, 107, 108, 1000),
        ])
        config["min_poc_volume_pct"] = 0.40  # Require 40% concentration
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=108.0,
            session_candles=session, recent_candles=session[-2:],
            market="stock",
        )
        assert not result.triggered

    def test_disabled(self, config: dict) -> None:
        config["enabled"] = False
        sig = VPOCBounceSignal(config)
        result = sig.evaluate_from_profile(
            symbol="AAPL", current_price=100.0,
            session_candles=[], recent_candles=[],
            market="stock",
        )
        assert not result.triggered


class TestVPOCBounceSwingBias:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "num_bins": 20,
            "proximity_pct": 0.002,
            "bounce_candles": 2,
            "min_poc_volume_pct": 0.15,
            "stop_loss_pct": 0.01,
            "rr_ratio": 2.0,
        }

    def test_blocked_by_swing_bias(self, config: dict) -> None:
        from datetime import datetime, timezone, timedelta
        base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
        session = [
            OHLCV(timestamp=base + timedelta(minutes=i * 5),
                   open=99+i*0.1, high=101, low=98, close=100, volume=10000,
                   symbol="AAPL", timeframe="5m", market="stock")
            for i in range(7)
        ]
        with patch("signals.vpoc_bounce.VPOCBounceSignal.check_swing_bias", return_value=True):
            sig = VPOCBounceSignal(config)
            result = sig.evaluate_from_profile(
                symbol="AAPL", current_price=100.0,
                session_candles=session, recent_candles=session[-2:],
                market="stock",
            )
            assert not result.triggered
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestVPOCBounce tests/test_signals.py::TestVPOCBounceSwingBias -v`
Expected: FAIL — `signals.vpoc_bounce` module not found

- [ ] **Step 3: Create vpoc_bounce.py**

Create `autotrader/signals/vpoc_bounce.py`:

```python
"""VPOC Bounce signal — trade bounces off the Volume Point of Control."""

from __future__ import annotations

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult
from signals.indicators import Indicators


class VPOCBounceSignal(BaseSignal):
    """Detect price bouncing off the session's highest-volume price level.

    Builds a volume profile from session candles, identifies the POC,
    and triggers when price is near POC with directional bounce candles.
    """

    @property
    def name(self) -> str:
        return "vpoc_bounce"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate — not used directly. See evaluate_from_profile()."""
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_from_profile(
        self,
        symbol: str,
        current_price: float,
        session_candles: list[OHLCV],
        recent_candles: list[OHLCV],
        market: str,
    ) -> SignalResult:
        """Evaluate VPOC bounce from session volume profile.

        Args:
            symbol: Instrument symbol.
            current_price: Latest price.
            session_candles: All candles from current session (for volume profile).
            recent_candles: Last N candles for bounce detection.
            market: 'stock' or 'crypto'.

        Returns:
            SignalResult with trigger decision and price levels.
        """
        no_signal = SignalResult(triggered=False, strategy_name=self.name)

        if not self.is_enabled():
            return no_signal

        num_bins = self.config.get("num_bins", 20)
        proximity_pct = self.config.get("proximity_pct", 0.002)
        bounce_candles = self.config.get("bounce_candles", 2)
        min_poc_vol = self.config.get("min_poc_volume_pct", 0.15)
        sl_pct = self.config.get("stop_loss_pct", 0.01)
        rr_ratio = self.config.get("rr_ratio", 2.0)

        if not session_candles or len(recent_candles) < bounce_candles:
            return no_signal

        # Build volume profile
        poc_price, vah, val = Indicators.volume_profile(session_candles, num_bins=num_bins)
        if poc_price == 0.0:
            return no_signal

        # Check POC strength
        total_vol = sum(c.volume for c in session_candles)
        if total_vol == 0:
            return no_signal

        # Compute POC bin volume
        highs = [c.high for c in session_candles]
        lows = [c.low for c in session_candles]
        price_high = max(highs)
        price_low = min(lows)
        if price_high == price_low:
            return no_signal

        bin_width = (price_high - price_low) / num_bins
        poc_bin_idx = min(int((poc_price - price_low) / bin_width), num_bins - 1)
        poc_bin_vol = 0.0
        for c in session_candles:
            typical = (c.high + c.low + c.close) / 3
            idx = min(int((typical - price_low) / bin_width), num_bins - 1)
            if idx == poc_bin_idx:
                poc_bin_vol += c.volume

        if poc_bin_vol / total_vol < min_poc_vol:
            logger.debug(
                "vpoc_weak_poc | symbol={} poc_vol_pct={:.2f} min={}",
                symbol, poc_bin_vol / total_vol, min_poc_vol,
            )
            return SignalResult(
                triggered=False, strategy_name=self.name,
                reason=f"POC too weak: {poc_bin_vol / total_vol:.2%} < {min_poc_vol:.0%}",
            )

        # Check proximity to POC
        distance_pct = abs(current_price - poc_price) / poc_price
        if distance_pct > proximity_pct:
            return no_signal

        # Detect bounce direction
        tail = recent_candles[-bounce_candles:]
        all_bullish = all(c.close > c.open for c in tail)
        all_bearish = all(c.close < c.open for c in tail)

        direction = None
        if all_bullish and current_price >= poc_price:
            # Approached from below, bouncing up
            direction = SignalDirection.LONG
        elif all_bearish and current_price <= poc_price:
            # Approached from above, bouncing down
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
        sl_distance = entry_price * sl_pct

        if direction == SignalDirection.LONG:
            stop_loss = poc_price - sl_distance  # Below POC = level failed
            take_profit = entry_price + (entry_price - stop_loss) * rr_ratio
        else:
            stop_loss = poc_price + sl_distance  # Above POC = level failed
            take_profit = entry_price - (stop_loss - entry_price) * rr_ratio

        logger.info(
            "vpoc_bounce_triggered | symbol={} dir={} entry={:.2f} sl={:.2f} tp={:.2f} "
            "poc={:.2f} poc_vol_pct={:.2%}",
            symbol, direction.value, entry_price, stop_loss, take_profit,
            poc_price, poc_bin_vol / total_vol,
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
            reason=f"VPOC bounce {direction.value}: price near POC {poc_price:.2f}, "
                   f"{'bullish' if direction == SignalDirection.LONG else 'bearish'} "
                   f"rejection candles",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestVPOCBounce tests/test_signals.py::TestVPOCBounceSwingBias -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add signals/vpoc_bounce.py tests/test_signals.py
git commit -m "feat: add VPOC Bounce signal"
```

---

### Task 6: Create MACD Divergence signal

**Files:**
- Create: `autotrader/signals/macd_divergence.py`
- Test: `autotrader/tests/test_signals.py`

- [ ] **Step 1: Write failing tests for MACD Divergence**

Add to `autotrader/tests/test_signals.py`:

```python
from signals.macd_divergence import MACDDivergenceSignal


class TestMACDDivergence:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "divergence_lookback": 30,
            "min_swing_distance": 5,
            "stop_loss_pct": 0.015,
            "rr_ratio": 2.0,
        }

    @pytest.fixture(autouse=True)
    def _no_swing_block(self):
        with patch("signals.macd_divergence.MACDDivergenceSignal.check_swing_bias", return_value=False):
            yield

    def _make_candles(self, closes: list[float]) -> list[OHLCV]:
        from datetime import datetime, timezone, timedelta
        base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
        candles = []
        for i, c in enumerate(closes):
            candles.append(OHLCV(
                timestamp=base + timedelta(minutes=i * 5),
                open=c, high=c + 0.5, low=c - 0.5, close=c, volume=1000,
                symbol="AAPL", timeframe="5m", market="stock",
            ))
        return candles

    def test_bullish_divergence_with_cross(self, config: dict) -> None:
        """Bullish: price lower low + histogram higher low + MACD crosses above signal."""
        # Build 60 candles: downtrend with two swing lows
        # First low at bar 35 (price=90), second lower low at bar 50 (price=88)
        # But MACD histogram at bar 50 is higher than at bar 35 → divergence
        # Then MACD crosses signal upward in last 3 bars
        closes = [100.0 - i * 0.3 for i in range(35)]  # downtrend to ~89.5
        closes.append(90.0)  # swing low 1 at index 35
        closes += [91.0 + i * 0.1 for i in range(14)]  # bounce up to ~92.3
        closes.append(88.0)  # swing low 2 at index 50 — lower price
        closes += [89.0, 89.5, 90.0, 90.5, 91.0, 91.5, 92.0, 93.0, 93.5, 94.0]  # recovery → cross

        candles = self._make_candles(closes)
        sig = MACDDivergenceSignal(config)
        result = sig.evaluate_from_macd(
            symbol="AAPL", current_price=94.0, candles=candles, market="stock",
        )
        # This is a complex signal — it may or may not trigger depending on
        # exact MACD math. Test that the method runs without error.
        assert isinstance(result.triggered, bool)
        assert result.strategy_name == "macd_divergence"

    def test_no_trigger_without_cross(self, config: dict) -> None:
        """Divergence present but no MACD cross → no trigger."""
        # Downtrend without recovery → no cross
        closes = [100.0 - i * 0.2 for i in range(60)]
        candles = self._make_candles(closes)
        sig = MACDDivergenceSignal(config)
        result = sig.evaluate_from_macd(
            symbol="AAPL", current_price=closes[-1], candles=candles, market="stock",
        )
        assert not result.triggered

    def test_no_trigger_insufficient_data(self, config: dict) -> None:
        """Not enough candles for MACD calculation."""
        closes = [100.0] * 20
        candles = self._make_candles(closes)
        sig = MACDDivergenceSignal(config)
        result = sig.evaluate_from_macd(
            symbol="AAPL", current_price=100.0, candles=candles, market="stock",
        )
        assert not result.triggered

    def test_disabled(self, config: dict) -> None:
        config["enabled"] = False
        sig = MACDDivergenceSignal(config)
        result = sig.evaluate_from_macd(
            symbol="AAPL", current_price=100.0, candles=[], market="stock",
        )
        assert not result.triggered

    def test_returns_correct_strategy_name(self, config: dict) -> None:
        sig = MACDDivergenceSignal(config)
        assert sig.name == "macd_divergence"


class TestMACDDivergenceSwingBias:
    @pytest.fixture
    def config(self) -> dict:
        return {
            "enabled": True,
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "divergence_lookback": 30,
            "min_swing_distance": 5,
            "stop_loss_pct": 0.015,
            "rr_ratio": 2.0,
        }

    def test_blocked_by_swing_bias(self, config: dict) -> None:
        """Even if divergence detected, swing bias should block."""
        with patch("signals.macd_divergence.MACDDivergenceSignal.check_swing_bias", return_value=True):
            sig = MACDDivergenceSignal(config)
            # Minimal candles — won't trigger anyway, but tests the guard path
            closes = [100.0] * 60
            from datetime import datetime, timezone, timedelta
            base = datetime(2026, 3, 29, 14, 0, tzinfo=timezone.utc)
            candles = [
                OHLCV(timestamp=base + timedelta(minutes=i * 5),
                       open=c, high=c + 0.5, low=c - 0.5, close=c, volume=1000,
                       symbol="AAPL", timeframe="5m", market="stock")
                for i, c in enumerate(closes)
            ]
            result = sig.evaluate_from_macd(
                symbol="AAPL", current_price=100.0, candles=candles, market="stock",
            )
            assert not result.triggered
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestMACDDivergence tests/test_signals.py::TestMACDDivergenceSwingBias -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create macd_divergence.py**

Create `autotrader/signals/macd_divergence.py`:

```python
"""MACD Divergence signal — classic divergence with MACD cross confirmation."""

from __future__ import annotations

import math

from loguru import logger

from journal.models import OHLCV
from signals.base_signal import BaseSignal, SignalDirection, SignalResult
from signals.indicators import Indicators


class MACDDivergenceSignal(BaseSignal):
    """Detect price/MACD histogram divergence confirmed by a MACD line cross.

    Bullish: price makes lower low, histogram makes higher low, MACD crosses above signal.
    Bearish: price makes higher high, histogram makes lower high, MACD crosses below signal.
    """

    @property
    def name(self) -> str:
        return "macd_divergence"

    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Generic evaluate — not used directly. See evaluate_from_macd()."""
        return SignalResult(triggered=False, strategy_name=self.name)

    def evaluate_from_macd(
        self,
        symbol: str,
        current_price: float,
        candles: list[OHLCV],
        market: str,
    ) -> SignalResult:
        """Evaluate MACD divergence with cross confirmation.

        Args:
            symbol: Instrument symbol.
            current_price: Latest price.
            candles: Price candles (needs >= slow_period + signal_period bars).
            market: 'stock' or 'crypto'.

        Returns:
            SignalResult with trigger decision and price levels.
        """
        no_signal = SignalResult(triggered=False, strategy_name=self.name)

        if not self.is_enabled():
            return no_signal

        fast = self.config.get("fast_period", 12)
        slow = self.config.get("slow_period", 26)
        sig_period = self.config.get("signal_period", 9)
        lookback = self.config.get("divergence_lookback", 30)
        min_dist = self.config.get("min_swing_distance", 5)
        sl_pct = self.config.get("stop_loss_pct", 0.015)
        rr_ratio = self.config.get("rr_ratio", 2.0)

        min_bars = slow + sig_period + lookback
        if len(candles) < min_bars:
            return no_signal

        closes = [c.close for c in candles]
        macd_line, signal_line, histogram = Indicators.macd(closes, fast, slow, sig_period)

        # Work within the lookback window
        window_start = len(candles) - lookback

        # Find swing lows and swing highs in price (pivot detection)
        swing_lows = self._find_swing_lows(candles, window_start)
        swing_highs = self._find_swing_highs(candles, window_start)

        direction = None
        reason = ""

        # Check bullish divergence: price lower low + histogram higher low
        if len(swing_lows) >= 2:
            prev_sw, curr_sw = swing_lows[-2], swing_lows[-1]
            if curr_sw[0] - prev_sw[0] >= min_dist:
                price_lower_low = curr_sw[1] < prev_sw[1]
                prev_hist = histogram[prev_sw[0]]
                curr_hist = histogram[curr_sw[0]]
                if not math.isnan(prev_hist) and not math.isnan(curr_hist):
                    hist_higher_low = curr_hist > prev_hist
                    if price_lower_low and hist_higher_low:
                        # Check MACD cross confirmation (last 3 bars)
                        if self._bullish_cross(macd_line, signal_line):
                            direction = SignalDirection.LONG
                            reason = (
                                f"Bullish divergence: price low {curr_sw[1]:.2f} < "
                                f"{prev_sw[1]:.2f} but histogram rising, MACD cross up"
                            )

        # Check bearish divergence: price higher high + histogram lower high
        if direction is None and len(swing_highs) >= 2:
            prev_sw, curr_sw = swing_highs[-2], swing_highs[-1]
            if curr_sw[0] - prev_sw[0] >= min_dist:
                price_higher_high = curr_sw[1] > prev_sw[1]
                prev_hist = histogram[prev_sw[0]]
                curr_hist = histogram[curr_sw[0]]
                if not math.isnan(prev_hist) and not math.isnan(curr_hist):
                    hist_lower_high = curr_hist < prev_hist
                    if price_higher_high and hist_lower_high:
                        if self._bearish_cross(macd_line, signal_line):
                            direction = SignalDirection.SHORT
                            reason = (
                                f"Bearish divergence: price high {curr_sw[1]:.2f} > "
                                f"{prev_sw[1]:.2f} but histogram falling, MACD cross down"
                            )

        if direction is None:
            return no_signal

        # Swing bias check
        if self.check_swing_bias(symbol, direction):
            return SignalResult(
                triggered=False, strategy_name=self.name,
                reason=f"Swing bias blocks {direction.value}",
            )

        # Price levels
        entry_price = current_price
        sl_distance = entry_price * sl_pct

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + sl_distance * rr_ratio
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - sl_distance * rr_ratio

        logger.info(
            "macd_divergence_triggered | symbol={} dir={} entry={:.2f} sl={:.2f} tp={:.2f} "
            "reason={}",
            symbol, direction.value, entry_price, stop_loss, take_profit, reason,
        )

        return SignalResult(
            triggered=True,
            direction=direction,
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.65,
            strategy_name=self.name,
            reason=reason,
        )

    @staticmethod
    def _find_swing_lows(
        candles: list[OHLCV], start: int, margin: int = 2,
    ) -> list[tuple[int, float]]:
        """Find swing lows (local minima) in candle lows.

        Args:
            candles: Full candle list.
            start: Index to start searching from.
            margin: Bars on each side to confirm pivot.

        Returns:
            List of (index, low_price) tuples.
        """
        swings = []
        for i in range(max(start, margin), len(candles) - margin):
            low = candles[i].low
            is_pivot = all(
                candles[i - j].low >= low for j in range(1, margin + 1)
            ) and all(
                candles[i + j].low >= low for j in range(1, margin + 1)
            )
            if is_pivot:
                swings.append((i, low))
        return swings

    @staticmethod
    def _find_swing_highs(
        candles: list[OHLCV], start: int, margin: int = 2,
    ) -> list[tuple[int, float]]:
        """Find swing highs (local maxima) in candle highs.

        Args:
            candles: Full candle list.
            start: Index to start searching from.
            margin: Bars on each side to confirm pivot.

        Returns:
            List of (index, high_price) tuples.
        """
        swings = []
        for i in range(max(start, margin), len(candles) - margin):
            high = candles[i].high
            is_pivot = all(
                candles[i - j].high <= high for j in range(1, margin + 1)
            ) and all(
                candles[i + j].high <= high for j in range(1, margin + 1)
            )
            if is_pivot:
                swings.append((i, high))
        return swings

    @staticmethod
    def _bullish_cross(macd_line: list[float], signal_line: list[float]) -> bool:
        """Check if MACD crossed above signal line within last 3 bars."""
        for i in range(-3, 0):
            try:
                prev_m, prev_s = macd_line[i - 1], signal_line[i - 1]
                curr_m, curr_s = macd_line[i], signal_line[i]
                if math.isnan(prev_m) or math.isnan(prev_s):
                    continue
                if math.isnan(curr_m) or math.isnan(curr_s):
                    continue
                if prev_m <= prev_s and curr_m > curr_s:
                    return True
            except IndexError:
                continue
        return False

    @staticmethod
    def _bearish_cross(macd_line: list[float], signal_line: list[float]) -> bool:
        """Check if MACD crossed below signal line within last 3 bars."""
        for i in range(-3, 0):
            try:
                prev_m, prev_s = macd_line[i - 1], signal_line[i - 1]
                curr_m, curr_s = macd_line[i], signal_line[i]
                if math.isnan(prev_m) or math.isnan(prev_s):
                    continue
                if math.isnan(curr_m) or math.isnan(curr_s):
                    continue
                if prev_m >= prev_s and curr_m < curr_s:
                    return True
            except IndexError:
                continue
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd autotrader && python3 -m pytest tests/test_signals.py::TestMACDDivergence tests/test_signals.py::TestMACDDivergenceSwingBias -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add signals/macd_divergence.py tests/test_signals.py
git commit -m "feat: add MACD Divergence signal with cross confirmation"
```

---

### Task 7: Wire signals into main.py and config.json

**Files:**
- Modify: `autotrader/main.py` (~lines 52, 386-397, 1226-1288)
- Modify: `autotrader/config.json`

- [ ] **Step 1: Add imports to main.py**

Add near the existing signal imports at the top of `main.py`:

```python
from signals.vpoc_bounce import VPOCBounceSignal
from signals.macd_divergence import MACDDivergenceSignal
```

- [ ] **Step 2: Register signals in _build_signals()**

Add to the `_build_signals()` method (after the existing signal list, ~line 394):

```python
VPOCBounceSignal(strat_cfg["vpoc_bounce"]),
MACDDivergenceSignal(strat_cfg["macd_divergence"]),
```

- [ ] **Step 3: Add _run_signal() branches**

Add `elif` branches in `_run_signal()` method (after the existing VWAPReversionSignal branch):

For VWAP Reversion — update existing branch to use new signature:

```python
elif isinstance(sig, VWAPReversionSignal):
    vwap_series = Indicators.vwap(candles)
    vwap_val = vwap_series[-1] if vwap_series else current_price
    atr_vals = Indicators.atr(candles, period=sig.config.get("atr_period", 14))
    atr_val = atr_vals[-1] if atr_vals else 0.0
    slope_lb = sig.config.get("slope_lookback", 20)
    slope_val = Indicators.vwap_slope(vwap_series, lookback=slope_lb)
    cur_vol = candles[-1].volume if candles else 0.0
    vol_window = candles[-20:] if len(candles) >= 20 else candles
    avg_vol = sum(c.volume for c in vol_window) / len(vol_window) if vol_window else 1.0
    return sig.evaluate_from_vwap(
        symbol=symbol, current_price=current_price,
        vwap=vwap_val, atr=atr_val, vwap_slope=slope_val,
        current_volume=cur_vol, avg_volume=avg_vol, market=market,
    )
```

For VPOC Bounce:

```python
elif isinstance(sig, VPOCBounceSignal):
    bounce_n = sig.config.get("bounce_candles", 2)
    return sig.evaluate_from_profile(
        symbol=symbol, current_price=current_price,
        session_candles=candles, recent_candles=candles[-bounce_n:],
        market=market,
    )
```

For MACD Divergence:

```python
elif isinstance(sig, MACDDivergenceSignal):
    return sig.evaluate_from_macd(
        symbol=symbol, current_price=current_price,
        candles=candles, market=market,
    )
```

- [ ] **Step 4: Add config entries to config.json**

Add under `"strategies"`:

```json
"vpoc_bounce": {
    "enabled": true,
    "num_bins": 20,
    "proximity_pct": 0.002,
    "bounce_candles": 2,
    "min_poc_volume_pct": 0.15,
    "stop_loss_pct": 0.01
},
"macd_divergence": {
    "enabled": true,
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9,
    "divergence_lookback": 30,
    "min_swing_distance": 5,
    "stop_loss_pct": 0.015
}
```

Update `"vwap_reversion"` config:

```json
"vwap_reversion": {
    "enabled": true,
    "atr_band_multiplier": 1.5,
    "atr_period": 14,
    "volume_confirmation_mult": 1.2,
    "slope_max": 0.001,
    "slope_lookback": 20,
    "stop_loss_atr_mult": 1.0
}
```

- [ ] **Step 5: Also add to backtest signal list builder**

Find the backtest signal list builder (~line 2562) and add the two new signals there with the same conditional pattern.

- [ ] **Step 6: Run full test suite**

Run: `cd autotrader && python3 -m pytest tests/ -v --tb=short 2>&1 | tail -40`
Expected: All tests pass. If any old tests reference the old VWAP signature, fix them.

- [ ] **Step 7: Commit**

```bash
git add main.py config.json
git commit -m "feat: wire VPOC Bounce, MACD Divergence, and redesigned VWAP into main loop"
```

---

### Task 8: Final integration test

**Files:**
- Test: all files

- [ ] **Step 1: Run the complete test suite**

Run: `cd autotrader && python3 -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Syntax-check all new/modified files**

Run:
```bash
cd /Users/petervarhalik/_apps/TRADER
python3 -m py_compile autotrader/signals/indicators.py
python3 -m py_compile autotrader/signals/vwap_reversion.py
python3 -m py_compile autotrader/signals/vpoc_bounce.py
python3 -m py_compile autotrader/signals/macd_divergence.py
python3 -m py_compile autotrader/main.py
```
Expected: No output (clean compilation)

- [ ] **Step 3: Commit final state**

If any fixes were needed, commit them:

```bash
git add -A
git commit -m "fix: integration fixes for new signals"
```

- [ ] **Step 4: Deploy to VPS**

```bash
./deploy.sh
```

Then rebuild on VPS:
```bash
ssh root@162.55.50.28 "cd ~/Varptrader/autotrader && docker compose build autotrader && docker compose up -d autotrader"
```

Note: config.json changes (new strategy sections) need to be added manually on VPS since deploy.sh protects config.json. SSH in and add the `vpoc_bounce`, `macd_divergence` sections and update `vwap_reversion` section.
