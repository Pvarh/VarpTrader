# New Trading Signals Design Spec

**Date:** 2026-03-29
**Scope:** Three signal changes — VPOC Bounce (new), MACD Divergence (new), VWAP Reversion (full rewrite)

---

## 1. New Indicators (signals/indicators.py)

Three new static methods on the existing `Indicators` class.

### 1.1 MACD

```
macd(closes, fast=12, slow=26, signal=9) → (macd_line[], signal_line[], histogram[])
```

- `macd_line = EMA(fast) - EMA(slow)`
- `signal_line = EMA(macd_line, signal)`
- `histogram = macd_line - signal_line`
- First `slow + signal - 2` values are NaN
- Reuses existing `Indicators.ema()` internally

### 1.2 Volume Profile

```
volume_profile(candles, num_bins=20) → (poc_price, value_area_high, value_area_low)
```

- Divides session price range into `num_bins` equal-width bins
- Accumulates volume per bin using typical price `(H+L+C)/3`
- POC = midpoint of the highest-volume bin
- Value area = bins containing 70% of total volume, expanding outward from POC
- Returns a single snapshot (not a time series) — called once per scan cycle

### 1.3 VWAP Slope

```
vwap_slope(vwap_values, lookback=20) → float
```

- Linear regression slope of the last `lookback` VWAP values
- Normalized by current VWAP level: `slope / current_vwap`
- Returns dimensionless number: near 0 = flat, large positive/negative = trending
- Returns 0.0 if insufficient data

---

## 2. VWAP Reversion Redesign (signals/vwap_reversion.py)

Full rewrite of existing file. Same config key `vwap_reversion`. All old logic replaced.

### Config Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `atr_band_multiplier` | 1.5 | Band width: VWAP +/- (ATR * multiplier) |
| `atr_period` | 14 | ATR lookback period |
| `volume_confirmation_mult` | 1.2 | Current volume must exceed 1.2x 20-bar average |
| `slope_max` | 0.001 | Max abs(normalized VWAP slope) — above = trending, don't fade |
| `slope_lookback` | 20 | Bars for slope calculation |
| `stop_loss_atr_mult` | 1.0 | SL distance beyond band: ATR * this multiplier |
| `rr_ratio` | 2.0 | Risk-reward ratio for take-profit |

### Entry Logic

1. Compute ATR-based dynamic bands: `upper = vwap + (atr * atr_band_multiplier)`, `lower = vwap - (atr * atr_band_multiplier)`
2. Check VWAP slope is flat: `abs(vwap_slope) < slope_max`
3. Check volume: current bar volume > `volume_confirmation_mult` * 20-bar average volume
4. **LONG:** price <= lower band AND slope flat AND volume confirmed
5. **SHORT:** price >= upper band AND slope flat AND volume confirmed
6. TP targets VWAP itself (natural mean-reversion target)
7. SL placed beyond the band by `stop_loss_atr_mult * ATR`

### Removed from Old Version

- Fixed % deviation bands → replaced by ATR-dynamic bands
- Session window restriction (180 min / 60 min) → slope filter handles this
- Momentum candle check (3 consecutive) → replaced by volume confirmation

### Specialized Method Signature

```python
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
```

**Confidence:** 0.60
**Markets:** Stocks and crypto

---

## 3. VPOC Bounce Signal (signals/vpoc_bounce.py)

New file. New config key `vpoc_bounce`.

### Config Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_bins` | 20 | Price bins for volume profile |
| `proximity_pct` | 0.002 | Price must be within 0.2% of POC |
| `bounce_candles` | 2 | Consecutive candles showing rejection |
| `min_poc_volume_pct` | 0.15 | POC bin must hold >= 15% of session volume |
| `stop_loss_pct` | 0.01 | SL beyond POC |
| `rr_ratio` | 2.0 | Risk-reward ratio |

### Entry Logic

1. Build volume profile from session candles using `Indicators.volume_profile()`
2. Validate POC strength: POC bin volume >= `min_poc_volume_pct` * total session volume
3. Check proximity: `abs(current_price - poc_price) / poc_price < proximity_pct`
4. Detect bounce direction from last `bounce_candles` candles:
   - **LONG:** price approached POC from below, last N candles are bullish (close > open) — buyers defending
   - **SHORT:** price approached POC from above, last N candles are bearish (close < open) — sellers defending
5. SL on the other side of POC (through level = level failed)
6. TP at `rr_ratio * risk_distance` from entry

### Specialized Method Signature

```python
def evaluate_from_profile(
    self,
    symbol: str,
    current_price: float,
    session_candles: list[OHLCV],
    recent_candles: list[OHLCV],
    market: str,
) -> SignalResult:
```

**Confidence:** 0.60
**Markets:** Stocks and crypto (volume profile resets daily)

---

## 4. MACD Divergence Signal (signals/macd_divergence.py)

New file. New config key `macd_divergence`.

### Config Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `fast_period` | 12 | MACD fast EMA |
| `slow_period` | 26 | MACD slow EMA |
| `signal_period` | 9 | MACD signal line EMA |
| `divergence_lookback` | 30 | Bars to search for swing points |
| `min_swing_distance` | 5 | Minimum bars between two swing points |
| `stop_loss_pct` | 0.015 | SL as fraction of entry |
| `rr_ratio` | 2.0 | Risk-reward ratio |

### Entry Logic

1. Compute MACD line, signal line, histogram from `Indicators.macd()`
2. Find swing lows/highs in price within last `divergence_lookback` bars:
   - Swing low: bar's low is lower than 2 bars on each side
   - Swing high: bar's high is higher than 2 bars on each side
3. Take the two most recent swing points that are >= `min_swing_distance` bars apart
4. Compare price swing points to histogram values at the same bar indices:
   - **Bullish divergence:** price lower low + histogram higher low
   - **Bearish divergence:** price higher high + histogram lower high
5. **Cross confirmation gate:** divergence triggers only when MACD line crosses signal line in the expected direction within the last 3 bars:
   - Bullish: MACD crosses above signal line
   - Bearish: MACD crosses below signal line
6. Entry at current price, SL at `stop_loss_pct` beyond entry, TP at `rr_ratio * risk`

### Swing Detection

Simple pivot method: a bar is a swing low if its low < the low of the 2 bars before and 2 bars after. Same logic for swing highs with highs. Only the two most recent qualifying points are used, filtered by `min_swing_distance`.

### Specialized Method Signature

```python
def evaluate_from_macd(
    self,
    symbol: str,
    current_price: float,
    candles: list[OHLCV],
    market: str,
) -> SignalResult:
```

**Confidence:** 0.65 (highest — divergence + cross is double confirmation)
**Markets:** Stocks and crypto

---

## 5. Integration (main.py)

### Signal Registration

Add to the signal registration block (~line 2562):

```python
if strat_cfg["vpoc_bounce"]["enabled"]:
    signals_list.append(VPOCBounceSignal(strat_cfg["vpoc_bounce"]))
if strat_cfg["macd_divergence"]["enabled"]:
    signals_list.append(MACDDivergenceSignal(strat_cfg["macd_divergence"]))
```

VWAP reversion already registered — only its internal logic changes.

### _run_signal() Branches

Add specialized evaluation branches:

- **vpoc_bounce:** pass session candles + recent candles + current price → signal computes volume profile internally
- **macd_divergence:** pass candles + current price → signal computes MACD and detects divergence internally
- **vwap_reversion:** update existing branch to pass ATR, VWAP, slope, and volume data (new method signature)

### config.json Additions

New entries under `"strategies"`:

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

VWAP reversion config updated in-place with new parameter names (old params removed):

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

---

## 6. Tests

### New Files

- `tests/test_vpoc_bounce.py` — enabled/disabled guard, trigger long/short, no-trigger when POC weak, no-trigger when price not near POC
- `tests/test_macd_divergence.py` — enabled/disabled guard, bullish divergence + cross, bearish divergence + cross, no-trigger without cross confirmation, no-trigger without divergence

### Updated Files

- `tests/test_vwap_reversion.py` — rewrite for new logic: ATR bands, slope filter, volume confirmation, trigger/no-trigger cases

### Test Pattern

Each test file follows the existing pattern:
- Construct config dict with test parameters
- Instantiate signal with config
- Call specialized evaluate method with crafted data
- Assert `triggered`, `direction`, `stop_loss`, `take_profit`, `confidence`, `strategy_name`

---

## 7. Files Changed Summary

| File | Action |
|------|--------|
| `signals/indicators.py` | Add `macd()`, `volume_profile()`, `vwap_slope()` |
| `signals/vwap_reversion.py` | Full rewrite |
| `signals/vpoc_bounce.py` | New file |
| `signals/macd_divergence.py` | New file |
| `main.py` | Add imports, registration, `_run_signal()` branches |
| `config.json` | Add vpoc_bounce + macd_divergence sections, update vwap_reversion section |
| `tests/test_vpoc_bounce.py` | New file |
| `tests/test_macd_divergence.py` | New file |
| `tests/test_vwap_reversion.py` | Rewrite for new logic |

No changes to read-only files (paper_engine, alpaca_executor, crypto_executor, kill_switch, position_sizer, run_overseer).

---

## 8. Existing Gate System (unchanged)

All three signals pass through the existing pipeline with zero modifications:

1. VWAP 1h directional filter
2. Regime filter (trend vs ranging)
3. Session bias gate
4. Whale detector gate
5. Strategy cooldown (3 consecutive losses → 60 min pause)
6. Strategy auto-disable (3 cooldown triggers → disabled)
7. Duplicate signal suppression
8. Correlation limits (max 2 same-direction per group)
9. Dynamic win-rate position sizing (0.5x–1.5x)
10. Reward ratio gate (min 2.0)
11. Partial profit taking (50% at 50% TP progress)
12. 3-tier trailing stop
