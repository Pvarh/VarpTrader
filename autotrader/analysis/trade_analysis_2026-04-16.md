# Autotrader Performance Analysis
**Period:** 2026-03-25 → 2026-04-16  
**Generated:** 2026-04-16  
**Total trades:** 130 (129 closed)  
**Net PnL:** -$3,953.04  

---

## Strategy Performance

| Strategy | Trades | Wins | Losses | Win Rate | Total PnL | Avg PnL |
|---|---|---|---|---|---|---|
| ema_cross | 11 | 6 | 5 | 54.5% | **+$378.80** | +$34.44 |
| funding_rate | 6 | 2 | 4 | 33.3% | -$81.98 | -$13.66 |
| bollinger_fade | 10 | 2 | 8 | 20.0% | -$239.38 | -$23.94 |
| vpoc_bounce | 10 | 2 | 8 | 20.0% | -$301.82 | -$30.18 |
| ema_pullback | 11 | 1 | 10 | 9.1% | -$328.85 | -$29.90 |
| macd_divergence | 29 | 9 | 20 | 31.0% | -$454.88 | -$15.69 |
| rsi_momentum | 13 | 4 | 9 | 30.8% | -$1,395.20 | -$107.32 |
| vwap_reversion | 39 | 14 | 25 | 35.9% | -$1,529.72 | -$39.22 |

**Finding:** Only `ema_cross` is profitable. `rsi_momentum` and `vwap_reversion` are the biggest losers.

---

## Symbol Performance

| Symbol | Trades | Win Rate | Total PnL | Avg PnL |
|---|---|---|---|---|
| AAPL | 12 | **83.3%** | **+$884.74** | +$73.73 |
| BTC/USDC | 8 | 37.5% | +$208.12 | +$26.01 |
| SOL/USDC | 9 | 44.4% | +$31.71 | +$3.52 |
| ETH/USDC | 11 | 18.2% | -$89.25 | -$8.11 |
| SPY | 2 | 50.0% | -$99.38 | -$49.69 |
| QQQ | 3 | **0.0%** | -$240.43 | -$80.14 |
| NVDA | 5 | **0.0%** | -$397.62 | -$79.52 |
| BTC/USDT | 17 | 29.4% | -$429.26 | -$25.25 |
| SOL/USDT | 33 | 30.3% | -$1,064.29 | -$32.25 |
| TSLA | 6 | **0.0%** | **-$1,155.76** | -$192.63 |
| ETH/USDT | 23 | 21.7% | -$1,601.61 | -$69.64 |

**Finding:** AAPL is the only strong winner (83.3% win rate). TSLA, QQQ, NVDA have 0% win rate.

---

## Strategy × Symbol Combinations (≥3 trades)

| Strategy | Symbol | Trades | Win Rate | PnL |
|---|---|---|---|---|
| vwap_reversion | AAPL | 10 | **100%** | **+$933.12** |
| ema_cross | SOL/USDT | 3 | **100%** | +$362.28 |
| ema_pullback | SOL/USDC | 3 | 33.3% | +$86.67 |
| vwap_reversion | BTC/USDT | 5 | 40.0% | -$21.91 |
| macd_divergence | SOL/USDT | 7 | 57.1% | -$27.79 |
| macd_divergence | ETH/USDC | 4 | 25.0% | -$32.26 |
| macd_divergence | BTC/USDT | 4 | 25.0% | -$45.76 |
| macd_divergence | BTC/USDC | 3 | 33.3% | -$48.50 |
| ema_pullback | ETH/USDC | 3 | 0.0% | -$75.64 |
| funding_rate | SOL/USDT | 6 | 33.3% | -$81.98 |
| ema_pullback | SOL/USDT | 3 | 0.0% | -$97.90 |
| bollinger_fade | SOL/USDT | 3 | 0.0% | -$142.79 |
| vwap_reversion | NVDA | 3 | 0.0% | -$212.46 |
| vwap_reversion | SOL/USDT | 5 | 0.0% | -$246.05 |
| macd_divergence | ETH/USDT | 7 | 14.3% | -$254.67 |
| vwap_reversion | ETH/USDT | 6 | 33.3% | -$363.59 |
| rsi_momentum | ETH/USDT | 3 | 0.0% | -$709.64 |
| rsi_momentum | SOL/USDT | 4 | 25.0% | -$740.42 |
| vwap_reversion | TSLA | 5 | 0.0% | **-$1,149.92** |

**Finding:** `vwap_reversion` on AAPL is the single best combo (100% win, +$933). Same strategy on TSLA is the worst (-$1,150, 0% win).

---

## Direction

| Direction | Trades | Win Rate | Total PnL |
|---|---|---|---|
| short | 69 | 31.9% | -$1,382.08 |
| long | 60 | 30.0% | -$2,570.96 |

**Finding:** Shorts outperform longs significantly.

---

## Day of Week

| Day | Trades | Win Rate | Total PnL |
|---|---|---|---|
| Saturday | 16 | 37.5% | +$39.37 |
| Wednesday | 33 | 45.5% | -$90.63 |
| Sunday | 9 | 33.3% | -$132.87 |
| Monday | 20 | 35.0% | -$590.27 |
| Tuesday | 25 | 24.0% | -$829.57 |
| Friday | 10 | 10.0% | -$996.74 |
| Thursday | 16 | 12.5% | -$1,352.33 |

**Finding:** Thursday and Friday are catastrophic (10-12.5% win rate). Wednesday and Saturday are best.

---

## Hour of Day (UTC)

### Profitable hours:
| Hour | Trades | Win Rate | PnL |
|---|---|---|---|
| 09:00 | 2 | 50% | +$263.91 |
| 15:00 | 8 | 50% | +$178.69 |
| 18:00 | 6 | 33% | +$140.54 |
| 21:00 | 2 | 50% | +$108.35 |
| 16:00 | 3 | 33% | +$74.19 |
| 05:00 | 3 | 67% | +$67.25 |
| 12:00 | 3 | 67% | +$19.39 |

### Losing hours:
| Hour | Trades | Win Rate | PnL |
|---|---|---|---|
| 00:00 | 4 | 0% | -$427.07 |
| 22:00 | 2 | 0% | -$339.84 |
| 04:00 | 3 | 0% | -$246.05 |
| 06:00 | 5 | 0% | -$301.55 |
| 23:00 | 9 | 11% | -$297.71 |
| 01:00 | 5 | 20% | -$578.52 |

**Finding:** Hours 9, 12, 15, 16, 18, 21 UTC are profitable. Midnight to 7am UTC is a consistent drain.

---

## Key Recommendations

1. **Disable TSLA entirely** — 0% win rate, -$1,155 lost. Worst performing symbol.
2. **Disable `rsi_momentum` strategy** — -$1,395 total, no winning symbol combination.
3. **Block trading Thursday & Friday** — 10-12.5% win rate, -$2,349 combined.
4. **Block trading 22:00–08:00 UTC** — consistent losses, multiple 0% win-rate hours.
5. **Concentrate on `vwap_reversion` + AAPL** — only 100% win rate combination (+$933).
6. **Disable QQQ and NVDA** — 0% win rate across all strategies.
7. **Prefer short direction** — 30% better PnL than longs in this period.

---

## Notes
- This is paper trading data — real execution costs, slippage, and liquidity may differ.
- Sample size is small (130 trades). Patterns may shift with more data.
- `vwap_reversion` on AAPL should be investigated to understand what makes it work — can those conditions be applied to other symbols?
