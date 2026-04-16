# VarpTrader Overseer — Hard Rules

## NEVER edit these files (read-only):
- execution/paper_engine.py
- execution/alpaca_executor.py
- execution/crypto_executor.py
- risk/kill_switch.py
- risk/position_sizer.py
- overseer/run_overseer.py (no self-modification)

## Config change bounds:
- position_size_pct: 0.002 – 0.01
- stop_loss_pct: 0.010 – 0.030
- rsi_oversold: 20 – 35
- rsi_overbought: 65 – 80
- trailing_stop_pct: 0.005 – 0.02

## Required before any bot restart:
- pytest tests/ must pass 100%
- Write report to overseer/reports/
- Log every file changed with reason

## Never:
- Disable the kill switch
- Change daily loss limit
- Modify position sizing logic
- Place or cancel real orders

---

## Performance Analysis Findings (2026-04-16)
**Source:** analysis/trade_analysis_2026-04-16.md  
**Period:** 2026-03-25 to 2026-04-16 | 130 trades | Net PnL: -$3,953

### Winning combinations:
- vwap_reversion + AAPL: 100% win rate, +$933 (best edge in the system)
- ema_cross + SOL/USDT: 100% win rate, +$362

### Dead weight — disable or avoid:
- TSLA: 0% win rate, -$1,155 — disable
- QQQ: 0% win rate, -$240 — disable
- NVDA: 0% win rate, -$397 — disable
- rsi_momentum strategy: -$1,395 total, no profitable symbol combination — disable
- vwap_reversion on TSLA: 0%, -$1,150 — disable

### Timing patterns:
- Thursday + Friday: 10-12.5% win rate, -$2,349 combined — avoid
- Hours 22:00-08:00 UTC: consistent 0% win rate — avoid
- Best hours: 09, 12, 15, 16, 18, 21 UTC

### Direction:
- Shorts outperform longs: -$1,382 vs -$2,571

### Overseer guidance:
- Prioritize conditions that replicate vwap_reversion on AAPL
- When adjusting strategy weights, favour ema_cross and vwap_reversion
- Do NOT increase position size on rsi_momentum, bollinger_fade, ema_pullback
- Consider adding day-of-week and hour-of-day filters before next config cycle

---

## Changes applied 2026-04-16 (post-analysis)

### Strategy code
- `signals/vwap_reversion.py`: TP fixed at VWAP (was overshoot formula that shrank with VWAP drift).
- `main.py`: VWAP reversion restricted to `market == "stock"` at dispatch — crypto runs blocked. Edge concentrated in AAPL; crypto had no supporting data.
- `data/feed_crypto.py`: funding-rate / OI symbol mapping normalised. `BTC/USDC` (and other USDC spot pairs) now correctly map to `BTC/USDT:USDT` perps. Previously produced `BTC/USDC:USDT` which Binance does not list.

### Per-combo auto-disable (main.py `_combo_allows`)
- After `COMBO_DISABLE_MIN_TRADES` (default 15) completed trades for a `(symbol, strategy)` combo, if win-rate ≤ 30% **and** total PnL < 0, that combo is muted.
- Gate applied after signal triggers, before execution. Preserves the watchlist while cutting dead combos.

### Per-symbol size multipliers (`symbol_size_multipliers` in config)
- New generic scaling layer on top of win-rate sizing. Currently `MSFT: 0.5`, `GOOGL: 0.5` — new additions trade at half size until they prove an edge.
- Applied after win-rate multiplier: `effective_mult = winrate_mult * symbol_mult`.

### Watchlist
- Added `MSFT`, `GOOGL` to `watchlist.stocks`. Sampled at 0.5× size per above.

---

## Pre-restart workflow (durable rule)

After any meaningful change:
1. Run `docker exec autotrader pytest tests/` (docker image may be stale — copy host files with `docker cp main.py signals/ data/` first if needed).
2. All tests must pass before restart.
3. Update this CLAUDE.md with the what + why of the change.
4. Log the file list and reason in a commit or overseer report.
