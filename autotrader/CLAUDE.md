# VarpTrader — Hard Rules

## Architecture
- **VPS** (root@162.55.50.28:~/Varptrader) is the **source of truth** for config, data, and runtime state
- **Docker container** runs the trading bot; config.json is bind-mounted from host
- **Host-side scripts** (overseer, nightly analysis) run outside Docker on VPS
- Deploy with `./deploy.sh` from repo root — never raw scp/rsync (protects config.json and runtime data)
- After deploy: `ssh root@162.55.50.28 "cd ~/Varptrader/autotrader && docker compose build autotrader && docker compose up -d autotrader"`

## NEVER edit these files (read-only):
- `execution/paper_engine.py`
- `execution/alpaca_executor.py`
- `execution/crypto_executor.py`
- `risk/kill_switch.py`
- `risk/position_sizer.py`
- `overseer/run_overseer.py` (no self-modification)

## Config change bounds:
| Parameter | Min | Max |
|-----------|-----|-----|
| position_size_pct | 0.002 | 0.01 |
| stop_loss_pct | 0.010 | 0.030 |
| rsi_oversold | 20 | 35 |
| rsi_overbought | 65 | 80 |
| trailing_stop_pct | 0.005 | 0.02 |

## Config ownership:
- **Only the overseer** writes to config.json (runs 04:00 UTC via host-side Claude CLI)
- **Nightly analysis** (03:30 UTC, inside Docker) is recommendation-only — must never call `apply_changes()`
- Never overwrite VPS config.json during deploys — `deploy.sh` handles this

## Required before any bot restart:
- `pytest tests/` must pass 100%
- Write report to `overseer/reports/`
- Log every file changed with reason

## Never:
- Disable the kill switch
- Change daily loss limit
- Modify position sizing logic (position_sizer.py is read-only)
- Place or cancel real orders
- Deploy config.json, data/, logs/, .env, or overseer state files to VPS

## Active safety features (deployed 2026-03-29):
- **Strategy cooldown**: pauses strategy for 60 min after 3 consecutive losses
- **Strategy auto-disable**: disables strategy in config.json after 3 cooldown triggers
- **Dynamic position sizing**: scales 0.5x–1.5x based on recent win rate (last 10 trades)
- **Partial profit taking**: closes 50% at 50% TP progress, remainder rides trailing stop
- **3-tier trailing stop**: breakeven at 50% TP, trail 50% at 75%, trail 75% at 100%+

## Key directories:
- `signals/` — 12+ signal generators (bollinger, macd, rsi, vwap, etc.)
- `execution/` — paper engine, position monitor, trade lifecycle, broker integrations
- `risk/` — kill switch, position sizer, reward ratio
- `journal/` — SQLite trade database (WAL mode, `PRAGMA synchronous=FULL`)
- `analysis/` — market analysis, LLM advisor, config updater
- `dashboard/` — Flask web dashboard with WebSocket live updates
- `overseer/` — automated system manager (host-side, runs outside Docker)
- `tests/` — 25 test files, run with `pytest tests/ -q`
## Audit 2026-05-06 (post-3-week period)

Audit window: 2026-04-16 → 2026-05-05. 114 closed trades, net **+$1,954**, but **96% of profit is one day** (2026-05-05: 29 trades, 96.6% WR, +$1,876). Excluding that day: 85 trades, 54.1% WR, +$79 — effectively breakeven. The vwap_reversion edge is regime-dependent (strong-trending-up days), not a stable signal.

### Crypto: net loser in every regime slice
| regime × direction | n | WR | PnL |
|---|---|---|---|
| crypto + trending_up + long | 7 | 28.6% | -$63 |
| crypto + trending_down + short | 14 | 42.9% | -$45 |
| crypto + ranging + short | 3 | 33.3% | -$38 |

### Change applied: tighter crypto fast-gate (`main.py` `_combo_allows`)
- Fast gate now mutes a `(symbol, strategy)` combo at **5 trades** for crypto (`"/" in symbol`) at ≤25% WR with negative PnL. Stocks stay at 8.
- Reason: every crypto regime slice is net-negative on small samples; combo gate at 8 lets bleeders run too long. A regime-specific block was rejected (n=7 too thin for an overfit rule).

### Telemetry reality
- `partial_notes` migration is working — last `market_condition` corruption was 2026-04-16 18:57 (pre-restart). Clean since then.
- 13 of 114 trades since 2026-04-16 still have NULL/empty `market_condition`; 21 have NULL `swing_bias`. Mostly in crypto slices where regime tagging didn't fire.

### Repo hygiene
- Stale nested `autotrader/autotrader/` dir (5 duplicate files from 2026-03-28) deleted.
- Local fork (4 commits past 5bb1f61) reconciled with origin/main via cherry-pick. Container is now running origin's bug fixes (incl. `check_swing_bias` inversion fix in `vpoc_bounce` / `vwap_reversion` / `macd_divergence`) **plus** the local feature set (per-combo gate, per-combo cooldown, USDC perp mapping, VWAP TP fix, telegram heartbeat redesign).

### Not changed (and why)
- `vwap_reversion` left enabled. Single-day profit isn't proof of edge, but isn't proof against either. Existing combo gate will mute it per-symbol if it stops working. **Do NOT scale up sizing on the May-5 result.**

