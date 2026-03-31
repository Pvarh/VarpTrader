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
