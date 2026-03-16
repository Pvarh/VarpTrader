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
