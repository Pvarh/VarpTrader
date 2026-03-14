<p align="center">
  <h1 align="center">VarpTrader</h1>
  <p align="center">
    Automated trading bot for US equities and crypto with AI-powered analysis
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Claude_AI-Anthropic-6B4FBB?logo=anthropic&logoColor=white" alt="Claude AI">
  <img src="https://img.shields.io/badge/Alpaca-Markets-FFCC00?logo=alpaca&logoColor=black" alt="Alpaca">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/SQLite-WAL_Mode-003B57?logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/tests-331_passing-brightgreen" alt="Tests">
</p>

---

## Overview

VarpTrader is a modular, event-driven automated trading system that scans US equities (via Alpaca) and crypto markets (via Binance/CCXT), generates trade signals across 5 independent strategies, monitors whale activity, enforces strict risk management, and uses Claude AI for nightly performance analysis and weekly swing bias advice.

Designed for paper trading first, with a clear path to live execution once strategy win rates meet configurable go-live thresholds.

## Architecture

```
                           VarpTrader Architecture
 ================================================================

  DATA LAYER                    SIGNAL LAYER               RISK LAYER
 +------------------+     +---------------------+     +----------------+
 | Alpaca WebSocket |---->| First Candle (ORB)  |     | Position Sizer |
 |  (real-time)     |     | EMA Cross (50/200)  |---->|  (ATR-based)   |
 | Polygon REST     |---->| VWAP Reversion      |     | Kill Switch    |
 |  (historical)    |     | RSI Momentum        |     |  (-3% daily)   |
 | CCXT/Binance     |---->| Bollinger Fade      |     | Reward Ratio   |
 +------------------+     +---------------------+     +----------------+
         |                          |                        |
         v                          v                        v
 +------------------+     +---------------------+     +----------------+
 | WHALE MONITORING |     |  EXECUTION ENGINE   |     |   JOURNAL DB   |
 | Block Trades     |     | Alpaca (stocks)     |     | SQLite + WAL   |
 | On-chain Alerts  |---->| CCXT (crypto)       |---->| Trade log      |
 | Orderbook Scan   |     | Paper Engine (sim)  |     | Analysis runs  |
 +------------------+     | Position Monitor    |     | Swing bias log |
                          +---------------------+     +----------------+
                                    |                        |
         +--------------------------+------------------------+
         |                          |                        |
         v                          v                        v
 +------------------+     +---------------------+     +----------------+
 | ALERTS           |     |  AI ANALYSIS        |     |  DASHBOARD     |
 | Telegram Bot     |     | Claude Nightly      |     | FastAPI + WS   |
 | Signal alerts    |     | Swing Advisor       |     | Equity curve   |
 | Kill switch      |     | Config Optimizer    |     | Trade journal  |
 | Trade updates    |     | Performance Report  |     | Config viewer  |
 +------------------+     +---------------------+     +----------------+
```

## Features

### Trading Strategies
- **First Candle / ORB** -- 60-minute opening range breakout, valid 10:30-13:00 EST, ATR filter to reject overextended ranges
- **EMA Cross** -- 50/200 exponential moving average crossover for trend following
- **VWAP Reversion** -- Mean reversion on VWAP deviations with configurable threshold
- **RSI Momentum** -- RSI-based momentum with neutral zone filtering (45-55 dead zone)
- **Bollinger Fade** -- Bollinger Band fade entries with RSI confirmation

### Risk Management
- **ATR-based position sizing** -- dynamically sizes positions based on volatility
- **Daily kill switch** -- halts all trading at -3% daily loss (configurable)
- **Minimum reward ratio** -- rejects trades below 2:1 reward-to-risk (configurable)
- **Trade lifecycle state machine** -- SIGNAL > VALIDATED > SUBMITTED > FILLED > MONITORING > CLOSED
- **Trailing stop to breakeven** -- tightens stop at 50% of target reached

### Whale Monitoring
- **Block trade detection** -- Polygon.io API scans for institutional-size trades (>50K shares or >$2M notional)
- **On-chain whale alerts** -- Whale Alert API monitors large crypto wallet movements
- **Orderbook scanning** -- detects large resting orders on crypto exchanges

### AI Analysis
- **Nightly performance review** -- Claude analyzes daily trades, computes win rates, suggests config adjustments
- **Weekly swing advisor** -- Claude assesses directional bias per ticker using news + fundamentals + price action
- **Auto config tuning** -- AI-suggested parameter changes with dashboard approval workflow
- **Go-live thresholds** -- strategies must hit minimum win rates before switching from paper to live

### Data Feeds
- **Alpaca WebSocket** -- real-time stock prices via free IEX feed (primary, zero-latency cache)
- **Polygon REST** -- historical OHLCV for backtesting (optional, yfinance fallback)
- **CCXT/Binance** -- crypto market data and execution
- **WebSocket streaming** -- real-time crypto data via Binance WebSocket

### Backtesting
- **VectorBT** -- vectorized backtesting engine for fast strategy evaluation
- **Custom engine** -- bar-by-bar backtesting with full signal replay (fallback)
- **Strategy optimizer** -- parameter sweep with CSV export
- **Performance report** -- auto-disable strategies failing go-live thresholds

### Dashboard
- **Multi-page web UI** -- dark theme, real-time updates via WebSocket
- **Pages** -- Overview (equity curve, PnL, positions), Trade Journal (filter/paginate), Analysis (win rates, whale correlation, swing bias), Config Viewer (approve AI changes)
- **WebSocket channels** -- `/ws/signals` for live signal events, `/ws/pnl` for PnL streaming
- **API key auth** -- config approval endpoint protected by `X-API-Key` header

### Alerts
- **Telegram notifications** -- signal fired, trade filled, stop/target hit, kill switch triggered, nightly summary

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Broker (Stocks) | Alpaca (`alpaca-py`) |
| Broker (Crypto) | Binance via CCXT |
| Real-time Data | Alpaca WebSocket (IEX) |
| Historical Data | Polygon REST / yfinance |
| AI | Anthropic Claude API |
| Web Framework | FastAPI + Jinja2 + WebSockets |
| Database | SQLite (WAL mode, thread-safe) |
| Backtesting | VectorBT + custom engine |
| Scheduling | APScheduler |
| Logging | Loguru (file rotation, 30-day retention) |
| Alerts | python-telegram-bot |
| Deployment | Docker + docker-compose |
| Testing | pytest (331 tests) |

## Project Structure

```
autotrader/
+-- main.py                    # Entry point, CLI, scheduler
+-- config.json                # All strategy/risk/whale parameters
+-- signals/                   # 5 strategy modules + indicators
|   +-- first_candle.py        #   60-min ORB breakout
|   +-- ema_cross.py           #   EMA 50/200 crossover
|   +-- vwap_reversion.py      #   VWAP mean reversion
|   +-- rsi_momentum.py        #   RSI with neutral zone
|   +-- bollinger_fade.py      #   Bollinger Band fade
|   +-- indicators.py          #   Pure numpy: EMA, RSI, ATR, VWAP, BB
|   +-- base_signal.py         #   Abstract base + swing bias check
+-- data/                      # Market data feeds
|   +-- feed_stocks.py         #   Alpaca WS (live) + Polygon (historical)
|   +-- feed_crypto.py         #   CCXT + yfinance
|   +-- stream_stocks.py       #   Alpaca WebSocket client
|   +-- stream_crypto.py       #   Binance WebSocket client
|   +-- normalizer.py          #   Unified OHLCV normalizer
|   +-- event_bus.py           #   Pub/sub event system
+-- execution/                 # Order execution + monitoring
|   +-- alpaca_executor.py     #   Alpaca REST orders
|   +-- crypto_executor.py     #   CCXT orders
|   +-- paper_engine.py        #   Simulated fills + slippage
|   +-- position_monitor.py    #   SL/TP checker (30s interval)
|   +-- trade_lifecycle.py     #   State machine
|   +-- order_validator.py     #   Pre-flight checks
+-- risk/                      # Risk management
|   +-- position_sizer.py      #   ATR-based sizing
|   +-- kill_switch.py         #   Daily loss circuit breaker
|   +-- reward_ratio.py        #   Min R:R gate
+-- whale/                     # Institutional activity detection
|   +-- block_trades.py        #   Polygon block trade scanner
|   +-- onchain.py             #   Whale Alert API
|   +-- orderbook.py           #   Large order detection
+-- analysis/                  # AI + performance analysis
|   +-- analyzer.py            #   Win rates, PnL stats
|   +-- swing_advisor.py       #   Weekly Claude bias advisor
|   +-- llm_advisor.py         #   Nightly Claude review
|   +-- config_updater.py      #   Atomic config writes
|   +-- report_builder.py      #   Markdown report generation
+-- journal/                   # Trade database
|   +-- db.py                  #   SQLite with WAL, parameterized queries
|   +-- models.py              #   Trade, AnalysisRun, OHLCV, SwingBias
+-- dashboard/                 # Web UI
|   +-- router.py              #   FastAPI routes + WebSocket endpoints
|   +-- templates/             #   Jinja2 (base, dashboard, trades, analysis, config)
|   +-- static/style.css       #   Dark theme CSS
+-- backtesting/               # VectorBT backtesting
|   +-- vectorbt_runner.py     #   5 strategy runners
|   +-- optimizer.py           #   Parameter sweep
|   +-- performance_report.py  #   Go-live threshold check
+-- backtest/                  # Custom backtesting (fallback)
|   +-- engine.py              #   Bar-by-bar replay
|   +-- data_loader.py         #   CSV/API data loading
+-- alerts/
|   +-- telegram_bot.py        #   Telegram notifications
+-- tests/                     # 331 tests across 13 files
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- .env.example
```

## Setup

### Prerequisites

- Python 3.11+
- [Alpaca](https://alpaca.markets/) account (free, paper trading)
- Optional: [Polygon.io](https://polygon.io/) API key (for historical data, yfinance used as fallback)
- Optional: [Anthropic](https://console.anthropic.com/) API key (for AI analysis)
- Optional: [Telegram Bot](https://core.telegram.org/bots) (for alerts)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/VarpTrader.git
cd VarpTrader/autotrader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys (at minimum: ALPACA_API_KEY + ALPACA_SECRET_KEY)

# Run tests
pytest tests/ -q

# Start paper trading
python main.py trade
```

### Docker Deployment

```bash
cd VarpTrader/autotrader

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Build and run
docker compose up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### CLI Commands

```bash
python main.py trade                          # Start live/paper trading
python main.py backtest -s AAPL --start 2025-01-01 --end 2025-12-31
python main.py analyze                        # Run nightly analysis now
python main.py optimize -s AAPL               # Run parameter optimization
python main.py bias                           # Run swing advisor now
python main.py dashboard                      # Start dashboard only
```

## Dashboard

<!-- Replace with actual screenshot -->
<p align="center">
  <img src="docs/dashboard-screenshot.png" alt="VarpTrader Dashboard" width="800">
  <br>
  <em>Real-time dashboard with equity curve, open positions, and PnL tracking</em>
</p>

> **Note:** To add a screenshot, take a capture of the dashboard at `http://localhost:8000` and save it as `docs/dashboard-screenshot.png`.

## Configuration

All parameters are in `config.json` and can be tuned via the AI analysis engine or the dashboard:

| Category | Key Parameters |
|----------|---------------|
| Strategies | `orb_window_minutes`, `fast_ema`/`slow_ema`, `vwap_deviation_pct`, `rsi_oversold`/`rsi_overbought`, `bb_period`/`bb_std` |
| Risk | `position_size_pct` (1%), `stop_loss_pct` (1.5%), `daily_loss_limit_pct` (3%), `min_reward_ratio` (2.0) |
| Go-Live | `orb_min_win_rate` (56%), `mean_reversion_min_win_rate` (55%), `trend_following_min_win_rate` (40%) |
| Whale | `stock_block_shares` (50K), `stock_block_usd` ($2M), `crypto_transfer_usd` ($1M) |

## Disclaimer

> **This software is for educational and research purposes only.**
>
> VarpTrader is not financial advice. Trading stocks and cryptocurrencies involves substantial risk of loss. Past performance of any trading strategy is not indicative of future results. You are solely responsible for evaluating the risks and merits of using this software with real money.
>
> The authors and contributors assume no liability for any financial losses incurred through the use of this software. Always start with paper trading and thoroughly test any strategy before considering live deployment.

## License

MIT License -- see [LICENSE](LICENSE) for details.

Copyright (c) 2026 Peter Varhalik
