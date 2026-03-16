#!/usr/bin/env python3
"""Polygon.io backtest runner for ema_cross and bollinger_fade.

Fetches 2 years of daily EOD data from Polygon.io for NVDA, AAPL, TSLA,
SPY, and QQQ, then runs the BacktestEngine with both strategies and prints
a summary table.

Usage::

    # Daily bars (free Polygon tier)
    python scripts/run_polygon_backtest.py

    # With explicit API key and custom date range
    python scripts/run_polygon_backtest.py \\
        --api-key YOUR_KEY \\
        --start 2023-01-01 \\
        --end   2024-12-31 \\
        --timeframe 1d

    # Intraday (requires paid Polygon plan)
    python scripts/run_polygon_backtest.py --timeframe 5m

    # Single symbol
    python scripts/run_polygon_backtest.py --symbols NVDA SPY

Environment:
    POLYGON_API_KEY  Polygon.io API key (can pass via --api-key instead)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.data_loader import HistoricalDataLoader
from backtest.engine import BacktestEngine
from signals.ema_cross import EMACrossSignal
from signals.bollinger_fade import BollingerFadeSignal
from main import load_config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS   = ["NVDA", "AAPL", "TSLA", "SPY", "QQQ"]
DEFAULT_CAPITAL   = 100_000.0


def _colour(text: str, code: str) -> str:
    """Wrap text in ANSI colour codes (terminal only)."""
    return f"\033[{code}m{text}\033[0m"


def _fmt_pct(v: float) -> str:
    pct = v * 100
    col = "32" if pct >= 0 else "31"   # green / red
    return _colour(f"{pct:+.1f}%", col)


def _fmt_usd(v: float) -> str:
    col = "32" if v >= 0 else "31"
    return _colour(f"${v:+,.2f}", col)


def run_backtest_for_symbol(
    loader: HistoricalDataLoader,
    engine: BacktestEngine,
    symbol: str,
    start: str,
    end: str,
    timeframe: str,
    api_key: str,
    signals: list,
) -> dict:
    """Run a single symbol backtest and return the summary dict."""
    print(f"  Loading {symbol} ({timeframe}) {start} → {end} ...")
    candles = loader.load_from_polygon(
        symbol, start, end, timeframe=timeframe, api_key=api_key
    )

    if not candles:
        print(f"    ! No data returned for {symbol}, skipping.")
        return {"symbol": symbol, "error": "no_data"}

    # Resample to 1h for multi-timeframe signals when daily data is used
    candles_1h = None
    if timeframe == "1d":
        candles_1h = candles   # treat daily as "1h" context for EMA cross
    elif timeframe in ("1m", "5m", "15m"):
        candles_1h = loader.resample(candles, "1h")

    result = engine.run(candles, signals, candles_1h=candles_1h)
    summary = result.summary()
    summary["symbol"] = symbol
    summary["candles"] = len(candles)
    return summary


def print_table(rows: list[dict], strategy_name: str, capital: float) -> None:
    """Print a formatted results table."""
    print(f"\n{'='*70}")
    print(f"  Strategy: {strategy_name}   Capital: ${capital:,.0f}")
    print(f"{'='*70}")

    header = f"{'Symbol':<8} {'Trades':>6} {'Win%':>7} {'Net P&L':>12} {'Return':>9} {'MaxDD':>9}"
    print(header)
    print("-" * 70)

    for row in rows:
        if "error" in row:
            print(f"{row['symbol']:<8}  {'ERROR: ' + row['error']}")
            continue

        sym      = row.get("symbol", "?")
        trades   = row.get("total_trades", 0)
        win_pct  = row.get("win_rate", 0.0)
        net_pnl  = row.get("total_pnl", 0.0)
        ret      = row.get("total_return", 0.0)
        max_dd   = row.get("max_drawdown", 0.0)

        line = (
            f"{sym:<8} "
            f"{trades:>6} "
            f"{_fmt_pct(win_pct):>20} "
            f"{_fmt_usd(net_pnl):>22} "
            f"{_fmt_pct(ret):>20} "
            f"{_fmt_pct(-abs(max_dd)):>20}"
        )
        print(line)

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Polygon.io backtests for ema_cross and bollinger_fade"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help="Symbols to backtest",
    )
    parser.add_argument(
        "--start",
        default=(datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d"),
        help="Start date YYYY-MM-DD (default: 2 years ago)",
    )
    parser.add_argument(
        "--end",
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--timeframe", default="1d",
        help="Bar timeframe: 1d (free), 1h/5m (paid Polygon). Default: 1d",
    )
    parser.add_argument(
        "--capital", type=float, default=DEFAULT_CAPITAL,
        help="Starting capital per symbol (default: 100000)",
    )
    parser.add_argument(
        "--api-key", default=os.getenv("POLYGON_API_KEY", ""),
        help="Polygon API key (env: POLYGON_API_KEY)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No POLYGON_API_KEY found.  Set env var or pass --api-key.")
        sys.exit(1)

    config  = load_config()
    loader  = HistoricalDataLoader()

    strat_cfg = config["strategies"]

    # ema_cross
    ema_signal  = EMACrossSignal(strat_cfg["ema_cross"])
    ema_engine  = BacktestEngine(config, initial_capital=args.capital)

    # bollinger_fade
    boll_signal  = BollingerFadeSignal(strat_cfg["bollinger_fade"])
    boll_engine  = BacktestEngine(config, initial_capital=args.capital)

    ema_rows:  list[dict] = []
    boll_rows: list[dict] = []

    print(f"\nRunning Polygon backtest on: {', '.join(args.symbols)}")
    print(f"Period : {args.start} → {args.end}   Timeframe: {args.timeframe}")
    print(f"Capital: ${args.capital:,.0f} per symbol\n")

    for sym in args.symbols:
        print(f"[EMA Cross]")
        row = run_backtest_for_symbol(
            loader, ema_engine, sym, args.start, args.end,
            args.timeframe, args.api_key, [ema_signal],
        )
        ema_rows.append(row)

        print(f"[Bollinger Fade]")
        row = run_backtest_for_symbol(
            loader, boll_engine, sym, args.start, args.end,
            args.timeframe, args.api_key, [boll_signal],
        )
        boll_rows.append(row)
        print()

    print_table(ema_rows,  "ema_cross",      args.capital)
    print_table(boll_rows, "bollinger_fade", args.capital)

    # Combined totals
    def _sum_row(rows: list[dict], key: str) -> float:
        return sum(r.get(key, 0.0) for r in rows if "error" not in r)

    for name, rows in [("ema_cross", ema_rows), ("bollinger_fade", boll_rows)]:
        total_trades = sum(r.get("total_trades", 0) for r in rows if "error" not in r)
        total_pnl    = _sum_row(rows, "total_pnl")
        valid        = [r for r in rows if "error" not in r]
        avg_win_rate = (
            sum(r.get("win_rate", 0) for r in valid) / len(valid)
            if valid else 0.0
        )
        print(
            f"\n  {name} TOTAL | {total_trades} trades | "
            f"avg win rate {_fmt_pct(avg_win_rate)} | "
            f"net P&L across all symbols {_fmt_usd(total_pnl)}"
        )


if __name__ == "__main__":
    main()
