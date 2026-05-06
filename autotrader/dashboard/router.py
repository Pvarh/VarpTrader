"""FastAPI dashboard router with HTML pages and WebSocket endpoints."""
from __future__ import annotations

import asyncio
import csv
import difflib
import io
import json
import math
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Body,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Header,
)
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ---------------------------------------------------------------------------
# State -- these get set by main.py when mounting the router
# ---------------------------------------------------------------------------
_db = None          # TradeDatabase
_analyzer = None    # PerformanceAnalyzer
_config: dict = {}
_start_time: float = time.time()
_whale_flags: dict = {}   # symbol -> {"buy": bool, "sell": bool, "expires": iso_str}
_recent_signals: list = []  # last 10 signal events
_kill_switch = None  # KillSwitch instance
_paper_executor = None  # PaperExecutor instance
_stock_feed = None
_crypto_feed = None
_order_validator = None
_telegram = None  # TelegramAlert instance
_config_updater = None  # ConfigUpdater instance


def init(
    db,
    analyzer,
    config: dict,
    kill_switch=None,
    paper_executor=None,
    stock_feed=None,
    crypto_feed=None,
    order_validator=None,
    telegram=None,
    config_updater=None,
) -> None:
    """Called from main.py to inject dependencies."""
    global _db, _analyzer, _config, _kill_switch, _paper_executor
    global _stock_feed, _crypto_feed, _order_validator, _telegram, _config_updater
    _db, _analyzer, _config = db, analyzer, config
    _kill_switch = kill_switch
    _paper_executor = paper_executor
    _stock_feed = stock_feed
    _crypto_feed = crypto_feed
    _order_validator = order_validator
    _telegram = telegram
    _config_updater = config_updater
    logger.info("dashboard_router_initialized")


def _is_paper_mode(config: dict) -> bool:
    """Resolve paper/live mode from env first, then config fallbacks."""
    env_value = os.getenv("PAPER_TRADE")
    if env_value is not None:
        return env_value.lower() == "true"
    if "trading" in config:
        return config["trading"].get("paper_trade", True)
    return config.get("paper_trade", True)


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming channels."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {"signals": [], "pnl": []}

    async def connect(self, channel: str, ws: WebSocket) -> None:
        await ws.accept()
        if channel not in self._connections:
            self._connections[channel] = []
        self._connections[channel].append(ws)

    async def disconnect(self, channel: str, ws: WebSocket) -> None:
        if channel in self._connections and ws in self._connections[channel]:
            self._connections[channel].remove(ws)

    async def broadcast(self, channel: str, data: dict) -> None:
        if channel not in self._connections:
            return
        dead: list[WebSocket] = []
        for ws in self._connections[channel]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(channel, ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Performance metrics computation
# ---------------------------------------------------------------------------

def _compute_performance_metrics(days: int = 30) -> dict:
    """Compute key performance metrics from trade history."""
    if not _db:
        return {"win_rate": 0, "sharpe": 0, "max_drawdown": 0, "profit_factor": 0, "total_trades": 0, "avg_pnl": 0}

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    trades = _db.get_trades_since_by_activity(since) or []
    closed = [t for t in trades if t.get("outcome") in ("win", "loss", "breakeven")]

    if not closed:
        return {"win_rate": 0, "sharpe": 0, "max_drawdown": 0, "profit_factor": 0, "total_trades": 0, "avg_pnl": 0}

    wins = sum(1 for t in closed if t.get("outcome") == "win")
    win_rate = wins / len(closed) if closed else 0

    pnls = [float(t.get("pnl", 0) or 0) for t in closed]
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0
    std_pnl = (sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5 if len(pnls) > 1 else 0
    sharpe = (avg_pnl / std_pnl) * (252 ** 0.5) if std_pnl > 0 else 0

    # Max drawdown from cumulative PnL
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    return {
        "win_rate": round(win_rate * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
        "total_trades": len(closed),
        "avg_pnl": round(avg_pnl, 2),
    }


def _compute_strategy_pnl(days: int = 30) -> dict:
    """Compute P&L breakdown per strategy."""
    if not _db:
        return {}
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    trades = _db.get_trades_since_by_activity(since) or []
    closed = [t for t in trades if t.get("outcome") in ("win", "loss", "breakeven")]

    result: dict[str, dict] = {}
    for t in closed:
        strat = t.get("strategy", "unknown")
        if strat not in result:
            result[strat] = {"pnl": 0.0, "wins": 0, "losses": 0, "trades": 0}
        pnl = float(t.get("pnl", 0) or 0)
        result[strat]["pnl"] += pnl
        result[strat]["trades"] += 1
        if t.get("outcome") == "win":
            result[strat]["wins"] += 1
        elif t.get("outcome") == "loss":
            result[strat]["losses"] += 1

    for strat in result:
        total = result[strat]["trades"]
        result[strat]["pnl"] = round(result[strat]["pnl"], 2)
        result[strat]["win_rate"] = round(result[strat]["wins"] / total * 100, 1) if total > 0 else 0

    return result


def _compute_daily_pnl(days: int = 30) -> list[dict]:
    """Compute daily P&L for bar chart."""
    if not _db:
        return []
    result = []
    for i in range(days):
        date = datetime.now(timezone.utc) - timedelta(days=days - 1 - i)
        date_str = date.strftime("%Y-%m-%d")
        pnl = _db.get_daily_pnl(date_str)
        result.append({"date": date_str, "pnl": round(pnl, 2)})
    return result


def _get_regime_status() -> dict:
    """Get current regime detection and filter status."""
    try:
        from main import AutoTrader
        # Get the trader instance's current state if available
        config = _load_config_from_disk()
        regime_cfg = config.get("regime", {})
        strategies = config.get("strategies", {})

        enabled_strategies = [name for name, cfg in strategies.items() if cfg.get("enabled")]
        disabled_strategies = [name for name, cfg in strategies.items() if not cfg.get("enabled")]

        return {
            "adx_threshold": regime_cfg.get("adx_threshold", 25.0),
            "enabled_strategies": enabled_strategies,
            "disabled_strategies": disabled_strategies,
            "session_bias_enabled": config.get("session_bias", {}).get("enabled", False),
            "whale_enabled": config.get("whale", {}).get("enabled", False),
            "polymarket_threshold": config.get("polymarket", {}).get("block_short_threshold", 0.65),
        }
    except Exception:
        return {}


def _get_unrealized_pnl() -> float:
    if not _paper_executor:
        return 0.0
    try:
        positions = _paper_executor.get_positions()
        if not positions:
            return 0.0
        return sum(p.get("unrealized_pl", 0.0) for p in positions)
    except Exception:
        return 0.0


def _get_portfolio_stats() -> dict:
    if not _paper_executor:
        return {"cash": 0.0, "invested": 0.0, "equity": 0.0, "unrealized_pnl": 0.0, "positions": []}
    try:
        portfolio = _paper_executor._portfolio
        cash = portfolio.cash
        positions = _paper_executor.get_positions()
        invested = sum(
            abs(p.get("qty", 0)) * p.get("avg_entry_price", 0) for p in positions
        )
        unrealized = sum(p.get("unrealized_pl", 0.0) for p in positions)
        equity = cash + invested + unrealized
        return {
            "cash": round(cash, 2),
            "invested": round(invested, 2),
            "equity": round(equity, 2),
            "unrealized_pnl": round(unrealized, 2),
            "positions": positions,
        }
    except Exception:
        return {"cash": 0.0, "invested": 0.0, "equity": 0.0, "unrealized_pnl": 0.0, "positions": []}


def _infer_market(symbol: str) -> str:
    return "crypto" if "/" in symbol else "stock"


def _estimate_position_price(position: dict) -> float:
    qty = abs(float(position.get("qty", 0) or 0))
    market_value = float(position.get("market_value", 0) or 0)
    entry_price = float(position.get("avg_entry_price", 0) or 0)
    if qty > 0 and market_value > 0:
        return market_value / qty
    return entry_price


def _dashboard_api_key_required(x_api_key: str | None) -> None:
    expected_key = os.getenv("DASHBOARD_API_KEY", "").strip()
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


def _resolve_manual_symbol(raw_symbol: str) -> tuple[str, str]:
    symbol = raw_symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    watchlist = _config.get("watchlist", {})
    stock_symbols = [str(item).upper() for item in watchlist.get("stocks", [])]
    crypto_symbols = [str(item).upper() for item in watchlist.get("crypto", [])]
    crypto_aliases = {item.replace("/", ""): item for item in crypto_symbols}

    if symbol in crypto_symbols:
        return symbol, "crypto"
    if symbol in crypto_aliases:
        return crypto_aliases[symbol], "crypto"
    if "/" in symbol:
        return symbol, "crypto"
    if symbol in stock_symbols:
        return symbol, "stock"

    match = difflib.get_close_matches(symbol, stock_symbols, n=1, cutoff=0.75)
    if match:
        return match[0], "stock"
    return symbol, "stock"


def _get_manual_market_price(symbol: str, market: str) -> float:
    feed = _stock_feed if market == "stock" else _crypto_feed
    if feed is None:
        raise HTTPException(status_code=500, detail=f"No {market} feed available")
    try:
        price = float(feed.get_latest_price(symbol) or 0.0)
    except Exception:
        logger.exception("dashboard_price_lookup_failed | symbol={symbol} market={market}", symbol=symbol, market=market)
        raise HTTPException(status_code=500, detail=f"Price lookup failed for {symbol}")
    if price <= 0:
        raise HTTPException(status_code=400, detail=f"No live price available for {symbol}")
    return price


def _calculate_manual_quantity(market: str, market_price: float) -> float:
    if not _paper_executor:
        raise HTTPException(status_code=400, detail="Paper executor not available")
    portfolio = _paper_executor._portfolio
    account_value = float(portfolio.equity)
    available_cash = float(portfolio.cash)
    risk_pct = float(_config.get("risk", {}).get("position_size_pct", 0.005) or 0.005)
    notional = min(available_cash, max(0.0, account_value * risk_pct))

    if market == "stock":
        quantity = int(notional / market_price) if market_price > 0 else 0
        if quantity <= 0 and available_cash >= market_price > 0:
            quantity = 1
        return float(quantity)

    if market_price <= 0:
        return 0.0
    scale = 10 ** 6
    return int((notional / market_price) * scale) / scale


def _load_config_from_disk() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.exception("config_load_failed | path={path}", path=str(config_path))
        return _config if _config else {}


def _load_weekly_bias() -> dict:
    bias_path = Path(__file__).resolve().parent.parent / "weekly_bias.json"
    try:
        with open(bias_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"week_start": "", "generated_at": "", "biases": {}}


def _is_kill_switch_active() -> bool:
    if _kill_switch is not None:
        return _kill_switch.halted
    return False


# ===================================================================
# HTML Pages
# ===================================================================

@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    config = _load_config_from_disk()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_pnl = _db.get_daily_pnl(today) if _db else 0.0
    open_trades = _db.get_open_trades() if _db else []

    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    all_trades = _db.get_trades_since_by_activity(since) if _db else []
    recent_trades = all_trades[:20]

    daily_pnl_data = _compute_daily_pnl(30)
    performance = _compute_performance_metrics(30)
    strategy_pnl = _compute_strategy_pnl(30)
    regime_status = _get_regime_status()

    paper_mode = _is_paper_mode(config)
    manual_symbols = list(config.get("watchlist", {}).get("stocks", [])) + list(config.get("watchlist", {}).get("crypto", []))
    portfolio = _get_portfolio_stats()

    return templates.TemplateResponse(request, "dashboard.html", {
        "daily_pnl": round(daily_pnl, 2),
        "unrealized_pnl": portfolio["unrealized_pnl"],
        "cash": portfolio["cash"],
        "invested": portfolio["invested"],
        "equity": portfolio["equity"],
        "open_trades": open_trades,
        "recent_trades": recent_trades,
        "daily_pnl_data": json.dumps(daily_pnl_data),
        "kill_switch_active": _is_kill_switch_active(),
        "paper_mode": paper_mode,
        "portfolio_positions": portfolio["positions"],
        "manual_symbols": manual_symbols,
        "performance": performance,
        "strategy_pnl": json.dumps(strategy_pnl),
        "regime_status": regime_status,
    })


@router.get("/trades", response_class=HTMLResponse)
async def trades_page(
    request: Request,
    strategy: Optional[str] = Query(default=None),
    outcome: Optional[str] = Query(default=None),
    market: Optional[str] = Query(default=None),
    days: int = Query(default=30, ge=1, le=365),
    page: int = Query(default=1, ge=1),
):
    per_page = 50
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    all_trades = _db.get_trades_since_by_activity(since) if _db else []

    if strategy:
        all_trades = [t for t in all_trades if t.get("strategy") == strategy]
    if outcome:
        all_trades = [t for t in all_trades if t.get("outcome") == outcome]
    if market:
        all_trades = [t for t in all_trades if t.get("market") == market]

    total = len(all_trades)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    trades = all_trades[start : start + per_page]

    strategies = _db.get_all_strategies() if _db else []
    markets = sorted({t.get("market", "") for t in all_trades if t.get("market")})

    return templates.TemplateResponse(request, "trades.html", {
        "trades": trades,
        "strategies": strategies,
        "markets": markets,
        "filter_strategy": strategy or "",
        "filter_outcome": outcome or "",
        "filter_market": market or "",
        "filter_days": days,
        "page": page,
        "total_pages": total_pages,
        "total": total,
    })


@router.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    latest_run = None
    if _db:
        conn = _db._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM analysis_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                latest_run = dict(row)
        except Exception:
            logger.exception("failed_to_fetch_latest_analysis_run")

    weekly_bias = _load_weekly_bias()
    win_rates = {}
    if _analyzer:
        win_rates = _analyzer.compute_strategy_win_rates(lookback_days=30)

    whale_corr = {"with_whale": 0.0, "without_whale": 0.0}
    if _analyzer:
        whale_corr = _analyzer.compute_whale_correlation()

    return templates.TemplateResponse(request, "analysis.html", {
        "latest_run": latest_run,
        "weekly_bias": weekly_bias,
        "win_rates": json.dumps(win_rates),
        "whale_correlation": whale_corr,
    })


@router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    config = _load_config_from_disk()
    pending_changes = []
    if _db:
        conn = _db._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM analysis_runs WHERE approved = 0 AND config_changes_json IS NOT NULL ORDER BY id DESC"
            ).fetchall()
            pending_changes = [dict(r) for r in rows]
        except Exception:
            logger.exception("failed_to_fetch_pending_changes")

    auto_apply = False
    if "analysis" in config:
        auto_apply = config["analysis"].get("auto_apply_changes", False)

    return templates.TemplateResponse(request, "config.html", {
        "config": config,
        "pending_changes": pending_changes,
        "auto_apply": auto_apply,
    })


# ===================================================================
# JSON API endpoints
# ===================================================================

@router.get("/whale")
async def whale_flags():
    return _whale_flags


@router.get("/signals")
async def recent_signals():
    return _recent_signals


@router.get("/health")
async def health_check():
    config = _load_config_from_disk()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_pnl = _db.get_daily_pnl(today) if _db else 0.0
    open_trades = _db.get_open_trades() if _db else []
    portfolio = _get_portfolio_stats()
    uptime_seconds = time.time() - _start_time

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(uptime_seconds, 1),
        "daily_pnl": round(daily_pnl, 2),
        "open_positions": max(len(open_trades), len(portfolio.get("positions", []))),
        "kill_switch_active": _is_kill_switch_active(),
        "whale_flags_count": len(_whale_flags),
        "recent_signals_count": len(_recent_signals),
    }


@router.get("/api/performance")
async def api_performance(days: int = Query(default=30, ge=1, le=365)):
    """Return performance metrics as JSON."""
    return _compute_performance_metrics(days)


@router.get("/api/strategy-pnl")
async def api_strategy_pnl(days: int = Query(default=30, ge=1, le=365)):
    """Return per-strategy P&L breakdown."""
    return _compute_strategy_pnl(days)


@router.get("/api/regime")
async def api_regime():
    """Return current regime/filter status."""
    return _get_regime_status()


@router.get("/export/trades")
async def export_trades_csv(
    days: int = Query(default=30, ge=1, le=365),
    x_api_key: Optional[str] = Header(default=None),
):
    """Export trades as CSV download."""
    _dashboard_api_key_required(x_api_key)
    if not _db:
        raise HTTPException(status_code=500, detail="Database not available")

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    trades = _db.get_trades_since_by_activity(since) or []

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "timestamp", "symbol", "market", "strategy", "direction",
        "entry_price", "exit_price", "quantity", "stop_loss", "take_profit",
        "pnl", "pnl_pct", "outcome", "whale_flag", "paper_trade",
    ])
    for t in trades:
        writer.writerow([
            t.get("id", ""), t.get("timestamp", ""), t.get("symbol", ""),
            t.get("market", ""), t.get("strategy", ""), t.get("direction", ""),
            t.get("entry_price", ""), t.get("exit_price", ""), t.get("quantity", ""),
            t.get("stop_loss", ""), t.get("take_profit", ""),
            t.get("pnl", ""), t.get("pnl_pct", ""), t.get("outcome", ""),
            t.get("whale_flag", ""), t.get("paper_trade", ""),
        ])

    output.seek(0)
    filename = f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/config/approve/{run_id}")
async def approve_config_change(
    run_id: int,
    x_api_key: Optional[str] = Header(default=None),
):
    _dashboard_api_key_required(x_api_key)
    if not _db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    conn = _db._get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM analysis_runs WHERE id = ?", (run_id,)
        ).fetchone()
    except Exception:
        raise HTTPException(status_code=500, detail="Database error")

    if not row:
        raise HTTPException(status_code=404, detail=f"Analysis run {run_id} not found")

    run = dict(row)
    if run.get("approved"):
        return {"status": "already_approved", "run_id": run_id}

    try:
        conn.execute("UPDATE analysis_runs SET approved = 1 WHERE id = ?", (run_id,))
        conn.commit()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to approve")

    return {"status": "approved", "run_id": run_id}


@router.post("/positions/close/{symbol:path}")
async def close_dashboard_position(
    symbol: str,
    x_api_key: Optional[str] = Header(default=None),
):
    _dashboard_api_key_required(x_api_key)
    if not _paper_executor or not _is_paper_mode(_config):
        raise HTTPException(status_code=400, detail="Dashboard close is available only in paper mode")

    try:
        positions = _paper_executor.get_positions()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch paper positions")

    position = next(
        (item for item in positions if str(item.get("symbol", "")).upper() == symbol.upper()),
        None,
    )
    if not position:
        raise HTTPException(status_code=404, detail=f"Open position {symbol} not found")

    normalized_symbol = str(position.get("symbol", symbol))
    market_price = _estimate_position_price(position)
    if market_price <= 0:
        raise HTTPException(status_code=400, detail=f"No usable price snapshot for {normalized_symbol}")

    market = _infer_market(normalized_symbol)
    try:
        result = _paper_executor.close_position(
            symbol=normalized_symbol,
            market_price=market_price,
            market=market,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Paper close failed")

    if not result:
        raise HTTPException(status_code=500, detail="Paper close failed")

    return {
        "status": "closed",
        "symbol": normalized_symbol,
        "market": market,
        "fill_price": result.get("fill_price"),
        "pnl": result.get("pnl"),
    }


@router.post("/orders/manual")
async def dashboard_manual_order(
    payload: dict = Body(...),
    x_api_key: Optional[str] = Header(default=None),
):
    _dashboard_api_key_required(x_api_key)
    if not _paper_executor or not _is_paper_mode(_config):
        raise HTTPException(status_code=400, detail="Dashboard manual trading is available only in paper mode")
    if _order_validator is None or _db is None or _kill_switch is None:
        raise HTTPException(status_code=500, detail="Manual trading dependencies not initialized")

    action = str(payload.get("action", "")).strip().lower()
    raw_symbol = str(payload.get("symbol", "")).strip()
    symbol, market = _resolve_manual_symbol(raw_symbol)

    if action == "buy":
        market_price = _get_manual_market_price(symbol, market)
        quantity = _calculate_manual_quantity(market, market_price)
        if quantity <= 0:
            raise HTTPException(status_code=400, detail=f"Quantity would be zero for {symbol} at current price")

        open_trades = _db.get_open_trades()
        open_symbols = {trade["symbol"] for trade in open_trades} if open_trades else set()
        current_positions = len(open_symbols)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_trade_count = _db.get_daily_trade_count(today_str)

        valid, reason = _order_validator.validate(
            symbol, "buy", quantity, current_positions,
            daily_trade_count, _kill_switch.halted,
            open_symbols=open_symbols,
        )
        if not valid:
            raise HTTPException(status_code=400, detail=reason)

        result = _paper_executor.submit_market_order(
            symbol=symbol, side="buy", quantity=quantity,
            market_price=market_price, market=market,
            strategy="manual_telegram",
            stop_loss=0.0, take_profit=0.0, whale_flag=0,
        )
        if not result:
            raise HTTPException(status_code=500, detail=f"Manual buy failed for {symbol}")
        return {
            "status": "opened",
            "action": "buy",
            "symbol": symbol,
            "market": market,
            "fill_price": result.get("fill_price"),
            "qty": result.get("qty", quantity),
        }

    if action == "sell":
        return await close_dashboard_position(symbol=symbol, x_api_key=x_api_key)

    raise HTTPException(status_code=400, detail="Unsupported action. Use buy or sell.")


# ===================================================================
# WebSocket endpoints
# ===================================================================

@router.websocket("/ws/signals")
async def ws_signals(ws: WebSocket):
    await manager.connect("signals", ws)
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await manager.disconnect("signals", ws)
    except Exception:
        await manager.disconnect("signals", ws)


@router.websocket("/ws/pnl")
async def ws_pnl(ws: WebSocket):
    await manager.connect("pnl", ws)
    logger.info("websocket_connected | channel=pnl total={}", len(manager._connections.get("pnl", [])))
    try:
        while True:
            try:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                daily_pnl = _db.get_daily_pnl(today) if _db else 0.0
                portfolio = _get_portfolio_stats()
                open_trades = _db.get_open_trades() if _db else []

                since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
                all_trades = _db.get_trades_since_by_activity(since) if _db else []
                recent_trades = all_trades[:20]

                uptime_seconds = time.time() - _start_time
                performance = _compute_performance_metrics(30)

                pnl_data = {
                    "type": "pnl_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "daily_pnl": round(daily_pnl, 2),
                    "unrealized_pnl": portfolio["unrealized_pnl"],
                    "cash": portfolio["cash"],
                    "invested": portfolio["invested"],
                    "equity": portfolio["equity"],
                    "portfolio_positions": portfolio["positions"],
                    "open_positions": len(open_trades),
                    "open_trades": open_trades,
                    "recent_trades": recent_trades,
                    "kill_switch_active": _is_kill_switch_active(),
                    "uptime_seconds": round(uptime_seconds, 1),
                    "performance": performance,
                }
                await ws.send_json(pnl_data)
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.exception("ws_pnl_data_build_error")

            # Use receive_text with timeout to detect client disconnect quickly
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=5.0)
            except asyncio.TimeoutError:
                pass  # Normal — client didn't send anything, loop again
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("ws_pnl_unexpected_error")
    finally:
        await manager.disconnect("pnl", ws)
        logger.info("websocket_disconnected | channel=pnl remaining={}", len(manager._connections.get("pnl", [])))
