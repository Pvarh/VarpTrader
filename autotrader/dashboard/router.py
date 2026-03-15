"""FastAPI dashboard router with HTML pages and WebSocket endpoints."""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Header,
)
from fastapi.responses import HTMLResponse
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


def init(db, analyzer, config: dict, kill_switch=None, paper_executor=None) -> None:
    """Called from main.py to inject dependencies."""
    global _db, _analyzer, _config, _kill_switch, _paper_executor
    _db, _analyzer, _config = db, analyzer, config
    _kill_switch = kill_switch
    _paper_executor = paper_executor
    logger.info("dashboard_router_initialized")


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming channels."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {"signals": [], "pnl": []}

    async def connect(self, channel: str, ws: WebSocket) -> None:
        """Accept and register a WebSocket connection on a channel."""
        await ws.accept()
        if channel not in self._connections:
            self._connections[channel] = []
        self._connections[channel].append(ws)
        logger.info(
            "websocket_connected | channel={channel} total={total}",
            channel=channel,
            total=len(self._connections[channel]),
        )

    async def disconnect(self, channel: str, ws: WebSocket) -> None:
        """Remove a WebSocket connection from a channel."""
        if channel in self._connections and ws in self._connections[channel]:
            self._connections[channel].remove(ws)
        logger.info(
            "websocket_disconnected | channel={channel} remaining={remaining}",
            channel=channel,
            remaining=len(self._connections.get(channel, [])),
        )

    async def broadcast(self, channel: str, data: dict) -> None:
        """Send a JSON message to all connections on a channel."""
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


def _get_unrealized_pnl() -> float:
    """Get total unrealized PnL from paper executor positions."""
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
    """Get portfolio stats: cash, invested, equity, positions with current prices."""
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


# ---------------------------------------------------------------------------
# Helper: load config from disk
# ---------------------------------------------------------------------------

def _load_config_from_disk() -> dict:
    """Load config.json from the project root."""
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    try:
        with open(config_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.exception("config_load_failed | path={path}", path=str(config_path))
        return _config if _config else {}


def _load_weekly_bias() -> dict:
    """Load weekly_bias.json from the project root."""
    bias_path = Path(__file__).resolve().parent.parent / "weekly_bias.json"
    try:
        with open(bias_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.warning("weekly_bias_load_failed | path={path}", path=str(bias_path))
        return {"week_start": "", "generated_at": "", "biases": {}}


def _is_kill_switch_active() -> bool:
    """Check if kill switch is currently halted."""
    if _kill_switch is not None:
        return _kill_switch.halted
    return False


# ═══════════════════════════════════════════════════════════════════════════
# HTML Pages
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve the main dashboard page."""
    config = _load_config_from_disk()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_pnl = _db.get_daily_pnl(today) if _db else 0.0
    unrealized_pnl = _get_unrealized_pnl()
    open_trades = _db.get_open_trades() if _db else []

    # Recent 20 trades
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    all_trades = _db.get_trades_since(since) if _db else []
    all_trades.reverse()
    recent_trades = all_trades[:20]

    # PnL history for equity curve (30 days)
    pnl_history = []
    for i in range(30):
        date = datetime.now(timezone.utc) - timedelta(days=29 - i)
        date_str = date.strftime("%Y-%m-%d")
        pnl = _db.get_daily_pnl(date_str) if _db else 0.0
        pnl_history.append({"date": date_str, "pnl": round(pnl, 2)})

    # Determine paper mode
    paper_mode = True
    # Check multiple possible config shapes
    if "trading" in config:
        paper_mode = config["trading"].get("paper_trade", True)
    elif "risk" in config:
        paper_mode = config.get("paper_trade", True)

    portfolio = _get_portfolio_stats()

    return templates.TemplateResponse(request, "dashboard.html", {
        "daily_pnl": round(daily_pnl, 2),
        "unrealized_pnl": portfolio["unrealized_pnl"],
        "cash": portfolio["cash"],
        "invested": portfolio["invested"],
        "equity": portfolio["equity"],
        "open_trades": open_trades,
        "recent_trades": recent_trades,
        "pnl_history": json.dumps(pnl_history),
        "kill_switch_active": _is_kill_switch_active(),
        "paper_mode": paper_mode,
        "portfolio_positions": portfolio["positions"],
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
    """Serve the trade journal page with filtering."""
    per_page = 50
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    all_trades = _db.get_trades_since(since) if _db else []
    all_trades.reverse()

    # Apply filters
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

    # Get unique strategies and markets for filter dropdowns
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
    """Serve the analysis page."""
    # Latest analysis run
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

    # Weekly bias
    weekly_bias = _load_weekly_bias()

    # Strategy win rates
    win_rates = {}
    if _analyzer:
        win_rates = _analyzer.compute_strategy_win_rates(lookback_days=30)

    # Whale correlation
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
    """Serve the config viewer page."""
    config = _load_config_from_disk()

    # Pending AI changes (unapproved analysis runs with config changes)
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


# ═══════════════════════════════════════════════════════════════════════════
# JSON API endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/whale")
async def whale_flags():
    """Return current whale flags dict."""
    return _whale_flags


@router.get("/signals")
async def recent_signals():
    """Return the last 10 fired signals."""
    return _recent_signals


@router.get("/health")
async def health_check():
    """Return system status JSON."""
    config = _load_config_from_disk()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_pnl = _db.get_daily_pnl(today) if _db else 0.0
    open_trades = _db.get_open_trades() if _db else []

    uptime_seconds = time.time() - _start_time

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(uptime_seconds, 1),
        "daily_pnl": round(daily_pnl, 2),
        "open_positions": len(open_trades),
        "kill_switch_active": _is_kill_switch_active(),
        "whale_flags_count": len(_whale_flags),
        "recent_signals_count": len(_recent_signals),
    }


@router.post("/config/approve/{run_id}")
async def approve_config_change(
    run_id: int,
    x_api_key: Optional[str] = Header(default=None),
):
    """Approve a pending AI config change. Requires X-API-Key header."""
    expected_key = os.getenv("DASHBOARD_API_KEY")
    if not expected_key or x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    if not _db:
        raise HTTPException(status_code=500, detail="Database not initialized")

    # Look up the analysis run
    conn = _db._get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM analysis_runs WHERE id = ?", (run_id,)
        ).fetchone()
    except Exception:
        logger.exception("approve_lookup_failed | run_id={run_id}", run_id=run_id)
        raise HTTPException(status_code=500, detail="Database error")

    if not row:
        raise HTTPException(status_code=404, detail=f"Analysis run {run_id} not found")

    run = dict(row)
    if run.get("approved"):
        return {"status": "already_approved", "run_id": run_id}

    # Mark as approved
    try:
        conn.execute(
            "UPDATE analysis_runs SET approved = 1 WHERE id = ?", (run_id,)
        )
        conn.commit()
    except Exception:
        logger.exception("approve_update_failed | run_id={run_id}", run_id=run_id)
        raise HTTPException(status_code=500, detail="Failed to approve")

    logger.info("config_change_approved | run_id={run_id}", run_id=run_id)
    return {"status": "approved", "run_id": run_id}


# ═══════════════════════════════════════════════════════════════════════════
# WebSocket endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.websocket("/ws/signals")
async def ws_signals(ws: WebSocket):
    """Stream signal events as JSON in real-time."""
    await manager.connect("signals", ws)
    try:
        while True:
            # Keep connection alive; listen for client pings or close
            data = await ws.receive_text()
            # Echo back as heartbeat acknowledgment
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await manager.disconnect("signals", ws)
    except Exception:
        await manager.disconnect("signals", ws)


@router.websocket("/ws/pnl")
async def ws_pnl(ws: WebSocket):
    """Stream live dashboard state every 5 seconds."""
    await manager.connect("pnl", ws)
    try:
        while True:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_pnl = _db.get_daily_pnl(today) if _db else 0.0
            portfolio = _get_portfolio_stats()
            open_trades = _db.get_open_trades() if _db else []

            since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            all_trades = _db.get_trades_since(since) if _db else []
            all_trades.reverse()
            recent_trades = all_trades[:20]

            uptime_seconds = time.time() - _start_time

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
            }
            await ws.send_json(pnl_data)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        await manager.disconnect("pnl", ws)
    except Exception:
        await manager.disconnect("pnl", ws)
