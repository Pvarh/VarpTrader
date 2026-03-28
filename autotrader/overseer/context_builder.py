"""Context builder for VarpTrader overseer.

Assembles a comprehensive text context from log events, trade database,
and config for consumption by an LLM-based overseer prompt. Keeps total
output under ~6000 tokens.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from journal.db import TradeDatabase
from overseer.change_memory import (
    load_json_list,
    summarize_change_log,
    summarize_strategy_log,
)
from overseer.log_extractor import extract_events


def _safe_read_json(path: str) -> dict:
    """Read a JSON file, returning empty dict on failure."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("overseer_config_read_failed | path={} err={}", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _fmt_trade_row(t: dict) -> str:
    """Format a single trade dict as a compact one-liner."""
    return (
        f"  {t.get('symbol','?'):>10s} | {t.get('strategy','?'):>16s} | "
        f"{t.get('direction','?'):>5s} | pnl={t.get('pnl', 0):+.2f} | "
        f"{t.get('outcome','?')}"
    )


def _compute_stats(trades: list[dict], key: str) -> str:
    """Compute win rate and avg PnL grouped by *key* (strategy or symbol).

    Returns a formatted multi-line string.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        groups[t.get(key, "unknown")].append(t)

    lines: list[str] = []
    for name, group in sorted(groups.items()):
        wins = sum(1 for t in group if t.get("outcome") == "win")
        total = len(group)
        wr = (wins / total * 100) if total else 0
        avg_pnl = sum(t.get("pnl", 0) or 0 for t in group) / total if total else 0
        lines.append(f"  {name:>16s}: {wins}/{total} wins ({wr:5.1f}%)  avg_pnl={avg_pnl:+.2f}")
    return "\n".join(lines) if lines else "  (no data)"


def _count_blocked(events: dict) -> str:
    """Summarise blocked signal counts by filter type."""
    counts: dict[str, int] = defaultdict(int)
    for b in events.get("blocked", []):
        counts[b.get("reason", "unknown")] += 1
    if not counts:
        return "  (none)"
    return "\n".join(f"  {k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))


def _fmt_event_list(events: list[dict], max_items: int = 10) -> str:
    """Format a list of event dicts as compact lines."""
    if not events:
        return "  (none)"
    lines = []
    for e in events[-max_items:]:
        parts = " | ".join(f"{k}={v}" for k, v in e.items() if k != "ts" and v)
        lines.append(f"  [{e.get('ts', '?')}] {parts}")
    return "\n".join(lines)


def build_context(
    db_path: str = "data/trades.db",
    log_path: str = "logs/autotrader.log",
    config_path: str = "config.json",
    change_log_path: str = "overseer/change_log.json",
    strategy_log_path: str = "overseer/strategy_log.json",
) -> str:
    """Assemble the full overseer context for an LLM prompt.

    Args:
        db_path: Path to the SQLite trades database.
        log_path: Path to the autotrader log file.
        config_path: Path to config.json.

    Returns:
        Formatted multi-line text string (target <6000 tokens).
    """
    logger.info(
        "overseer_building_context | db={} log={} config={} change_log={} strategy_log={}",
        db_path,
        log_path,
        config_path,
        change_log_path,
        strategy_log_path,
    )

    # 1. Structured log events (last 24h)
    events = extract_events(log_path=log_path, hours=24)

    # 2. Database queries
    db = TradeDatabase(db_path)
    try:
        # Last 30 closed trades
        all_closed: list[dict] = []
        try:
            with db._cursor() as cur:
                cur.execute(
                    "SELECT * FROM trades WHERE outcome IS NOT NULL AND outcome != 'open' "
                    "ORDER BY timestamp DESC LIMIT 30"
                )
                all_closed = [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            logger.warning("overseer_query_closed_failed | err={}", exc)

        # Trades from last 7 days for stats
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        recent_trades = db.get_trades_since(seven_days_ago)
        closed_recent = [t for t in recent_trades if t.get("outcome") and t["outcome"] != "open"]

        # Open positions
        open_positions: list[dict] = []
        try:
            open_positions = db.get_open_trades()
        except Exception as exc:
            logger.warning("overseer_query_open_failed | err={}", exc)

        # Latest nightly analysis run
        latest_analysis: dict | None = None
        try:
            with db._cursor() as cur:
                cur.execute(
                    "SELECT run_timestamp, trades_analyzed, report_markdown, "
                    "config_changes_json, approved "
                    "FROM analysis_runs ORDER BY id DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row:
                    latest_analysis = dict(row)
        except Exception as exc:
            logger.warning("overseer_query_analysis_failed | err={}", exc)
    finally:
        db.close()

    # 3. Config
    config = _safe_read_json(config_path)

    # 3b. Persistent overseer memory
    change_log_entries = load_json_list(change_log_path)
    strategy_log_entries = load_json_list(strategy_log_path)

    # 4. Per-strategy stats (7 days)
    strategy_stats = _compute_stats(closed_recent, "strategy")

    # 5. Per-symbol stats (7 days)
    symbol_stats = _compute_stats(closed_recent, "symbol")

    # 6. Blocked signal counts
    blocked_summary = _count_blocked(events)

    # 7. Build the output
    sections: list[str] = []

    # Header
    sections.append("=" * 60)
    sections.append("VARPTRADER OVERSEER CONTEXT")
    sections.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    sections.append("=" * 60)

    # Config summary (compact)
    sections.append("\n--- CURRENT CONFIG ---")
    config_compact = json.dumps(config, indent=2)
    # Truncate config if too long
    if len(config_compact) > 1500:
        config_compact = config_compact[:1500] + "\n  ... (truncated)"
    sections.append(config_compact)

    # Strategy performance (7 days)
    sections.append("\n--- STRATEGY PERFORMANCE (7 days) ---")
    sections.append(strategy_stats)

    # Symbol performance (7 days)
    sections.append("\n--- SYMBOL PERFORMANCE (7 days) ---")
    sections.append(symbol_stats)

    # Last 30 closed trades
    sections.append(f"\n--- LAST {len(all_closed)} CLOSED TRADES ---")
    for t in all_closed[:20]:
        sections.append(_fmt_trade_row(t))

    # Open positions
    sections.append(f"\n--- OPEN POSITIONS ({len(open_positions)}) ---")
    for t in open_positions:
        sections.append(
            f"  {t.get('symbol','?'):>10s} | {t.get('strategy','?'):>16s} | "
            f"{t.get('direction','?'):>5s} | entry={t.get('entry_price', 0):.4f}"
        )
    if not open_positions:
        sections.append("  (none)")

    # Blocked signals
    sections.append("\n--- BLOCKED SIGNALS (24h) ---")
    sections.append(blocked_summary)

    # Fired vs blocked ratio
    fired_count = len(events.get("entries", []))
    blocked_count = len(events.get("blocked", []))
    sections.append(f"\nFired signals (24h): {fired_count}")
    sections.append(f"Blocked signals (24h): {blocked_count}")
    if fired_count > 0:
        sections.append(f"Blocked:Fired ratio: {blocked_count / fired_count:.1f}:1")
    elif blocked_count > 0:
        sections.append(f"Blocked:Fired ratio: {blocked_count}:0 (ALL blocked)")

    # Trade entries (24h)
    sections.append("\n--- TRADE ENTRIES (24h log) ---")
    sections.append(_fmt_event_list(events.get("entries", []), max_items=10))

    # Trade exits (24h)
    sections.append("\n--- TRADE EXITS (24h log) ---")
    sections.append(_fmt_event_list(events.get("exits", []), max_items=10))

    # Regime detections
    sections.append("\n--- REGIME DETECTIONS (24h) ---")
    sections.append(_fmt_event_list(events.get("regimes", []), max_items=10))

    # Kill switch
    if events.get("kill_switch"):
        sections.append("\n--- KILL SWITCH EVENTS ---")
        sections.append(_fmt_event_list(events["kill_switch"]))

    # Signal starvation
    if events.get("starvation"):
        sections.append("\n--- SIGNAL STARVATION ---")
        sections.append(_fmt_event_list(events["starvation"]))

    # Config changes
    if events.get("config_changes"):
        sections.append("\n--- CONFIG CHANGES (24h) ---")
        sections.append(_fmt_event_list(events["config_changes"]))

    # Persistent overseer change memory
    sections.append("\n--- CHANGE MEMORY SUMMARY ---")
    sections.append(summarize_change_log(change_log_entries))
    sections.append("\n--- FULL CHANGE LOG JSON ---")
    sections.append(json.dumps(change_log_entries, indent=2))

    sections.append("\n--- STRATEGY MEMORY SUMMARY ---")
    sections.append(summarize_strategy_log(strategy_log_entries))
    sections.append("\n--- FULL STRATEGY LOG JSON ---")
    sections.append(json.dumps(strategy_log_entries, indent=2))

    # Errors / warnings
    sections.append(f"\n--- ERRORS/WARNINGS (24h, {len(events.get('errors', []))} total) ---")
    sections.append(_fmt_event_list(events.get("errors", []), max_items=15))

    # Latest nightly analysis report
    sections.append("\n--- LATEST NIGHTLY ANALYSIS REPORT ---")
    if latest_analysis:
        sections.append(
            f"  Run: {latest_analysis.get('run_timestamp', '?')} | "
            f"trades_analyzed={latest_analysis.get('trades_analyzed', '?')} | "
            f"approved={latest_analysis.get('approved', '?')}"
        )
        report_md = latest_analysis.get("report_markdown") or ""
        if len(report_md) > 2000:
            report_md = report_md[:2000] + "\n  ... (truncated)"
        sections.append(report_md if report_md else "  (no report text)")
        cfg_changes = latest_analysis.get("config_changes_json") or ""
        if cfg_changes:
            if len(cfg_changes) > 500:
                cfg_changes = cfg_changes[:500] + " ... (truncated)"
            sections.append(f"  config_changes: {cfg_changes}")
    else:
        sections.append("  (no nightly analysis runs found)")

    sections.append("\n" + "=" * 60)

    context = "\n".join(sections)

    # Hard truncation safety net (~6000 tokens ~ 24000 chars)
    max_chars = 48000
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n... (context truncated to fit token budget)"
        logger.warning("overseer_context_truncated | chars={}", len(context))

    logger.info("overseer_context_built | chars={}", len(context))
    return context
