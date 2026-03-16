"""Log extractor for VarpTrader overseer.

Parses autotrader.log (loguru format) and extracts meaningful events
from a configurable time window. Returns structured dicts by category,
capped at 20 entries each (most recent kept) to fit a ~4000 token budget.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Loguru line pattern
# Example: 2026-03-16 20:26:15.474 | INFO     | module:function:line - message
# ---------------------------------------------------------------------------
_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
    r"\s*\|\s*(?P<level>\w+)\s*\|\s*"
    r"(?P<location>\S+)"
    r"\s*-\s*(?P<body>.*)$"
)

# Key=value pairs inside the log body
_KV_RE = re.compile(r"(\w+)=(\S+)")

# Category match patterns
_CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    "entries": re.compile(r"trade_opened"),
    "exits": re.compile(r"trade_closed|exit_alert_sent"),
    "regimes": re.compile(r"regime_detected"),
    "blocked": re.compile(
        r"regime_blocked|session_bias_blocked|polymarket_blocked"
        r"|whale_suppressed|swing_bias_blocked"
    ),
    "kill_switch": re.compile(r"kill_switch_triggered"),
    "starvation": re.compile(r"signal_starvation"),
    "config_changes": re.compile(r"config_applied|config_changes|migration_applied"),
}

# Noise patterns to skip for errors/warnings
_NOISE_RE = re.compile(r"polygon_rate_limited|polygon_fallback_to_yfinance")

MAX_PER_CATEGORY = 20


def _parse_kv(body: str) -> dict[str, str]:
    """Extract key=value pairs from a log body string."""
    return dict(_KV_RE.findall(body))


def _parse_line(line: str) -> dict[str, Any] | None:
    """Parse a single loguru log line into its components.

    Returns:
        Dict with keys ts, level, location, body, kv -- or None if
        the line does not match the expected format.
    """
    m = _LINE_RE.match(line.strip())
    if not m:
        return None
    return {
        "ts": m.group("ts"),
        "level": m.group("level").strip(),
        "location": m.group("location"),
        "body": m.group("body"),
        "kv": _parse_kv(m.group("body")),
    }


def _extract_entry(parsed: dict) -> dict:
    """Build an entry record from a trade_opened log line."""
    kv = parsed["kv"]
    return {
        "ts": parsed["ts"],
        "trade_id": kv.get("trade_id", ""),
        "symbol": kv.get("symbol", ""),
        "strategy": kv.get("strategy", ""),
        "direction": kv.get("direction", ""),
        "entry_price": kv.get("entry", kv.get("entry_price", "")),
    }


def _extract_exit(parsed: dict) -> dict:
    """Build an exit record from a trade_closed / exit_alert_sent line."""
    kv = parsed["kv"]
    return {
        "ts": parsed["ts"],
        "trade_id": kv.get("trade_id", ""),
        "symbol": kv.get("symbol", ""),
        "reason": kv.get("reason", ""),
        "pnl": kv.get("pnl", ""),
        "outcome": kv.get("outcome", ""),
    }


def _extract_regime(parsed: dict) -> dict:
    """Build a regime record from a regime_detected line."""
    kv = parsed["kv"]
    return {
        "ts": parsed["ts"],
        "symbol": kv.get("symbol", ""),
        "regime": kv.get("regime", ""),
        "ema20": kv.get("ema20", ""),
        "ema50": kv.get("ema50", ""),
        "adx": kv.get("adx", ""),
    }


def _extract_blocked(parsed: dict) -> dict:
    """Build a blocked-signal record."""
    kv = parsed["kv"]
    # Determine the reason from the message key
    body = parsed["body"]
    reason = ""
    for tag in (
        "regime_blocked", "session_bias_blocked", "polymarket_blocked",
        "whale_suppressed", "swing_bias_blocked",
    ):
        if tag in body:
            reason = tag
            break
    return {
        "ts": parsed["ts"],
        "symbol": kv.get("symbol", ""),
        "strategy": kv.get("strategy", ""),
        "direction": kv.get("direction", ""),
        "reason": reason,
    }


def _extract_error(parsed: dict) -> dict:
    """Build an error/warning record."""
    return {
        "ts": parsed["ts"],
        "level": parsed["level"],
        "location": parsed["location"],
        "message": parsed["body"][:200],
    }


def _extract_generic(parsed: dict) -> dict:
    """Build a generic record (kill_switch, starvation, config_changes)."""
    return {
        "ts": parsed["ts"],
        "message": parsed["body"][:200],
        **parsed["kv"],
    }


def _cap(lst: list, limit: int = MAX_PER_CATEGORY) -> list:
    """Keep only the most recent *limit* entries."""
    return lst[-limit:]


def extract_events(
    log_path: str = "logs/autotrader.log",
    hours: int = 24,
) -> dict[str, list[dict]]:
    """Parse the autotrader log and extract meaningful events.

    Args:
        log_path: Path to the loguru log file.
        hours: How many hours back to look (default 24).

    Returns:
        Dict keyed by category name, each value is a list of dicts
        describing the extracted events (capped at 20 per category).
    """
    result: dict[str, list[dict]] = {
        "entries": [],
        "exits": [],
        "regimes": [],
        "blocked": [],
        "errors": [],
        "kill_switch": [],
        "starvation": [],
        "config_changes": [],
    }

    path = Path(log_path)
    if not path.exists():
        logger.warning("overseer_log_not_found | path={}", log_path)
        return result

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    # Also build a naive cutoff for comparison (logs may not have tz info)
    cutoff_naive = datetime.now() - timedelta(hours=hours)

    logger.info(
        "overseer_extracting_logs | path={} hours={} cutoff={}",
        log_path, hours, cutoff.isoformat(),
    )

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parsed = _parse_line(line)
            if parsed is None:
                continue

            # Time filter -- accept if timestamp is after cutoff
            try:
                line_ts = datetime.strptime(parsed["ts"], "%Y-%m-%d %H:%M:%S.%f")
                if line_ts < cutoff_naive:
                    continue
            except ValueError:
                continue

            body = parsed["body"]
            level = parsed["level"]

            # Categorise the line
            categorised = False
            for cat, pattern in _CATEGORY_PATTERNS.items():
                if pattern.search(body):
                    if cat == "entries":
                        result["entries"].append(_extract_entry(parsed))
                    elif cat == "exits":
                        result["exits"].append(_extract_exit(parsed))
                    elif cat == "regimes":
                        result["regimes"].append(_extract_regime(parsed))
                    elif cat == "blocked":
                        result["blocked"].append(_extract_blocked(parsed))
                    elif cat in ("kill_switch", "starvation", "config_changes"):
                        result[cat].append(_extract_generic(parsed))
                    categorised = True
                    break

            # Errors / warnings (not already categorised, skip noise)
            if not categorised and level in ("ERROR", "WARNING"):
                if not _NOISE_RE.search(body):
                    result["errors"].append(_extract_error(parsed))

    # Cap each category to most recent MAX_PER_CATEGORY
    for cat in result:
        result[cat] = _cap(result[cat])

    counts = {k: len(v) for k, v in result.items()}
    logger.info("overseer_extraction_complete | counts={}", counts)

    return result
