"""Persistent change memory helpers for the overseer."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from journal.db import TradeDatabase

CHANGESET_JSON_START = "CHANGESET_JSON_START"
CHANGESET_JSON_END = "CHANGESET_JSON_END"
MIN_TRADES_AFTER = 10
MIN_WIN_RATE_DELTA = 0.05
RECENT_CHANGE_LOOKBACK_DAYS = 60

TRACKED_EXTENSIONS = {
    ".py",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".txt",
    ".html",
    ".css",
    ".js",
}

IGNORED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "logs",
}

IGNORED_RELATIVE_PREFIXES = (
    "data/",
    "logs/",
    "overseer/reports/",
)

IGNORED_RELATIVE_PATHS = {
    "overseer/change_log.json",
    "overseer/strategy_log.json",
    "weekly_bias.json",
}


def ensure_memory_files(
    change_log_path: str | Path = "overseer/change_log.json",
    strategy_log_path: str | Path = "overseer/strategy_log.json",
) -> None:
    """Create empty memory files when missing."""
    for raw_path in (change_log_path, strategy_log_path):
        path = Path(raw_path)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]\n", encoding="utf-8")
        logger.info("overseer_memory_file_created | path={}", path)


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON list from disk, returning an empty list on failure."""
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.warning("overseer_memory_read_failed | path={} err={}", file_path, exc)
        return []

    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]

    logger.warning("overseer_memory_invalid_shape | path={} type={}", file_path, type(data).__name__)
    return []


def save_json_list(path: str | Path, entries: list[dict[str, Any]]) -> None:
    """Persist a JSON list to disk."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def save_json_object(path: str | Path, data: dict[str, Any]) -> None:
    """Persist a JSON object to disk."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk, returning an empty mapping on failure."""
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("overseer_json_read_failed | path={} err={}", file_path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def capture_repo_snapshot(root: str | Path = ".") -> dict[str, dict[str, str]]:
    """Capture hashes and contents of tracked text files for diffing/restores."""
    root_path = Path(root)
    snapshot: dict[str, dict[str, str]] = {}
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root_path).as_posix()
        if _is_ignored_path(rel_path):
            continue
        try:
            raw_content = path.read_bytes()
            content = raw_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.warning("overseer_snapshot_decode_failed | path={} err={}", path, exc)
            continue
        except OSError as exc:
            logger.warning("overseer_snapshot_read_failed | path={} err={}", path, exc)
            continue
        snapshot[rel_path] = {
            "sha256": hashlib.sha256(raw_content).hexdigest(),
            "content": content,
        }
    logger.info("overseer_snapshot_captured | files={}", len(snapshot))
    return snapshot


def diff_repo_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return file-level additions, removals, and modifications."""
    changes: list[dict[str, Any]] = []
    all_paths = sorted(set(before) | set(after))
    for rel_path in all_paths:
        old_hash = _snapshot_hash(before.get(rel_path))
        new_hash = _snapshot_hash(after.get(rel_path))
        if old_hash == new_hash:
            continue

        if old_hash is None:
            change_type = "file_added"
        elif new_hash is None:
            change_type = "file_removed"
        else:
            change_type = "file_modified"

        changes.append({
            "parameter": f"file:{rel_path}",
            "old_value": old_hash,
            "new_value": new_hash,
            "change_type": change_type,
        })
    return changes


def restore_repo_snapshot(
    snapshot: dict[str, Any],
    *,
    root: str | Path = ".",
    paths: list[str] | None = None,
) -> list[str]:
    """Restore tracked files from a previously captured snapshot."""
    root_path = Path(root)
    restored: list[str] = []
    target_paths = paths if paths is not None else sorted(snapshot)
    for rel_path in target_paths:
        file_path = root_path / rel_path
        entry = snapshot.get(rel_path)
        if entry is None:
            if file_path.exists():
                file_path.unlink()
                restored.append(rel_path)
            continue

        content = _snapshot_content(entry)
        if content is None:
            continue

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        restored.append(rel_path)
    return restored


def diff_config_values(
    before: dict[str, Any],
    after: dict[str, Any],
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Compute a recursive scalar diff between two config dicts."""
    changes: list[dict[str, Any]] = []
    keys = sorted(set(before) | set(after))
    for key in keys:
        param = f"{prefix}.{key}" if prefix else key
        before_has = key in before
        after_has = key in after

        if not before_has:
            changes.append({
                "parameter": param,
                "old_value": None,
                "new_value": after[key],
                "change_type": "config_added",
            })
            continue

        if not after_has:
            changes.append({
                "parameter": param,
                "old_value": before[key],
                "new_value": None,
                "change_type": "config_removed",
            })
            continue

        old_value = before[key]
        new_value = after[key]
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            changes.extend(diff_config_values(old_value, new_value, param))
        elif old_value != new_value:
            changes.append({
                "parameter": param,
                "old_value": old_value,
                "new_value": new_value,
                "change_type": "config_modified",
            })
    return changes


def extract_changeset_metadata(report: str) -> dict[str, dict[str, Any]]:
    """Parse the optional machine-readable change metadata from the report."""
    match = re.search(
        rf"{CHANGESET_JSON_START}\s*(.*?)\s*{CHANGESET_JSON_END}",
        report,
        flags=re.DOTALL,
    )
    if not match:
        return {}

    raw_json = match.group(1).strip()
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning("overseer_changeset_parse_failed | err={}", exc)
        return {}

    items: list[dict[str, Any]]
    if isinstance(parsed, list):
        items = [item for item in parsed if isinstance(item, dict)]
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("changes"), list):
            items = [item for item in parsed["changes"] if isinstance(item, dict)]
        else:
            items = [parsed]
    else:
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    for item in items:
        parameter = item.get("parameter")
        if isinstance(parameter, str) and parameter:
            metadata[parameter] = item
    return metadata


def summarize_change_log(entries: list[dict[str, Any]]) -> str:
    """Summarize change outcomes for overseer context."""
    if not entries:
        return "  (none)"

    applied = [entry for entry in entries if entry.get("type", "applied") == "applied"]
    blocked = [entry for entry in entries if entry.get("type") == "blocked"]
    improved = [entry for entry in applied if entry.get("outcome") == "improved"]
    degraded = [entry for entry in applied if entry.get("outcome") == "degraded"]
    neutral = [entry for entry in applied if entry.get("outcome") == "neutral"]
    insufficient = [entry for entry in applied if entry.get("outcome") == "insufficient_data"]
    pending = [entry for entry in applied if entry.get("outcome") in (None, "pending")]
    lines = [
        f"  applied_changes={len(applied)}",
        f"  blocked_attempts={len(blocked)}",
        f"  improved={len(improved)}",
        f"  degraded={len(degraded)}",
        f"  neutral={len(neutral)}",
        f"  insufficient_data={len(insufficient)}",
        f"  pending_review={len(pending)}",
    ]

    if blocked:
        blocked_counts: dict[str, int] = {}
        for entry in blocked:
            parameter = str(entry.get("parameter", "unknown"))
            blocked_counts[parameter] = blocked_counts.get(parameter, 0) + 1
        lines.append("  most_contested_parameters:")
        for parameter, count in sorted(blocked_counts.items(), key=lambda item: (-item[1], item[0]))[:5]:
            lines.append(f"    {parameter}: {count} blocked attempts")

    recent_degraded = _recent_entries(degraded, days=30)
    if recent_degraded:
        lines.append("  recent_degraded_last_30d:")
        for entry in recent_degraded[-5:]:
            lines.append(
                "    "
                + _format_change_summary(
                    entry,
                    default_reason="avoid repeating or reverting without new evidence",
                )
            )

    recent_improved = _recent_entries(improved, days=30)
    if recent_improved:
        lines.append("  recent_improved_last_30d:")
        for entry in recent_improved[-5:]:
            lines.append(
                "    "
                + _format_change_summary(
                    entry,
                    default_reason="consider extending only with supporting evidence",
                )
            )

    return "\n".join(lines)


def check_proposed_change(
    *,
    parameter: str,
    old_value: Any,
    new_value: Any,
    change_entries: list[dict[str, Any]] | None = None,
    change_log_path: str | Path = "overseer/change_log.json",
    as_of: datetime | None = None,
    lookback_days: int = RECENT_CHANGE_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """Decide whether a config change is allowed by recent change history."""
    now = as_of or datetime.now(timezone.utc)
    entries = change_entries if change_entries is not None else load_json_list(change_log_path)
    recent_same_param = []
    for entry in entries:
        if entry.get("type", "applied") != "applied":
            continue
        if entry.get("parameter") != parameter:
            continue
        entry_ts = _parse_iso_timestamp(
            entry.get("timestamp") or entry.get("run_timestamp") or entry.get("date")
        )
        if entry_ts is None:
            continue
        if now - entry_ts <= timedelta(days=lookback_days):
            recent_same_param.append(entry)

    recent_same_param.sort(
        key=lambda item: item.get("timestamp") or item.get("run_timestamp") or item.get("date") or ""
    )
    degraded = [entry for entry in recent_same_param if entry.get("outcome") == "degraded"]

    for entry in reversed(degraded):
        if _values_equal(new_value, entry.get("new_value")):
            return {
                "allowed": False,
                "requires_justification": False,
                "reason": (
                    f"Blocked {parameter}: value {new_value!r} previously degraded on "
                    f"{entry.get('date') or entry.get('timestamp')}"
                ),
                "policy": "blocked_previous_degraded_value",
                "matched_entry": entry,
            }

    if degraded:
        latest = degraded[-1]
        return {
            "allowed": True,
            "requires_justification": True,
            "reason": (
                f"{parameter} changed recently with degraded outcome on "
                f"{latest.get('date') or latest.get('timestamp')}; require market context."
            ),
            "policy": "requires_market_context_after_recent_degraded_change",
            "matched_entry": latest,
        }

    recent_unresolved = [
        entry for entry in recent_same_param
        if entry.get("outcome") in (None, "pending", "insufficient_data")
    ]
    for entry in reversed(recent_unresolved):
        if _values_equal(new_value, entry.get("old_value")) or _values_equal(new_value, entry.get("new_value")):
            return {
                "allowed": True,
                "requires_justification": True,
                "reason": (
                    f"{parameter} already changed recently without a conclusive outcome; "
                    "require market context to avoid oscillation."
                ),
                "policy": "requires_market_context_due_to_recent_unresolved_change",
                "matched_entry": entry,
            }

    return {
        "allowed": True,
        "requires_justification": False,
        "reason": "No blocking recent history for this parameter.",
        "policy": "allowed",
        "matched_entry": None,
    }


def record_blocked_change_attempts(
    *,
    blocked_changes: list[dict[str, Any]],
    report: str,
    run_timestamp: str,
    run_id: str,
    change_log_path: str | Path = "overseer/change_log.json",
) -> int:
    """Append blocked change attempts to the persistent audit log."""
    if not blocked_changes:
        return 0

    entries = load_json_list(change_log_path)
    metadata = extract_changeset_metadata(report)
    created_at = datetime.now(timezone.utc).isoformat()
    created = 0
    for change in blocked_changes:
        parameter = str(change.get("parameter"))
        meta = metadata.get(parameter, {})
        if _upsert_blocked_entry(
            entries=entries,
            run_timestamp=run_timestamp,
            run_id=run_id,
            created_at=created_at,
            parameter=parameter,
            proposed_value=change.get("new_value"),
            old_value=change.get("old_value"),
            block_reason=str(change.get("reason", "Blocked by guard.")),
            market_context_provided=_has_text(meta.get("market_context")),
            change_type=change.get("change_type"),
        ):
            created += 1

    if created:
        save_json_list(change_log_path, entries)
    return created


def enforce_config_change_guard(
    *,
    before_config: dict[str, Any],
    after_config: dict[str, Any],
    report: str,
    change_entries: list[dict[str, Any]] | None = None,
    change_log_path: str | Path = "overseer/change_log.json",
    config_path: str | Path | None = "config.json",
    as_of: datetime | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Revert blocked config changes and require justification when needed."""
    metadata = extract_changeset_metadata(report)
    recent_entries = change_entries if change_entries is not None else load_json_list(change_log_path)
    guarded_config = copy.deepcopy(after_config)
    blocked_changes: list[dict[str, Any]] = []

    for change in diff_config_values(before_config, after_config):
        decision = check_proposed_change(
            parameter=change["parameter"],
            old_value=change["old_value"],
            new_value=change["new_value"],
            change_entries=recent_entries,
            as_of=as_of,
        )
        meta = metadata.get(change["parameter"], {})
        market_context = meta.get("market_context")
        if decision["requires_justification"] and not _has_text(market_context):
            decision = {
                **decision,
                "allowed": False,
                "reason": (
                    f"{decision['reason']} Missing required market_context in CHANGESET_JSON."
                ),
                "policy": "blocked_missing_market_context",
            }

        if not decision["allowed"]:
            _restore_config_value(
                cfg=guarded_config,
                parameter=change["parameter"],
                old_value=change["old_value"],
            )
            blocked_changes.append({
                **change,
                **decision,
            })

    if blocked_changes and config_path is not None:
        save_json_object(config_path, guarded_config)
        logger.warning(
            "overseer_config_guard_reverted_changes | count={} params={}",
            len(blocked_changes),
            [change["parameter"] for change in blocked_changes],
        )

    return guarded_config, blocked_changes


def summarize_strategy_log(entries: list[dict[str, Any]]) -> str:
    """Summarize strategy enable/disable memory for overseer context."""
    if not entries:
        return "  (none)"

    latest_by_strategy: dict[str, dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda item: item.get("timestamp", "")):
        strategy = entry.get("strategy")
        if isinstance(strategy, str) and strategy:
            latest_by_strategy[strategy] = entry

    disabled = [
        entry for entry in latest_by_strategy.values()
        if entry.get("action") == "disabled"
    ]
    reenabled = [
        entry for entry in latest_by_strategy.values()
        if entry.get("action") == "enabled"
    ]

    lines = [
        f"  total_strategy_events={len(entries)}",
        f"  currently_disabled_by_memory={len(disabled)}",
        f"  currently_enabled_after_memory={len(reenabled)}",
    ]
    for entry in sorted(disabled, key=lambda item: item.get("timestamp", ""))[-5:]:
        reason = entry.get("reason") or "no reason recorded"
        lines.append(
            f"  disabled {entry.get('strategy')} on {entry.get('date')} because {reason}"
        )
    return "\n".join(lines)


def reconcile_run_memory(
    *,
    run_timestamp: str,
    run_id: str,
    report: str,
    db_path: str = "data/trades.db",
    config_path: str = "config.json",
    repo_root: str | Path = ".",
    before_snapshot: dict[str, Any] | None = None,
    after_snapshot: dict[str, Any] | None = None,
    before_config: dict[str, Any] | None = None,
    after_config: dict[str, Any] | None = None,
    change_log_path: str | Path = "overseer/change_log.json",
    strategy_log_path: str | Path = "overseer/strategy_log.json",
) -> dict[str, int]:
    """Reconcile log memory with changes observed in the latest overseer run."""
    ensure_memory_files(change_log_path, strategy_log_path)
    change_entries = load_json_list(change_log_path)
    strategy_entries = load_json_list(strategy_log_path)
    metadata = extract_changeset_metadata(report)

    before_cfg = before_config if before_config is not None else read_json_object(config_path)
    after_cfg = after_config if after_config is not None else read_json_object(config_path)
    cfg_changes = diff_config_values(before_cfg, after_cfg)

    before_rate, before_trades = compute_window_win_rate(
        db_path=db_path,
        window_end=_parse_iso_timestamp(run_timestamp),
        days=7,
    )

    current_run_ts = datetime.now(timezone.utc).isoformat()

    created_change_entries = 0
    for change in cfg_changes:
        parameter = change["parameter"]
        meta = metadata.get(parameter, {})
        inserted = _upsert_change_entry(
            entries=change_entries,
            run_timestamp=run_timestamp,
            run_id=run_id,
            created_at=current_run_ts,
            parameter=parameter,
            old_value=change["old_value"],
            new_value=change["new_value"],
            reason=_resolve_reason(
                meta,
                fallback=f"Overseer changed {parameter} during run {run_timestamp}.",
            ),
            market_context=meta.get("market_context"),
            win_rate_before=before_rate,
            trades_before=before_trades,
            change_type=change["change_type"],
        )
        if inserted:
            created_change_entries += 1

        if _is_strategy_toggle(parameter):
            strategy_name = parameter.split(".")[1]
            action = "enabled" if change["new_value"] else "disabled"
            inserted = _upsert_strategy_entry(
                entries=strategy_entries,
                run_timestamp=run_timestamp,
                run_id=run_id,
                created_at=current_run_ts,
                strategy=strategy_name,
                action=action,
                old_enabled=change["old_value"],
                new_enabled=change["new_value"],
                reason=_resolve_reason(
                    meta,
                    fallback=f"Overseer {action} strategy {strategy_name} during run {run_timestamp}.",
                ),
            )
            if inserted:
                logger.info(
                    "overseer_strategy_memory_recorded | strategy={} action={}",
                    strategy_name,
                    action,
                )

    before_repo = before_snapshot if before_snapshot is not None else capture_repo_snapshot(repo_root)
    after_repo = after_snapshot if after_snapshot is not None else capture_repo_snapshot(repo_root)
    repo_changes = diff_repo_snapshots(before_repo, after_repo)
    created_code_entries = 0
    for change in repo_changes:
        if change["parameter"] == "file:config.json":
            continue
        meta = metadata.get(change["parameter"], {})
        inserted = _upsert_change_entry(
            entries=change_entries,
            run_timestamp=run_timestamp,
            run_id=run_id,
            created_at=current_run_ts,
            parameter=change["parameter"],
            old_value=change["old_value"],
            new_value=change["new_value"],
            reason=_resolve_reason(
                meta,
                fallback=f"Overseer modified {change['parameter']} during run {run_timestamp}.",
            ),
            market_context=meta.get("market_context"),
            win_rate_before=before_rate,
            trades_before=before_trades,
            change_type=change["change_type"],
        )
        if inserted:
            created_code_entries += 1

    save_json_list(change_log_path, change_entries)
    save_json_list(strategy_log_path, strategy_entries)

    logger.info(
        "overseer_run_memory_reconciled | config_entries={} code_entries={} strategy_events={}",
        created_change_entries,
        created_code_entries,
        len(strategy_entries),
    )
    return {
        "config_entries": created_change_entries,
        "code_entries": created_code_entries,
        "strategy_events": len(strategy_entries),
    }


def backfill_change_outcomes(
    db_path: str = "data/trades.db",
    change_log_path: str | Path = "overseer/change_log.json",
    as_of: datetime | None = None,
    min_trades_after: int = MIN_TRADES_AFTER,
) -> int:
    """Fill win_rate_after and outcome for changes older than seven days."""
    ensure_memory_files(change_log_path, "overseer/strategy_log.json")
    entries = load_json_list(change_log_path)
    if not entries:
        return 0

    now = as_of or datetime.now(timezone.utc)
    updated = 0
    for entry in entries:
        if entry.get("type", "applied") != "applied":
            continue
        if entry.get("outcome") not in (None, "pending"):
            continue

        entry_ts = _parse_iso_timestamp(entry.get("timestamp") or entry.get("run_timestamp") or entry.get("date"))
        if entry_ts is None or now < entry_ts + timedelta(days=7):
            continue

        after_rate, after_trades = compute_window_win_rate(
            db_path=db_path,
            window_end=entry_ts + timedelta(days=7),
            days=7,
            window_start=entry_ts,
        )

        entry["win_rate_after"] = after_rate
        entry["trades_after"] = after_trades
        if after_trades < min_trades_after:
            entry["outcome"] = "insufficient_data"
        else:
            before_rate = entry.get("win_rate_before")
            entry["outcome"] = _classify_outcome(before_rate, after_rate)
        updated += 1

    if updated:
        save_json_list(change_log_path, entries)
        logger.info("overseer_change_outcomes_backfilled | updated={}", updated)
    return updated


def compute_window_win_rate(
    *,
    db_path: str,
    window_end: datetime,
    days: int,
    window_start: datetime | None = None,
) -> tuple[float | None, int]:
    """Compute win rate for closed trades in a time window."""
    start = window_start or (window_end - timedelta(days=days))
    db = TradeDatabase(db_path)
    try:
        trades = db.get_closed_trades_for_period(start.isoformat(), window_end.isoformat())
    finally:
        db.close()

    closed = [trade for trade in trades if trade.get("outcome") and trade.get("outcome") != "open"]
    if not closed:
        return None, 0

    wins = sum(1 for trade in closed if trade.get("outcome") == "win")
    win_rate = wins / len(closed)
    return round(win_rate, 4), len(closed)


def _is_ignored_path(rel_path: str) -> bool:
    parts = rel_path.split("/")
    if any(part in IGNORED_DIR_NAMES for part in parts[:-1]):
        return True
    if rel_path in IGNORED_RELATIVE_PATHS:
        return True
    if any(rel_path.startswith(prefix) for prefix in IGNORED_RELATIVE_PREFIXES):
        return True
    return Path(rel_path).suffix.lower() not in TRACKED_EXTENSIONS


def _restore_config_value(*, cfg: dict[str, Any], parameter: str, old_value: Any) -> None:
    path = parameter.split(".")
    node = cfg
    for key in path[:-1]:
        if not isinstance(node, dict):
            return
        node = node.setdefault(key, {})

    if not isinstance(node, dict):
        return

    leaf = path[-1]
    if old_value is None:
        node.pop(leaf, None)
    else:
        node[leaf] = old_value


def _snapshot_hash(entry: Any) -> str | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        value = entry.get("sha256")
        return value if isinstance(value, str) else None
    return entry if isinstance(entry, str) else None


def _snapshot_content(entry: Any) -> str | None:
    if isinstance(entry, dict):
        value = entry.get("content")
        return value if isinstance(value, str) else None
    return None


def _recent_entries(entries: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent: list[dict[str, Any]] = []
    for entry in entries:
        entry_ts = _parse_iso_timestamp(entry.get("timestamp") or entry.get("run_timestamp") or entry.get("date"))
        if entry_ts and entry_ts >= cutoff:
            recent.append(entry)
    return recent


def _format_change_summary(entry: dict[str, Any], default_reason: str) -> str:
    reason = entry.get("reason") or default_reason
    return (
        f"{entry.get('parameter')} {entry.get('old_value')} -> {entry.get('new_value')} "
        f"on {entry.get('date')} ({reason})"
    )


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None

    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.strptime(normalized, "%Y-%m-%d")
        except ValueError:
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_reason(metadata: dict[str, Any], fallback: str) -> str:
    reason = metadata.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return fallback


def _is_strategy_toggle(parameter: str) -> bool:
    parts = parameter.split(".")
    return len(parts) == 3 and parts[0] == "strategies" and parts[2] == "enabled"


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _values_equal(left: Any, right: Any) -> bool:
    return left == right


def _upsert_change_entry(
    *,
    entries: list[dict[str, Any]],
    run_timestamp: str,
    run_id: str,
    created_at: str,
    parameter: str,
    old_value: Any,
    new_value: Any,
    reason: str,
    market_context: Any,
    win_rate_before: float | None,
    trades_before: int,
    change_type: str,
) -> bool:
    for entry in entries:
        if entry.get("run_timestamp") != run_timestamp:
            continue
        if entry.get("type", "applied") != "applied":
            continue
        if entry.get("parameter") != parameter:
            continue
        existing_old = entry.get("old_value")
        existing_new = entry.get("new_value")
        if existing_old not in (None, old_value):
            continue
        if existing_new not in (None, new_value):
            continue
        entry.setdefault("date", created_at[:10])
        entry.setdefault("timestamp", created_at)
        if existing_old is None:
            entry["old_value"] = old_value
        if existing_new is None:
            entry["new_value"] = new_value
        entry.setdefault("overseer_run_id", run_id)
        entry.setdefault("type", "applied")
        if not entry.get("reason"):
            entry["reason"] = reason
        if _has_text(market_context) and not entry.get("market_context"):
            entry["market_context"] = market_context
        if entry.get("win_rate_before") is None:
            entry["win_rate_before"] = win_rate_before
        entry.setdefault("win_rate_after", None)
        entry.setdefault("outcome", None)
        if not entry.get("change_type"):
            entry["change_type"] = change_type
        if entry.get("trades_before") is None:
            entry["trades_before"] = trades_before
        return False

    entries.append({
        "date": created_at[:10],
        "timestamp": created_at,
        "run_timestamp": run_timestamp,
        "overseer_run_id": run_id,
        "type": "applied",
        "parameter": parameter,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason,
        "market_context": market_context if _has_text(market_context) else None,
        "win_rate_before": win_rate_before,
        "win_rate_after": None,
        "outcome": None,
        "change_type": change_type,
        "trades_before": trades_before,
    })
    logger.info("overseer_change_memory_recorded | parameter={} type={}", parameter, change_type)
    return True


def _upsert_strategy_entry(
    *,
    entries: list[dict[str, Any]],
    run_timestamp: str,
    run_id: str,
    created_at: str,
    strategy: str,
    action: str,
    old_enabled: Any,
    new_enabled: Any,
    reason: str,
) -> bool:
    for entry in entries:
        if entry.get("run_timestamp") == run_timestamp and entry.get("strategy") == strategy and entry.get("action") == action:
            entry.setdefault("date", created_at[:10])
            entry.setdefault("timestamp", created_at)
            entry.setdefault("overseer_run_id", run_id)
            if not entry.get("reason"):
                entry["reason"] = reason
            if entry.get("old_enabled") is None:
                entry["old_enabled"] = old_enabled
            if entry.get("new_enabled") is None:
                entry["new_enabled"] = new_enabled
            return False

    entries.append({
        "date": created_at[:10],
        "timestamp": created_at,
        "run_timestamp": run_timestamp,
        "overseer_run_id": run_id,
        "strategy": strategy,
        "action": action,
        "old_enabled": old_enabled,
        "new_enabled": new_enabled,
        "reason": reason,
    })
    return True


def _upsert_blocked_entry(
    *,
    entries: list[dict[str, Any]],
    run_timestamp: str,
    run_id: str,
    created_at: str,
    parameter: str,
    proposed_value: Any,
    old_value: Any,
    block_reason: str,
    market_context_provided: bool,
    change_type: Any,
) -> bool:
    for entry in entries:
        if entry.get("type") != "blocked":
            continue
        if entry.get("run_timestamp") != run_timestamp:
            continue
        if entry.get("parameter") != parameter:
            continue
        if entry.get("proposed_value") != proposed_value:
            continue
        entry.setdefault("date", created_at[:10])
        entry.setdefault("timestamp", created_at)
        entry.setdefault("overseer_run_id", run_id)
        entry.setdefault("block_reason", block_reason)
        entry.setdefault("market_context_provided", market_context_provided)
        if entry.get("old_value") is None:
            entry["old_value"] = old_value
        if not entry.get("change_type"):
            entry["change_type"] = change_type
        return False

    entries.append({
        "date": created_at[:10],
        "timestamp": created_at,
        "run_timestamp": run_timestamp,
        "overseer_run_id": run_id,
        "type": "blocked",
        "parameter": parameter,
        "old_value": old_value,
        "proposed_value": proposed_value,
        "block_reason": block_reason,
        "market_context_provided": market_context_provided,
        "change_type": change_type,
    })
    return True


def _classify_outcome(before_rate: Any, after_rate: float) -> str:
    if before_rate is None:
        return "neutral"

    try:
        before = float(before_rate)
    except (TypeError, ValueError):
        return "neutral"

    delta = after_rate - before
    if delta > MIN_WIN_RATE_DELTA:
        return "improved"
    if delta < -MIN_WIN_RATE_DELTA:
        return "degraded"
    return "neutral"
