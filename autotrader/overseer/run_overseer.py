"""VarpTrader overseer runner.

Standalone script that builds context from logs, database, and config,
then invokes Claude Code with a diagnostic/action prompt. Saves the
resulting report to overseer/reports/YYYY-MM-DD_HH.txt.

Usage:
    cd autotrader

    # Nightly fast scan - Sonnet (default)
    ANTHROPIC_MODEL=claude-sonnet-4-6 python -m overseer.run_overseer

    # Weekly deep refactor - Opus
    ANTHROPIC_MODEL=claude-opus-4-6 python -m overseer.run_overseer --deep

    # Context only (for host wrapper scripts)
    python -m overseer.run_overseer --context-only

Models (set via ANTHROPIC_MODEL env var):
    claude-sonnet-4-6   Fast, cheap - good for nightly checks (default)
    claude-opus-4-6     Thorough, expensive - good for weekly deep refactor
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from loguru import logger
from dotenv import load_dotenv

from alerts.telegram_bot import TelegramAlert

from overseer.change_memory import (
    CHANGESET_JSON_END,
    CHANGESET_JSON_START,
    backfill_change_outcomes,
    capture_repo_snapshot,
    enforce_config_change_guard,
    ensure_memory_files,
    load_json_list,
    read_json_object,
    record_blocked_change_attempts,
    reconcile_run_memory,
    restore_repo_snapshot,
    save_json_object,
    diff_repo_snapshots,
)
from overseer.context_builder import build_context

REPORTS_DIR = Path("overseer/reports")
HOST_RUN_STATE_DIR = Path("logs/overseer_state/host_runs")

_DEFAULT_MODEL = "claude-sonnet-4-6"
_CLAUDE_AUTH_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)

load_dotenv()


def _resolve_claude_auth_mode() -> str:
    """Return the Claude auth mode for overseer runs."""
    mode = os.getenv("VARPTRADER_CLAUDE_AUTH_MODE", "login").strip().lower()
    return mode if mode in {"login", "api", "auto"} else "login"


def _build_claude_env() -> dict[str, str]:
    """Build the Claude subprocess environment."""
    env = os.environ.copy()
    if _resolve_claude_auth_mode() == "login":
        for key in _CLAUDE_AUTH_ENV_KEYS:
            env.pop(key, None)
    return env


def _normalize_claude_error(report: str) -> str:
    """Convert raw Claude auth failures into actionable errors."""
    if _resolve_claude_auth_mode() == "login" and "not logged in" in report.lower():
        return (
            "[Overseer Error] Claude login mode is enabled, but this environment is not "
            "logged in. Run `claude` and complete `/login` for the user that executes "
            "the overseer."
        )
    return report


def _format_value(value: object, max_len: int = 60) -> str:
    """Render compact values for Telegram summaries."""
    text = json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= max_len else f"{text[: max_len - 3]}..."


def _trim_report_line(report: str, max_len: int = 220) -> str:
    """Extract a short human-readable headline from the overseer report."""
    for raw_line in report.splitlines():
        line = raw_line.strip().lstrip("#*- ").strip()
        if not line or line == CHANGESET_JSON_START or line == CHANGESET_JSON_END:
            continue
        return line if len(line) <= max_len else f"{line[: max_len - 3]}..."
    return "No report summary returned."


def _format_change_entry(entry: dict[str, object]) -> str:
    """Format one applied change for Telegram."""
    parameter = str(entry.get("parameter", "?"))
    change_type = str(entry.get("change_type", "") or "")
    if parameter.startswith("file:"):
        suffix = f" ({change_type})" if change_type else ""
        return f"- {parameter}{suffix}"
    return (
        f"- {parameter}: "
        f"{_format_value(entry.get('old_value'))} -> {_format_value(entry.get('new_value'))}"
    )


def _format_blocked_entry(entry: dict[str, object]) -> str:
    """Format one blocked change for Telegram."""
    reason = str(entry.get("block_reason", "blocked"))[:90]
    return (
        f"- {entry.get('parameter', '?')} -> {_format_value(entry.get('proposed_value'))} "
        f"({reason})"
    )


def _format_strategy_entry(entry: dict[str, object]) -> str:
    """Format one strategy log entry for Telegram."""
    return f"- {entry.get('strategy', '?')}: {entry.get('action', '?')}"


def _build_telegram_summary(
    *,
    run_id: str,
    trigger_reason: str,
    deep: bool,
    report: str,
    change_entries: list[dict[str, object]],
    strategy_entries: list[dict[str, object]],
) -> str:
    """Build a concise Telegram message summarizing the overseer run."""
    applied = [entry for entry in change_entries if entry.get("type", "applied") == "applied"]
    blocked = [entry for entry in change_entries if entry.get("type") == "blocked"]

    mode = "DEEP" if deep else "NIGHTLY"
    lines = [
        "OVERSEER UPDATE",
        "",
        f"Mode: {mode} | Trigger: {trigger_reason}",
        f"Run ID: {run_id}",
        f"Applied: {len(applied)} | Blocked: {len(blocked)} | Strategy events: {len(strategy_entries)}",
    ]

    if applied:
        lines.append("")
        lines.append("Applied:")
        for entry in applied[:3]:
            lines.append(_format_change_entry(entry))
        if len(applied) > 3:
            lines.append(f"- ... and {len(applied) - 3} more")

    if blocked:
        lines.append("")
        lines.append("Blocked:")
        for entry in blocked[:2]:
            lines.append(_format_blocked_entry(entry))
        if len(blocked) > 2:
            lines.append(f"- ... and {len(blocked) - 2} more")

    if strategy_entries:
        lines.append("")
        lines.append("Strategies:")
        for entry in strategy_entries[:3]:
            lines.append(_format_strategy_entry(entry))
        if len(strategy_entries) > 3:
            lines.append(f"- ... and {len(strategy_entries) - 3} more")

    lines.append("")
    lines.append(f"Summary: {_trim_report_line(report)}")
    return "\n".join(lines)


def _send_telegram_summary(
    *,
    run_id: str,
    trigger_reason: str,
    deep: bool,
    report: str,
    change_log_path: str = "overseer/change_log.json",
    strategy_log_path: str = "overseer/strategy_log.json",
) -> None:
    """Send the overseer result summary to Telegram when configured."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        logger.info("overseer_telegram_skipped | reason=missing_credentials")
        return

    change_entries = [
        entry for entry in load_json_list(change_log_path)
        if entry.get("overseer_run_id") == run_id
    ]
    strategy_entries = [
        entry for entry in load_json_list(strategy_log_path)
        if entry.get("overseer_run_id") == run_id
    ]
    summary = _build_telegram_summary(
        run_id=run_id,
        trigger_reason=trigger_reason,
        deep=deep,
        report=report,
        change_entries=change_entries,
        strategy_entries=strategy_entries,
    )
    TelegramAlert(bot_token=bot_token, chat_id=chat_id).send_message(
        summary,
        parse_mode="",
    )


def _build_memory_rules(run_timestamp: str, run_id: str) -> str:
    """Return shared change-memory rules for Claude."""
    return f"""=== CHANGE MEMORY RULES ===

RUN_TIMESTAMP: {run_timestamp}
RUN_ID: {run_id}

Before suggesting or applying any config/code change:
- Read `overseer/change_log.json` and `overseer/strategy_log.json` from the provided context.
- If the same parameter was changed in the last 60 days and the recorded outcome was `degraded`, do not repeat that failed value.
- If the same parameter improved performance, prefer extending that direction cautiously instead of undoing it.
- If a strategy was deliberately disabled in `strategy_log.json`, do not re-enable it unless you have explicit evidence of regime change and you record the reason.

Before editing `config.json`, any tracked code file, or changing any strategy enable flag:
- For every config change, run:
  `python -m overseer.check_proposed_change --parameter "<param>" --old-json "<json>" --new-json "<json>"`
- Append the planned entry to `overseer/change_log.json` first using this schema:
  `date`, `timestamp`, `run_timestamp`, `overseer_run_id`, `type`, `parameter`, `old_value`, `new_value`, `reason`, `market_context`, `win_rate_before`, `win_rate_after`, `outcome`
- If you enable/disable a strategy, also append to `overseer/strategy_log.json` with:
  `date`, `timestamp`, `run_timestamp`, `overseer_run_id`, `strategy`, `action`, `reason`

HARD RULES - NEVER violate:
- Before ANY config change, call the guard command above.
- If the guard output says `"allowed": false`, DO NOT make the change.
- If the guard output says `"requires_justification": true`, you MUST include `market_context` in CHANGESET_JSON.

At the end of your report, include a machine-readable JSON array between these exact markers:
{CHANGESET_JSON_START}
[{{"parameter": "strategies.rsi_momentum.rsi_overbought", "reason": "example reason", "market_context": "example regime change"}}]
{CHANGESET_JSON_END}

List every config or code change you actually made in that JSON block. Use `file:relative/path.py` for code-file edits.
"""


def _append_guard_summary(
    report: str,
    blocked_changes: list[dict[str, object]],
    restored_paths: list[str],
) -> str:
    """Append a short guard summary to the overseer report."""
    if not blocked_changes:
        return report

    lines = [
        "",
        "[Overseer Guard]",
        "Reverted blocked config changes:",
    ]
    for change in blocked_changes:
        lines.append(f"- {change['parameter']}: {change['reason']}")
    if restored_paths:
        lines.append(f"Restored tracked files from pre-run snapshot: {len(restored_paths)}")
    summary = "\n".join(lines)
    return f"{report}\n{summary}" if report else summary


def _report_filename_for_run(*, deep: bool, when: datetime | None = None) -> Path:
    """Return the standard report filename for a run timestamp."""
    stamp = when or datetime.now(timezone.utc)
    suffix = "_deep" if deep else ""
    return REPORTS_DIR / f"{stamp.strftime('%Y-%m-%d_%H')}{suffix}.txt"


def _finalize_run_outputs(
    *,
    deep: bool,
    resolved_model: str,
    resolved_trigger: str,
    run_timestamp: str,
    run_id: str,
    report: str,
    history_before: list[dict[str, object]],
    before_snapshot: dict[str, object],
    before_config: dict[str, object],
) -> dict[str, object]:
    """Apply guard/reconcile/report persistence for a completed Claude run."""
    post_claude_snapshot = capture_repo_snapshot(".")
    raw_after_config = read_json_object("config.json")
    if not raw_after_config and before_config:
        guarded_config = before_config
        save_json_object("config.json", guarded_config)
        blocked_changes = [
            {
                "parameter": "config.json",
                "reason": "Config became unreadable after overseer run; restored the pre-run snapshot.",
                "policy": "blocked_invalid_config",
            }
        ]
    else:
        guarded_config, blocked_changes = enforce_config_change_guard(
            before_config=before_config,
            after_config=raw_after_config,
            report=report,
            change_entries=history_before,
            config_path="config.json",
        )
    restored_paths: list[str] = []
    if blocked_changes:
        changed_paths = [
            change["parameter"][5:]
            for change in diff_repo_snapshots(before_snapshot, post_claude_snapshot)
            if str(change.get("parameter", "")).startswith("file:")
        ]
        restored_paths = restore_repo_snapshot(
            before_snapshot,
            paths=changed_paths,
        )
        after_snapshot = capture_repo_snapshot(".")
        after_config = read_json_object("config.json")
        record_blocked_change_attempts(
            blocked_changes=blocked_changes,
            report=report,
            run_timestamp=run_timestamp,
            run_id=run_id,
        )
        report = _append_guard_summary(report, blocked_changes, restored_paths)
    else:
        after_snapshot = post_claude_snapshot
        after_config = guarded_config

    reconcile_run_memory(
        run_timestamp=run_timestamp,
        run_id=run_id,
        report=report,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        before_config=before_config,
        after_config=after_config,
    )
    backfill_change_outcomes()

    report_path = _report_filename_for_run(deep=deep)
    report_path.write_text(report, encoding="utf-8")
    logger.info("overseer_report_saved | path={} model={}", report_path, resolved_model)
    _send_telegram_summary(
        run_id=run_id,
        trigger_reason=resolved_trigger,
        deep=deep,
        report=report,
    )
    return {
        "report": report,
        "report_path": str(report_path),
        "blocked_changes": len(blocked_changes),
        "restored_paths": len(restored_paths),
    }


def prepare_host_run(
    *,
    deep: bool = False,
    trigger_reason: str | None = None,
    model: str | None = None,
) -> dict[str, object]:
    """Prepare prompt and snapshots for a host-side Claude run."""
    resolved_model = model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)
    resolved_trigger = trigger_reason or ("deep_weekly" if deep else "nightly")

    ensure_memory_files()
    backfill_change_outcomes()

    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    history_before = load_json_list("overseer/change_log.json")
    before_snapshot = capture_repo_snapshot(".")
    before_config = read_json_object("config.json")
    context = build_context()
    prompt = (
        _build_prompt_deep(context, run_timestamp, run_id)
        if deep
        else _build_prompt_nightly(context, run_timestamp, run_id)
    )

    HOST_RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = HOST_RUN_STATE_DIR / f"{run_id}.json"
    raw_report_file = REPORTS_DIR / f"{run_id}_host_raw.txt"
    state_payload = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "trigger_reason": resolved_trigger,
        "model": resolved_model,
        "deep": deep,
        "history_before": history_before,
        "before_snapshot": before_snapshot,
        "before_config": before_config,
        "raw_report_file": raw_report_file.resolve().as_posix(),
    }
    state_file.write_text(json.dumps(state_payload), encoding="utf-8")
    return {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "trigger_reason": resolved_trigger,
        "model": resolved_model,
        "deep": deep,
        "prompt": prompt,
        "state_file": state_file.resolve().as_posix(),
        "raw_report_file": raw_report_file.resolve().as_posix(),
    }


def finalize_host_run(
    *,
    state_file: str,
    report_file: str,
) -> dict[str, object]:
    """Finalize a host-side Claude run using stored pre-run state."""
    state = json.loads(Path(state_file).read_text(encoding="utf-8"))
    report = Path(report_file).read_text(encoding="utf-8").strip()
    if not report:
        report = "[Overseer Error] Empty host Claude report."

    result = _finalize_run_outputs(
        deep=bool(state["deep"]),
        resolved_model=str(state["model"]),
        resolved_trigger=str(state["trigger_reason"]),
        run_timestamp=str(state["run_timestamp"]),
        run_id=str(state["run_id"]),
        report=report,
        history_before=list(state.get("history_before", [])),
        before_snapshot=dict(state.get("before_snapshot", {})),
        before_config=dict(state.get("before_config", {})),
    )
    result["run_id"] = state["run_id"]
    result["trigger_reason"] = state["trigger_reason"]
    return result


def _build_prompt_nightly(context: str, run_timestamp: str, run_id: str) -> str:
    """Standard nightly prompt for the fast scan."""
    return f"""You are VarpTrader Overseer, an autonomous trading-system supervisor.
Below is the current state of the VarpTrader system - config, recent trades,
log events, and performance stats.

{context}

{_build_memory_rules(run_timestamp, run_id)}

=== YOUR TASK ===

Analyze the above data and act IN ORDER:

1. Diagnose the single biggest performance problem with evidence.
   Cite specific numbers (win rate, PnL, counts).

2. Check for signal starvation: If 0 trades in last 24h, identify which
   filter is blocking signals and recommend loosening.

3. Auto-disable losing strategies: If any strategy has 0% win rate over
   7+ trades in the last 7 days, edit config.json to set "enabled": false.

4. Adjust aggressive filters: If blocked:fired ratio > 10:1, loosen the
   most aggressive filter threshold in config.json.

5. Run tests: Execute `python -m pytest tests/ -q` and fix any failures.

6. Restart if needed: If config.json was modified, run
   `docker compose restart autotrader` from the project directory.

7. Write your report: Summarize what you found and did. Include key numbers.

Output your full report as plain text.
"""


def _build_prompt_deep(context: str, run_timestamp: str, run_id: str) -> str:
    """Deep weekly prompt for the slower thorough pass."""
    return f"""You are VarpTrader Overseer performing a DEEP WEEKLY REVIEW.
You have full autonomy to refactor config, fix code bugs, and improve strategy
parameters. Be thorough - this is the weekly deep pass, not a quick nightly scan.

{context}

{_build_memory_rules(run_timestamp, run_id)}

=== YOUR TASK ===

Work through ALL of the following:

1. Full performance audit: For every strategy, compute win rate, avg PnL,
   Sharpe-like ratio, max losing streak. Identify the worst-performing one.

2. Root-cause analysis: WHY is that strategy losing? Wrong direction bias?
   Wrong thresholds? Wrong regime filter? Cite log evidence.

3. Parameter optimisation: For each enabled strategy, suggest and apply
   tighter parameter values based on the trade data (RSI thresholds, EMA
   periods, BB std, position size). Edit config.json directly.

4. Regime filter audit: Review the last 7 days of regime_blocked and
   session_bias_blocked log entries. Is the ADX threshold too tight or too
   loose? Adjust if needed.

5. Signal starvation deep check: If fired:blocked < 1:5, loosen the two
   most aggressive filters. If fired:blocked > 5:1, tighten stop_loss_pct.

6. Risk sizing review: If max_drawdown > 5% in the period, reduce
   position_size_pct by 20%. If max_drawdown < 1% with >10 trades, increase
   position_size_pct by 10%.

7. Strategy enable/disable: Disable any strategy with Sharpe < 0 over
   20+ trades. Re-enable any previously disabled strategy only when the
   change memory and current regime evidence both justify it.

8. Run full test suite: `python -m pytest tests/ -v` - fix ALL failures, do not skip.

9. Rebuild and restart: After all config/code changes:
   `docker compose build --no-cache autotrader && docker compose up -d autotrader`

10. Weekly report: Write a full markdown report with tables showing before/
    after metrics for every change made.

Output your full report as markdown.
"""


def run_overseer(
    deep: bool = False,
    model: str | None = None,
    trigger_reason: str | None = None,
) -> str:
    """Execute the overseer pipeline.

    Args:
        deep: Use the deep weekly prompt (Opus-level analysis).
        model: Claude model ID. Falls back to ANTHROPIC_MODEL env var,
            then to ``claude-sonnet-4-6``.
        trigger_reason: Human-readable reason for the run, used in Telegram alerts.

    Returns:
        The report text produced by Claude.
    """
    resolved_model = model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)
    resolved_trigger = trigger_reason or ("deep_weekly" if deep else "nightly")
    logger.info(
        "overseer_run_started | deep={} model={} trigger={}",
        deep,
        resolved_model,
        resolved_trigger,
    )

    ensure_memory_files()
    backfill_change_outcomes()

    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    history_before = load_json_list("overseer/change_log.json")
    before_snapshot = capture_repo_snapshot(".")
    before_config = read_json_object("config.json")

    context = build_context()
    prompt = (
        _build_prompt_deep(context, run_timestamp, run_id)
        if deep
        else _build_prompt_nightly(context, run_timestamp, run_id)
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "overseer_invoking_claude | model={} prompt_chars={} deep={} auth_mode={}",
        resolved_model,
        len(prompt),
        deep,
        _resolve_claude_auth_mode(),
    )
    try:
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--permission-mode",
                "bypassPermissions",
                "--model",
                resolved_model,
                "-p",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=900,
            env=_build_claude_env(),
        )
        report = result.stdout.strip()
        if result.returncode != 0:
            logger.warning(
                "overseer_claude_nonzero_exit | code={} stderr={}",
                result.returncode,
                result.stderr[:500],
            )
            if not report:
                report = (
                    f"[Overseer Error] Claude exited {result.returncode}\n"
                    f"{result.stderr[:1000]}"
                )
        report = _normalize_claude_error(report)
    except FileNotFoundError:
        report = (
            "[Overseer Error] 'claude' not found. Install: "
            "npm install -g @anthropic-ai/claude-code"
        )
        logger.error("overseer_claude_not_found")
    except subprocess.TimeoutExpired:
        report = "[Overseer Error] Claude timed out after 900 seconds."
        logger.error("overseer_claude_timeout")
    except Exception as exc:
        report = f"[Overseer Error] {exc}"
        logger.error("overseer_error | err={}", exc)

    result = _finalize_run_outputs(
        deep=deep,
        resolved_model=resolved_model,
        resolved_trigger=resolved_trigger,
        run_timestamp=run_timestamp,
        run_id=run_id,
        report=report,
        history_before=history_before,
        before_snapshot=before_snapshot,
        before_config=before_config,
    )
    return str(result["report"])


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "overseer.log",
        rotation="5 MB",
        retention="14 days",
        level="DEBUG",
    )

    parser = argparse.ArgumentParser(description="VarpTrader Overseer")
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Print context to stdout and exit (no Claude call - used by host wrapper).",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use deep weekly prompt (Opus). Default: nightly Sonnet prompt.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Claude model override. Default: ANTHROPIC_MODEL env var or claude-sonnet-4-6.",
    )
    parser.add_argument(
        "--trigger-reason",
        default=None,
        help="Optional run label used in Telegram summaries, e.g. nightly or signal_starvation.",
    )
    parser.add_argument(
        "--host-prepare",
        action="store_true",
        help="Print JSON payload for a host-side Claude run and exit.",
    )
    parser.add_argument(
        "--host-finalize",
        action="store_true",
        help="Finalize a host-side Claude run using stored snapshots and a report file.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="State file produced by --host-prepare.",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Report file produced by the host-side Claude run.",
    )
    args = parser.parse_args()

    if args.context_only:
        ensure_memory_files()
        backfill_change_outcomes()
        sys.stdout.write(build_context())
        sys.exit(0)

    if args.host_prepare:
        payload = prepare_host_run(
            deep=args.deep,
            trigger_reason=args.trigger_reason,
            model=args.model,
        )
        sys.stdout.write(json.dumps(payload))
        sys.exit(0)

    if args.host_finalize:
        if not args.state_file or not args.report_file:
            raise SystemExit("--host-finalize requires --state-file and --report-file")
        payload = finalize_host_run(
            state_file=args.state_file,
            report_file=args.report_file,
        )
        sys.stdout.write(json.dumps(payload))
        sys.exit(0)

    report = run_overseer(
        deep=args.deep,
        model=args.model,
        trigger_reason=args.trigger_reason,
    )
    print("\n" + "=" * 60)
    print(f"OVERSEER REPORT  [{'DEEP' if args.deep else 'NIGHTLY'}]")
    print("=" * 60)
    print(report)
