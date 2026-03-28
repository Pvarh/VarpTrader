"""Container-side prepare/finalize helpers for host Claude nightly runs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from dotenv import load_dotenv
from loguru import logger

from alerts.telegram_bot import TelegramAlert
from analysis.analyzer import PerformanceAnalyzer
from analysis.config_updater import ConfigUpdater
from analysis.llm_advisor import LLMAdvisor
from analysis.report_builder import ReportBuilder
from journal.db import TradeDatabase
from journal.models import AnalysisRun

load_dotenv()

CONFIG_PATH = Path("config.json")
STATE_DIR = Path("logs/nightly_analysis_state")


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _lookup_nested_value(data: dict, path: list[str]) -> object:
    node: object = data
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _format_message_value(value: object, max_len: int = 48) -> str:
    text = json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= max_len else f"{text[: max_len - 3]}..."


def _build_nightly_change_summary(
    config_before: dict,
    report_data: dict,
    approved: dict[str, object],
    rejected: list[dict[str, object]],
    *,
    auto_apply: bool,
    min_trades: int | None = None,
) -> str:
    total_trades = int(report_data.get("total_trades", 0) or 0)
    status = "Applied" if auto_apply else "Pending approval"
    lines = [
        "NIGHTLY CONFIG UPDATE",
        "",
        f"Trades analyzed: {total_trades}",
        f"{status}: {len(approved)} | Rejected: {len(rejected)}",
    ]

    if min_trades is not None and total_trades < min_trades:
        lines.append(f"Insufficient trades for AI changes: need at least {min_trades}.")

    if approved:
        lines.append("")
        lines.append(f"{status}:")
        for param, new_value in list(approved.items())[:5]:
            path = ConfigUpdater.PARAM_PATHS.get(param, [])
            old_value = _lookup_nested_value(config_before, path) if path else None
            if old_value is None:
                lines.append(f"- {param} -> {_format_message_value(new_value)}")
            else:
                lines.append(
                    f"- {param}: {_format_message_value(old_value)} -> {_format_message_value(new_value)}"
                )
    elif min_trades is None:
        lines.append("")
        lines.append("No config changes proposed.")

    if rejected:
        lines.append("")
        lines.append("Rejected:")
        for item in rejected[:3]:
            lines.append(
                f"- {item.get('param', '?')} -> {_format_message_value(item.get('value'))} ({str(item.get('reason', 'unknown'))[:90]})"
            )
    return "\n".join(lines)


def _current_tunables(config: dict) -> dict[str, object]:
    return {
        "stop_loss_pct": _lookup_nested_value(config, ["risk", "stop_loss_pct"]),
        "position_size_pct": _lookup_nested_value(config, ["risk", "position_size_pct"]),
        "rsi_oversold": _lookup_nested_value(config, ["strategies", "rsi_momentum", "rsi_oversold"]),
        "rsi_overbought": _lookup_nested_value(config, ["strategies", "rsi_momentum", "rsi_overbought"]),
    }


def prepare_host_nightly(model: str | None = None) -> dict[str, object]:
    """Build report data and Claude prompt for a host-side nightly run."""
    config = _load_config()
    db_path = __import__("os").getenv("DB_PATH", "data/trades.db")
    db = TradeDatabase(db_path)
    analyzer = PerformanceAnalyzer(db)
    analysis_cfg = config.get("analysis", {})
    lookback = int(analysis_cfg.get("lookback_days", 30))
    auto_apply = bool(analysis_cfg.get("auto_apply_changes", False))
    min_trades = int(analysis_cfg.get("min_trades_for_suggestion", 15))
    resolved_model = model or analysis_cfg.get("model", "claude-sonnet-4-6")

    report_data = analyzer.build_full_report(lookback_days=lookback)
    config_before = config
    tunables = _current_tunables(config_before)
    _, user_prompt = LLMAdvisor.build_messages(report_data, tunables)

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"{run_id}.json"
    response_file = STATE_DIR / f"{run_id}_response.txt"
    state_file.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": resolved_model,
                "config_before": config_before,
                "report_data": report_data,
                "auto_apply": auto_apply,
                "min_trades": min_trades,
            }
        ),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "model": resolved_model,
        "prompt": user_prompt,
        "state_file": state_file.resolve().as_posix(),
        "response_file": response_file.resolve().as_posix(),
        "min_trades_met": report_data.get("total_trades", 0) >= min_trades,
    }


def finalize_host_nightly(state_file: str, response_file: str) -> dict[str, object]:
    """Apply Claude recommendations and send the nightly outputs."""
    state = json.loads(Path(state_file).read_text(encoding="utf-8"))
    config_before = dict(state["config_before"])
    report_data = dict(state["report_data"])
    auto_apply = bool(state["auto_apply"])
    min_trades = int(state["min_trades"])
    run_id = str(state["run_id"])

    db_path = __import__("os").getenv("DB_PATH", "data/trades.db")
    db = TradeDatabase(db_path)
    updater = ConfigUpdater(str(CONFIG_PATH))
    builder = ReportBuilder()
    telegram = TelegramAlert(
        bot_token=__import__("os").getenv("TELEGRAM_BOT_TOKEN", ""),
        chat_id=__import__("os").getenv("TELEGRAM_CHAT_ID", ""),
    )

    raw_output = Path(response_file).read_text(encoding="utf-8") if Path(response_file).exists() else ""

    approved: dict[str, object] = {}
    rejected: list[dict[str, object]] = []

    if report_data.get("total_trades", 0) >= min_trades:
        recommendations = LLMAdvisor.parse_response(raw_output)
        approved, rejected = updater.validate_changes(recommendations)
        if approved and auto_apply:
            updater.apply_changes(approved)

    report_md = builder.build_daily_report(
        report_data,
        config_changes=approved,
        rejected=rejected,
    )
    telegram.send_daily_report(report_md)
    telegram.send_message(
        _build_nightly_change_summary(
            config_before,
            report_data,
            approved,
            rejected,
            auto_apply=auto_apply,
            min_trades=None if report_data.get("total_trades", 0) >= min_trades else min_trades,
        ),
        parse_mode="",
    )

    db.insert_analysis_run(
        AnalysisRun(
            trades_analyzed=report_data.get("total_trades", 0),
            report_markdown=report_md,
            config_changes_json=json.dumps(approved) if approved else None,
            approved=1 if (approved and auto_apply) else 0,
        )
    )
    logger.info(
        "host_nightly_analysis_complete | run_id={} approved={} rejected={}",
        run_id,
        len(approved),
        len(rejected),
    )
    return {
        "run_id": run_id,
        "approved": len(approved),
        "rejected": len(rejected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Host nightly analysis helper")
    parser.add_argument("--host-prepare", action="store_true")
    parser.add_argument("--host-finalize", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--response-file", default=None)
    args = parser.parse_args()

    if args.host_prepare:
        sys.stdout.write(json.dumps(prepare_host_nightly(model=args.model)))
        return
    if args.host_finalize:
        if not args.state_file or not args.response_file:
            raise SystemExit("--host-finalize requires --state-file and --response-file")
        sys.stdout.write(
            json.dumps(
                finalize_host_nightly(
                    state_file=args.state_file,
                    response_file=args.response_file,
                )
            )
        )
        return
    raise SystemExit("Choose --host-prepare or --host-finalize")


if __name__ == "__main__":
    main()
