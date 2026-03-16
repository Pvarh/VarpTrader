"""VarpTrader overseer runner.

Standalone script that builds context from logs, database, and config,
then invokes Claude Code with a diagnostic/action prompt. Saves the
resulting report to overseer/reports/YYYY-MM-DD_HH.txt.

Usage:
    cd autotrader

    # Nightly fast scan — Sonnet (default)
    ANTHROPIC_MODEL=claude-sonnet-4-6 python -m overseer.run_overseer

    # Weekly deep refactor — Opus
    ANTHROPIC_MODEL=claude-opus-4-6 python -m overseer.run_overseer --deep

    # Context only (for host wrapper scripts)
    python -m overseer.run_overseer --context-only

Models (set via ANTHROPIC_MODEL env var):
    claude-sonnet-4-6   Fast, cheap — good for nightly checks (default)
    claude-opus-4-6     Thorough, expensive — good for weekly deep refactor
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from loguru import logger

from overseer.context_builder import build_context

REPORTS_DIR = Path("overseer/reports")

_DEFAULT_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_prompt_nightly(context: str) -> str:
    """Standard nightly prompt — fast Sonnet scan."""
    return f"""You are VarpTrader Overseer, an autonomous trading-system supervisor.
Below is the current state of the VarpTrader system — config, recent trades,
log events, and performance stats.

{context}

=== YOUR TASK ===

Analyze the above data and act IN ORDER:

1. **Diagnose the single biggest performance problem** with evidence.
   Cite specific numbers (win rate, PnL, counts).

2. **Check for signal starvation**: If 0 trades in last 24h, identify which
   filter is blocking signals and recommend loosening.

3. **Auto-disable losing strategies**: If any strategy has 0% win rate over
   7+ trades in the last 7 days, edit config.json to set "enabled": false.

4. **Adjust aggressive filters**: If blocked:fired ratio > 10:1, loosen the
   most aggressive filter threshold in config.json.

5. **Run tests**: Execute `pytest tests/ -q` and fix any failures.

6. **Restart if needed**: If config.json was modified, run
   `docker compose restart autotrader` from the project directory.

7. **Write your report**: Summarize what you found and did. Include key numbers.

Output your full report as plain text.
"""


def _build_prompt_deep(context: str) -> str:
    """Deep weekly prompt — thorough Opus refactor."""
    return f"""You are VarpTrader Overseer performing a DEEP WEEKLY REVIEW.
You have full autonomy to refactor config, fix code bugs, and improve strategy
parameters. Be thorough — this is the weekly deep pass, not a quick nightly scan.

{context}

=== YOUR TASK ===

Work through ALL of the following:

1. **Full performance audit**: For every strategy, compute win rate, avg PnL,
   Sharpe-like ratio, max losing streak. Identify the worst-performing one.

2. **Root-cause analysis**: WHY is that strategy losing? Wrong direction bias?
   Wrong thresholds? Wrong regime filter? Cite log evidence.

3. **Parameter optimisation**: For each enabled strategy, suggest and apply
   tighter parameter values based on the trade data (RSI thresholds, EMA
   periods, BB std, position size). Edit config.json directly.

4. **Regime filter audit**: Review the last 7 days of regime_blocked and
   session_bias_blocked log entries. Is the ADX threshold too tight or too
   loose? Adjust if needed.

5. **Signal starvation deep check**: If fired:blocked < 1:5, loosen the two
   most aggressive filters. If fired:blocked > 5:1, tighten stop_loss_pct.

6. **Risk sizing review**: If max_drawdown > 5% in the period, reduce
   position_size_pct by 20%. If max_drawdown < 1% with >10 trades, increase
   position_size_pct by 10%.

7. **Strategy enable/disable**: Disable any strategy with Sharpe < 0 over
   20+ trades. Re-enable any previously disabled strategy if market regime
   has changed (e.g. ranging → trending).

8. **Run full test suite**: `pytest tests/ -v` — fix ALL failures, don't skip.

9. **Rebuild and restart**: After all config/code changes:
   `docker compose build --no-cache autotrader && docker compose up -d autotrader`

10. **Weekly report**: Write a full markdown report with tables showing before/
    after metrics for every change made.

Output your full report as markdown.
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_overseer(deep: bool = False, model: str | None = None) -> str:
    """Execute the overseer pipeline.

    Args:
        deep:  Use the deep weekly prompt (Opus-level analysis).
        model: Claude model ID.  Falls back to ANTHROPIC_MODEL env var,
               then to ``claude-sonnet-4-6``.

    Returns:
        The report text produced by Claude.
    """
    resolved_model = model or os.getenv("ANTHROPIC_MODEL", _DEFAULT_MODEL)
    logger.info(
        "overseer_run_started | deep={} model={}", deep, resolved_model
    )

    context = build_context()
    prompt = _build_prompt_deep(context) if deep else _build_prompt_nightly(context)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "overseer_invoking_claude | model={} prompt_chars={} deep={}",
        resolved_model, len(prompt), deep,
    )

    try:
        result = subprocess.run(
            ["claude", "--print", "--model", resolved_model, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=900,   # Opus can take longer
        )
        report = result.stdout.strip()
        if result.returncode != 0:
            logger.warning(
                "overseer_claude_nonzero_exit | code={} stderr={}",
                result.returncode, result.stderr[:500],
            )
            if not report:
                report = (
                    f"[Overseer Error] Claude exited {result.returncode}\n"
                    f"{result.stderr[:1000]}"
                )
    except FileNotFoundError:
        report = "[Overseer Error] 'claude' not found. Install: npm install -g @anthropic-ai/claude-code"
        logger.error("overseer_claude_not_found")
    except subprocess.TimeoutExpired:
        report = "[Overseer Error] Claude timed out after 900 seconds."
        logger.error("overseer_claude_timeout")
    except Exception as exc:
        report = f"[Overseer Error] {exc}"
        logger.error("overseer_error | err={}", exc)

    suffix = "_deep" if deep else ""
    filename = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H") + suffix + ".txt"
    report_path = REPORTS_DIR / filename
    report_path.write_text(report, encoding="utf-8")
    logger.info("overseer_report_saved | path={} model={}", report_path, resolved_model)

    return report


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
        help="Print context to stdout and exit (no Claude call — used by host wrapper).",
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
    args = parser.parse_args()

    if args.context_only:
        sys.stdout.write(build_context())
        sys.exit(0)

    report = run_overseer(deep=args.deep, model=args.model)
    print("\n" + "=" * 60)
    print(f"OVERSEER REPORT  [{'DEEP' if args.deep else 'NIGHTLY'}]")
    print("=" * 60)
    print(report)
