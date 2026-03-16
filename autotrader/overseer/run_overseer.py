"""VarpTrader overseer runner.

Standalone script that builds context from logs, database, and config,
then invokes Claude Code with a diagnostic/action prompt. Saves the
resulting report to overseer/reports/YYYY-MM-DD_HH.txt.

Usage:
    cd autotrader
    python -m overseer.run_overseer               # full run (requires claude CLI)
    python -m overseer.run_overseer --context-only # print context to stdout and exit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure autotrader root is on sys.path so local imports resolve.
sys.path.insert(0, ".")

from loguru import logger

from overseer.context_builder import build_context

# Directory for saved reports
REPORTS_DIR = Path("overseer/reports")


def _build_prompt(context: str) -> str:
    """Construct the full LLM prompt with context and instructions.

    Args:
        context: The assembled overseer context string.

    Returns:
        Complete prompt string for Claude Code.
    """
    return f"""You are VarpTrader Overseer, an autonomous trading-system supervisor.
Below is the current state of the VarpTrader system -- config, recent trades,
log events, and performance stats.

{context}

=== YOUR TASK ===

Analyze the above data and take the following actions IN ORDER:

1. **Diagnose the single biggest performance problem** with evidence from the
   data. Cite specific numbers (win rate, PnL, counts). If no clear problem
   exists, say so.

2. **Check for signal starvation**: If zero trades were opened in the last 24h
   despite the bot running, identify which filters are blocking all signals
   and recommend loosening.

3. **Auto-disable losing strategies**: If any strategy has a 0% win rate over
   7 or more trades in the last 7 days, edit config.json to set that
   strategy's "enabled" to false. Explain what you changed and why.

4. **Adjust aggressive filters**: If the blocked:fired signal ratio exceeds
   10:1, identify the most aggressive filter (the one blocking the most
   signals) and loosen its threshold in config.json. Explain what you changed.

5. **Run tests**: Execute `pytest tests/` and fix any failures. If tests pass
   on first try, simply note that.

6. **Restart if needed**: If you modified config.json, run
   `docker compose restart autotrader` to apply the changes.

7. **Write your report**: Summarize everything you found and did. Be concise
   but include the key numbers.

Output your full report as plain text.
"""


def run_overseer() -> str:
    """Execute the overseer pipeline.

    1. Build context from logs, DB, and config.
    2. Send prompt to Claude Code via subprocess.
    3. Save report to overseer/reports/.

    Returns:
        The report text produced by Claude Code.
    """
    logger.info("overseer_run_started")

    # Build context
    context = build_context()

    # Build prompt
    prompt = _build_prompt(context)

    # Ensure reports directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run Claude Code
    logger.info("overseer_invoking_claude | prompt_chars={}", len(prompt))
    try:
        result = subprocess.run(
            ["claude", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=600,
        )
        report = result.stdout.strip()
        if result.returncode != 0:
            logger.warning(
                "overseer_claude_nonzero_exit | code={} stderr={}",
                result.returncode,
                result.stderr[:500],
            )
            if not report:
                report = f"[Overseer Error] Claude exited with code {result.returncode}\n{result.stderr[:1000]}"
    except FileNotFoundError:
        report = "[Overseer Error] 'claude' CLI not found on PATH. Install Claude Code first."
        logger.error("overseer_claude_not_found")
    except subprocess.TimeoutExpired:
        report = "[Overseer Error] Claude Code timed out after 600 seconds."
        logger.error("overseer_claude_timeout")
    except Exception as exc:
        report = f"[Overseer Error] Unexpected error invoking Claude: {exc}"
        logger.error("overseer_claude_error | err={}", exc)

    # Save report
    now = datetime.now(timezone.utc)
    filename = now.strftime("%Y-%m-%d_%H") + ".txt"
    report_path = REPORTS_DIR / filename
    report_path.write_text(report, encoding="utf-8")
    logger.info("overseer_report_saved | path={}", report_path)

    return report


if __name__ == "__main__":
    # Logging setup for standalone runs
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
        help="Print assembled context to stdout and exit (no Claude call).",
    )
    args = parser.parse_args()

    if args.context_only:
        # Just output the context — used by the host wrapper script
        context = build_context()
        sys.stdout.write(context)
        sys.exit(0)

    report = run_overseer()
    print("\n" + "=" * 60)
    print("OVERSEER REPORT")
    print("=" * 60)
    print(report)
