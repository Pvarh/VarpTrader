"""Markdown report builder for Telegram daily summaries.

Formats performance analytics, config changes, and rejected proposals
into a Telegram-friendly Markdown string.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger


class ReportBuilder:
    """Build Markdown-formatted daily performance reports."""

    def build_daily_report(
        self,
        performance_data: dict[str, Any],
        config_changes: Optional[dict[str, Any]] = None,
        rejected: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Build a Markdown report summarising daily trading performance.

        The report includes:
        - Overall stats (total trades, wins, losses, PnL).
        - Per-strategy win rates.
        - Whale correlation analysis.
        - Worst-performing strategy highlight.
        - Config changes applied (if any).
        - Rejected config proposals (if any).

        Args:
            performance_data: Full report dict produced by
                :pyfunc:`PerformanceAnalyzer.build_full_report`.
            config_changes: Dict of parameter changes that were applied,
                or ``None`` if no changes were made.
            rejected: List of rejected change dicts (each with ``param``,
                ``value``, ``reason``), or ``None``.

        Returns:
            A Markdown-formatted string suitable for Telegram delivery.
        """
        lines: list[str] = []

        # ---- Header ----
        lines.append("*Daily Trading Report*")
        lines.append("")

        # ---- Overall stats ----
        total_trades = performance_data.get("total_trades", 0)
        total_wins = performance_data.get("total_wins", 0)
        total_losses = performance_data.get("total_losses", 0)
        total_pnl = performance_data.get("total_pnl", 0.0)
        lookback = performance_data.get("lookback_days", 30)

        overall_wr = (
            f"{total_wins / total_trades * 100:.1f}%"
            if total_trades > 0
            else "N/A"
        )
        pnl_sign = "+" if total_pnl >= 0 else ""

        lines.append(f"*Overall Stats* (last {lookback} days)")
        lines.append(f"  Trades: {total_trades}")
        lines.append(f"  Wins: {total_wins} | Losses: {total_losses}")
        lines.append(f"  Win Rate: {overall_wr}")
        lines.append(f"  PnL: {pnl_sign}{total_pnl:.4f}")
        lines.append("")

        # ---- Per-strategy win rates ----
        strategy_wr: dict[str, float] = performance_data.get(
            "strategy_win_rates", {}
        )
        if strategy_wr:
            lines.append("*Strategy Win Rates*")
            for strategy, wr in sorted(
                strategy_wr.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {strategy}: {wr * 100:.1f}%")
            lines.append("")

        # ---- Average PnL per strategy ----
        avg_pnl: dict[str, float] = performance_data.get(
            "avg_pnl_per_strategy", {}
        )
        if avg_pnl:
            lines.append("*Avg PnL per Strategy*")
            for strategy, pnl in sorted(
                avg_pnl.items(), key=lambda x: x[1], reverse=True
            ):
                sign = "+" if pnl >= 0 else ""
                lines.append(f"  {strategy}: {sign}{pnl:.4f}")
            lines.append("")

        # ---- Whale correlation ----
        whale: dict[str, float] = performance_data.get(
            "whale_correlation", {}
        )
        if whale:
            w_rate = whale.get("with_whale", 0.0)
            wo_rate = whale.get("without_whale", 0.0)
            diff = (w_rate - wo_rate) * 100

            lines.append("*Whale Correlation*")
            lines.append(f"  With whale signal: {w_rate * 100:.1f}%")
            lines.append(f"  Without whale signal: {wo_rate * 100:.1f}%")
            if diff > 0:
                lines.append(f"  Whale advantage: +{diff:.1f}pp")
            elif diff < 0:
                lines.append(f"  Whale disadvantage: {diff:.1f}pp")
            else:
                lines.append("  Whale advantage: 0.0pp")
            lines.append("")

        # ---- Worst strategy ----
        worst: Optional[str] = performance_data.get("worst_strategy")
        if worst:
            worst_wr = strategy_wr.get(worst, 0.0)
            lines.append("*Worst Strategy*")
            lines.append(f"  {worst} ({worst_wr * 100:.1f}% win rate)")
            lines.append("")

        # ---- Config changes ----
        if config_changes:
            lines.append("*Config Changes Applied*")
            for param, value in config_changes.items():
                lines.append(f"  {param}: {value}")
            lines.append("")
        else:
            lines.append("_No config changes applied._")
            lines.append("")

        # ---- Rejected changes ----
        if rejected:
            lines.append("*Rejected Changes*")
            for item in rejected:
                param = item.get("param", "?")
                value = item.get("value", "?")
                reason = item.get("reason", "unknown")
                lines.append(f"  {param}={value} -- {reason}")
            lines.append("")

        report = "\n".join(lines)

        logger.info(
            "daily_report_built | total_trades={total_trades} total_pnl={total_pnl} config_changes_count={config_changes_count} rejected_count={rejected_count}",
            total_trades=total_trades,
            total_pnl=total_pnl,
            config_changes_count=len(config_changes) if config_changes else 0,
            rejected_count=len(rejected) if rejected else 0,
        )
        return report
