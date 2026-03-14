"""Performance reporting and go-live threshold checking."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from loguru import logger


class PerformanceReport:
    """Checks backtest results against go-live thresholds from ``config.json``.

    Each strategy is mapped to a category (``orb``, ``mean_reversion``, or
    ``trend_following``) and validated against the matching win-rate
    threshold plus the universal minimum profit factor.
    """

    STRATEGY_CATEGORIES: dict[str, str] = {
        "first_candle": "orb",
        "ema_cross": "trend_following",
        "vwap_reversion": "mean_reversion",
        "rsi_momentum": "mean_reversion",
        "bollinger_fade": "mean_reversion",
    }

    # Maps category to the config key that stores the win-rate threshold
    _WIN_RATE_KEYS: dict[str, str] = {
        "orb": "orb_min_win_rate",
        "mean_reversion": "mean_reversion_min_win_rate",
        "trend_following": "trend_following_min_win_rate",
    }

    def __init__(self, config: dict):
        """Initialise with the full config dictionary (top-level)."""
        self._thresholds = config.get("go_live_thresholds", {})
        self._config = config

    # ------------------------------------------------------------------
    # Single-strategy check
    # ------------------------------------------------------------------

    def check_strategy(self, strategy_name: str, metrics: dict) -> dict:
        """Check a single strategy against its go-live thresholds.

        Args:
            strategy_name: One of the keys in :attr:`STRATEGY_CATEGORIES`.
            metrics: Metrics dictionary produced by
                :meth:`VectorBTRunner._extract_metrics`.

        Returns:
            Dictionary with fields: ``strategy``, ``category``,
            ``win_rate``, ``profit_factor``, ``win_rate_threshold``,
            ``pf_threshold``, ``win_rate_pass``, ``pf_pass``,
            ``overall_pass``.
        """
        category = self.STRATEGY_CATEGORIES.get(strategy_name, "unknown")
        wr_key = self._WIN_RATE_KEYS.get(category)
        wr_threshold = self._thresholds.get(wr_key, 0.0) if wr_key else 0.0
        pf_threshold = self._thresholds.get(
            "all_strategies_min_profit_factor", 1.0
        )

        actual_wr = metrics.get("win_rate", 0.0)
        actual_pf = metrics.get("profit_factor", 0.0)

        wr_pass = actual_wr >= wr_threshold
        pf_pass = actual_pf >= pf_threshold

        result = {
            "strategy": strategy_name,
            "category": category,
            "win_rate": round(actual_wr, 4),
            "profit_factor": round(actual_pf, 4),
            "win_rate_threshold": round(wr_threshold, 4),
            "pf_threshold": round(pf_threshold, 4),
            "win_rate_pass": wr_pass,
            "pf_pass": pf_pass,
            "overall_pass": wr_pass and pf_pass,
        }

        status = "PASS" if result["overall_pass"] else "FAIL"
        logger.info(
            "{} [{}] — WR {:.2%} (>={:.2%} {}) | PF {:.2f} (>={:.2f} {}) => {}",
            strategy_name,
            category,
            actual_wr,
            wr_threshold,
            "ok" if wr_pass else "FAIL",
            actual_pf,
            pf_threshold,
            "ok" if pf_pass else "FAIL",
            status,
        )
        return result

    # ------------------------------------------------------------------
    # Check all strategies
    # ------------------------------------------------------------------

    def check_all(self, all_metrics: dict[str, dict]) -> list[dict]:
        """Check every strategy in *all_metrics* against thresholds.

        Args:
            all_metrics: Dictionary mapping strategy name to its metrics
                dictionary (as returned by
                :meth:`VectorBTRunner.run_all`).

        Returns:
            List of check-result dictionaries (one per strategy).
        """
        results: list[dict] = []
        for strategy_name, metrics in all_metrics.items():
            results.append(self.check_strategy(strategy_name, metrics))
        return results

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    def build_report(self, all_metrics: dict[str, dict]) -> str:
        """Build a Markdown report showing PASS/FAIL per strategy.

        The report includes a summary table with win rate, profit factor,
        total trades, Sharpe ratio, max drawdown, and pass/fail verdicts.

        Args:
            all_metrics: Dictionary mapping strategy name to its metrics.

        Returns:
            Multi-line Markdown string.
        """
        checks = self.check_all(all_metrics)

        lines: list[str] = [
            "# Backtest Performance Report",
            "",
            "## Go-Live Threshold Summary",
            "",
            "| Strategy | Category | Win Rate | WR Thresh | PF | PF Thresh "
            "| Trades | Sharpe | Max DD | Verdict |",
            "|----------|----------|----------|-----------|------|-----------|"
            "--------|--------|--------|---------|",
        ]

        pass_count = 0
        fail_count = 0

        for chk in checks:
            sname = chk["strategy"]
            metrics = all_metrics.get(sname, {})
            verdict = "PASS" if chk["overall_pass"] else "FAIL"
            if chk["overall_pass"]:
                pass_count += 1
            else:
                fail_count += 1

            lines.append(
                "| {strategy} | {category} | {wr} | {wr_t} | {pf} | {pf_t} "
                "| {trades} | {sharpe} | {dd} | **{verdict}** |".format(
                    strategy=sname,
                    category=chk["category"],
                    wr=f"{chk['win_rate']:.2%}",
                    wr_t=f"{chk['win_rate_threshold']:.2%}",
                    pf=f"{chk['profit_factor']:.2f}",
                    pf_t=f"{chk['pf_threshold']:.2f}",
                    trades=metrics.get("total_trades", "N/A"),
                    sharpe=f"{metrics.get('sharpe_ratio', 0.0):.2f}",
                    dd=f"{metrics.get('max_drawdown_pct', 0.0):.2%}",
                    verdict=verdict,
                )
            )

        lines.append("")
        lines.append(f"**Passed:** {pass_count} / {pass_count + fail_count}")
        lines.append(f"**Failed:** {fail_count} / {pass_count + fail_count}")
        lines.append("")

        # Per-strategy detail sections
        lines.append("## Detailed Metrics")
        lines.append("")
        for sname, metrics in all_metrics.items():
            lines.append(f"### {sname}")
            lines.append("")
            lines.append(f"- **Total Return:** {metrics.get('total_return', 0.0):.4%}")
            lines.append(f"- **Win Rate:** {metrics.get('win_rate', 0.0):.2%}")
            lines.append(f"- **Profit Factor:** {metrics.get('profit_factor', 0.0):.2f}")
            lines.append(f"- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 0.0):.2f}")
            lines.append(f"- **Calmar Ratio:** {metrics.get('calmar_ratio', 0.0):.2f}")
            lines.append(f"- **Max Drawdown:** {metrics.get('max_drawdown_pct', 0.0):.2%}")
            lines.append(f"- **Total Trades:** {metrics.get('total_trades', 0)}")
            lines.append(f"- **Avg Win:** ${metrics.get('avg_win', 0.0):,.2f}")
            lines.append(f"- **Avg Loss:** ${metrics.get('avg_loss', 0.0):,.2f}")
            lines.append("")

        report = "\n".join(lines)
        logger.info(
            "Performance report built — {} strategies ({} pass, {} fail)",
            pass_count + fail_count,
            pass_count,
            fail_count,
        )
        return report

    # ------------------------------------------------------------------
    # Auto-disable failing strategies
    # ------------------------------------------------------------------

    def auto_disable_failing(
        self,
        all_metrics: dict[str, dict],
        config_path: str = "config.json",
    ) -> list[str]:
        """Disable strategies that FAIL threshold checks.

        For each failing strategy the ``enabled`` flag inside
        ``config["strategies"][strategy_name]`` is set to ``false``.
        The updated config is written **atomically** — a temporary file
        is written in the same directory and then renamed over the
        original to avoid partial writes.

        Args:
            all_metrics: Strategy name to metrics mapping.
            config_path: Path to the config file to update.

        Returns:
            List of strategy names that were disabled.
        """
        checks = self.check_all(all_metrics)
        failing = [c["strategy"] for c in checks if not c["overall_pass"]]

        if not failing:
            logger.info("All strategies passed — nothing to disable")
            return []

        cfg_path = Path(config_path).resolve()
        if not cfg_path.exists():
            logger.error("Config file not found at {}", cfg_path)
            return []

        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read config: {}", exc)
            return []

        strategies_section = config.get("strategies", {})
        disabled: list[str] = []

        for sname in failing:
            if sname in strategies_section:
                if strategies_section[sname].get("enabled", True):
                    strategies_section[sname]["enabled"] = False
                    disabled.append(sname)
                    logger.warning(
                        "Disabling strategy '{}' — failed go-live thresholds",
                        sname,
                    )
            else:
                logger.warning(
                    "Strategy '{}' not found in config — skipping disable",
                    sname,
                )

        if not disabled:
            logger.info("No strategies needed disabling")
            return []

        # Atomic write: temp file in same directory, then rename
        try:
            dir_path = cfg_path.parent
            fd, tmp_path = tempfile.mkstemp(
                dir=str(dir_path), suffix=".tmp", prefix=".config_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_fh:
                    json.dump(config, tmp_fh, indent=2)
                    tmp_fh.write("\n")
                # On Windows os.rename fails if destination exists;
                # os.replace handles this cross-platform.
                os.replace(tmp_path, str(cfg_path))
            except BaseException:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            logger.error("Failed to write config atomically: {}", exc)
            return []

        logger.info(
            "Disabled {} strategies in {}: {}",
            len(disabled),
            cfg_path,
            ", ".join(disabled),
        )
        return disabled
