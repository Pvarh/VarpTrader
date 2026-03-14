"""Performance analytics for trade journal data.

Provides win-rate breakdowns by strategy, day-of-week, hour-of-day,
whale correlation, and PnL summaries using raw SQL against the trade database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger

from journal.db import TradeDatabase


class PerformanceAnalyzer:
    """Compute trading performance metrics from the trade database."""

    def __init__(self, db: TradeDatabase) -> None:
        """Initialize with a TradeDatabase instance.

        Args:
            db: An initialized TradeDatabase used for querying closed trades.
        """
        self.db = db

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _since_iso(self, lookback_days: int) -> str:
        """Return an ISO-format timestamp ``lookback_days`` ago from now."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        return cutoff.isoformat()

    # ------------------------------------------------------------------
    # Public analytics
    # ------------------------------------------------------------------

    def compute_strategy_win_rates(
        self, lookback_days: int = 30
    ) -> dict[str, float]:
        """Query closed trades, group by strategy, compute wins/total for each.

        Args:
            lookback_days: Only consider trades from the last N days.

        Returns:
            Mapping of strategy name to win-rate (0.0 -- 1.0).
            Returns an empty dict when no closed trades exist.
        """
        since = self._since_iso(lookback_days)
        sql = """
            SELECT
                strategy,
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS wins
            FROM trades
            WHERE outcome IN ('win', 'loss', 'breakeven')
              AND timestamp >= ?
            GROUP BY strategy
        """
        conn = self.db._get_connection()
        try:
            rows = conn.execute(sql, (since,)).fetchall()
        except Exception:
            logger.exception("compute_strategy_win_rates_failed")
            return {}

        result: dict[str, float] = {}
        for row in rows:
            total = row["total"]
            wins = row["wins"]
            result[row["strategy"]] = wins / total if total else 0.0
        logger.info(
            "strategy_win_rates_computed | strategies={strategies} lookback_days={lookback_days}",
            strategies=len(result),
            lookback_days=lookback_days,
        )
        return result

    def compute_win_rate_by_day(self) -> dict[str, dict[str, float]]:
        """Win rate per strategy per day_of_week.

        Returns:
            Nested dict ``{strategy: {day_of_week: win_rate}}``.
            Returns an empty dict when no closed trades exist.
        """
        sql = """
            SELECT
                strategy,
                day_of_week,
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS wins
            FROM trades
            WHERE outcome IN ('win', 'loss', 'breakeven')
              AND day_of_week IS NOT NULL
            GROUP BY strategy, day_of_week
        """
        conn = self.db._get_connection()
        try:
            rows = conn.execute(sql).fetchall()
        except Exception:
            logger.exception("compute_win_rate_by_day_failed")
            return {}

        result: dict[str, dict[str, float]] = {}
        for row in rows:
            strategy = row["strategy"]
            day = row["day_of_week"]
            total = row["total"]
            wins = row["wins"]
            result.setdefault(strategy, {})[day] = wins / total if total else 0.0
        logger.info("win_rate_by_day_computed | strategies={strategies}", strategies=len(result))
        return result

    def compute_win_rate_by_hour(self) -> dict[str, dict[int, float]]:
        """Win rate per strategy per hour_of_day.

        Returns:
            Nested dict ``{strategy: {hour: win_rate}}``.
            Returns an empty dict when no closed trades exist.
        """
        sql = """
            SELECT
                strategy,
                hour_of_day,
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS wins
            FROM trades
            WHERE outcome IN ('win', 'loss', 'breakeven')
              AND hour_of_day IS NOT NULL
            GROUP BY strategy, hour_of_day
        """
        conn = self.db._get_connection()
        try:
            rows = conn.execute(sql).fetchall()
        except Exception:
            logger.exception("compute_win_rate_by_hour_failed")
            return {}

        result: dict[str, dict[int, float]] = {}
        for row in rows:
            strategy = row["strategy"]
            hour = row["hour_of_day"]
            total = row["total"]
            wins = row["wins"]
            result.setdefault(strategy, {})[hour] = wins / total if total else 0.0
        logger.info("win_rate_by_hour_computed | strategies={strategies}", strategies=len(result))
        return result

    def compute_avg_pnl_per_strategy(self) -> dict[str, float]:
        """Average PnL per strategy across all closed trades.

        Returns:
            Mapping of strategy name to average PnL value.
            Returns an empty dict when no closed trades exist.
        """
        sql = """
            SELECT
                strategy,
                AVG(pnl) AS avg_pnl
            FROM trades
            WHERE outcome IN ('win', 'loss', 'breakeven')
              AND pnl IS NOT NULL
            GROUP BY strategy
        """
        conn = self.db._get_connection()
        try:
            rows = conn.execute(sql).fetchall()
        except Exception:
            logger.exception("compute_avg_pnl_per_strategy_failed")
            return {}

        result: dict[str, float] = {
            row["strategy"]: round(row["avg_pnl"], 4) for row in rows
        }
        logger.info("avg_pnl_per_strategy_computed | strategies={strategies}", strategies=len(result))
        return result

    def compute_whale_correlation(self) -> dict[str, float]:
        """Compute win rates with and without whale_flag.

        Returns:
            Dict with keys ``"with_whale"`` and ``"without_whale"``, each
            mapping to the corresponding win rate (0.0 -- 1.0).
            Returns ``{"with_whale": 0.0, "without_whale": 0.0}`` when no
            data is available.
        """
        sql = """
            SELECT
                whale_flag,
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS wins
            FROM trades
            WHERE outcome IN ('win', 'loss', 'breakeven')
            GROUP BY whale_flag
        """
        conn = self.db._get_connection()
        try:
            rows = conn.execute(sql).fetchall()
        except Exception:
            logger.exception("compute_whale_correlation_failed")
            return {"with_whale": 0.0, "without_whale": 0.0}

        rates: dict[int, float] = {}
        for row in rows:
            total = row["total"]
            wins = row["wins"]
            rates[row["whale_flag"]] = wins / total if total else 0.0

        result = {
            "with_whale": rates.get(1, 0.0),
            "without_whale": rates.get(0, 0.0),
        }
        logger.info(
            "whale_correlation_computed | with_whale={with_whale} without_whale={without_whale}",
            with_whale=result["with_whale"],
            without_whale=result["without_whale"],
        )
        return result

    def get_worst_strategy(
        self, lookback_days: int = 30
    ) -> Optional[str]:
        """Return the strategy name with the lowest win rate.

        Args:
            lookback_days: Only consider trades from the last N days.

        Returns:
            Strategy name string, or ``None`` when no data is available.
        """
        win_rates = self.compute_strategy_win_rates(lookback_days)
        if not win_rates:
            return None
        worst = min(win_rates, key=win_rates.get)  # type: ignore[arg-type]
        logger.info(
            "worst_strategy_identified | strategy={strategy} win_rate={win_rate}",
            strategy=worst,
            win_rate=win_rates[worst],
        )
        return worst

    def build_full_report(self, lookback_days: int = 30) -> dict:
        """Build a comprehensive report dict with all available metrics.

        The report contains:
        - ``strategy_win_rates``: per-strategy win rates for the period.
        - ``win_rate_by_day``: per-strategy, per-weekday win rates.
        - ``win_rate_by_hour``: per-strategy, per-hour win rates.
        - ``avg_pnl_per_strategy``: average PnL per strategy.
        - ``whale_correlation``: win rates with/without whale flag.
        - ``worst_strategy``: name of the weakest strategy.
        - ``lookback_days``: the lookback window used.
        - ``total_trades``: number of closed trades in the window.
        - ``total_wins``: number of winning trades in the window.
        - ``total_losses``: number of losing trades in the window.
        - ``total_pnl``: aggregate PnL in the window.

        Args:
            lookback_days: Number of days to look back.

        Returns:
            Dictionary containing the full performance report.
        """
        since = self._since_iso(lookback_days)

        # Aggregate totals
        totals_sql = """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses,
                COALESCE(SUM(pnl), 0.0) AS total_pnl
            FROM trades
            WHERE outcome IN ('win', 'loss', 'breakeven')
              AND timestamp >= ?
        """
        conn = self.db._get_connection()
        try:
            row = conn.execute(totals_sql, (since,)).fetchone()
        except Exception:
            logger.exception("build_full_report_totals_failed")
            row = None

        total_trades = row["total"] if row else 0
        total_wins = row["wins"] if row else 0
        total_losses = row["losses"] if row else 0
        total_pnl = round(row["total_pnl"], 4) if row else 0.0

        report: dict = {
            "lookback_days": lookback_days,
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_pnl": total_pnl,
            "strategy_win_rates": self.compute_strategy_win_rates(lookback_days),
            "win_rate_by_day": self.compute_win_rate_by_day(),
            "win_rate_by_hour": self.compute_win_rate_by_hour(),
            "avg_pnl_per_strategy": self.compute_avg_pnl_per_strategy(),
            "whale_correlation": self.compute_whale_correlation(),
            "worst_strategy": self.get_worst_strategy(lookback_days),
        }
        logger.info(
            "full_report_built | total_trades={total_trades} total_pnl={total_pnl}",
            total_trades=total_trades,
            total_pnl=total_pnl,
        )
        return report
