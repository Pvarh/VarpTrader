"""Parameter sweep optimizer using vectorbt."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

try:
    import vectorbt as vbt
except ImportError:
    vbt = None

from backtesting.vectorbt_runner import VectorBTRunner, _require_vbt


class StrategyOptimizer:
    """Uses vectorbt parameter sweeps to find optimal strategy parameters.

    Each ``optimize_*`` method runs a grid search over the specified
    parameter ranges, builds a ``vbt.Portfolio`` for every combination,
    and returns a :class:`pd.DataFrame` of results ranked by
    **profit factor** (descending).
    """

    def __init__(self, initial_capital: float = 100_000.0, fees: float = 0.001):
        self._initial_capital = initial_capital
        self._fees = fees
        self._runner = VectorBTRunner(
            initial_capital=initial_capital, fees=fees
        )

    # ------------------------------------------------------------------
    # ORB optimisation
    # ------------------------------------------------------------------

    def optimize_orb(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sweep ORB parameters and rank by profit factor.

        Parameter grid
        ~~~~~~~~~~~~~~
        * ``orb_window``: 15, 30, 45, 60 minutes
        * ``volume_multiplier``: 1.2, 1.5, 2.0

        Returns:
            :class:`pd.DataFrame` with columns
            ``orb_window, volume_multiplier`` plus all standard metrics,
            sorted by ``profit_factor`` descending.
        """
        _require_vbt()

        windows = [15, 30, 45, 60]
        vol_mults = [1.2, 1.5, 2.0]
        rows: list[dict] = []

        total_combos = len(windows) * len(vol_mults)
        logger.info("ORB optimisation — {} parameter combinations", total_combos)

        for orb_win, vol_m in itertools.product(windows, vol_mults):
            try:
                metrics = self._runner.run_orb_strategy(
                    df, orb_minutes=orb_win, volume_multiplier=vol_m
                )
                row = {"orb_window": orb_win, "volume_multiplier": vol_m}
                row.update(metrics)
                rows.append(row)
            except Exception as exc:
                logger.warning(
                    "ORB combo (win={}, vol={}) failed: {}", orb_win, vol_m, exc
                )

        result = pd.DataFrame(rows)
        if not result.empty:
            result.sort_values("profit_factor", ascending=False, inplace=True)
            result.reset_index(drop=True, inplace=True)
        logger.info("ORB optimisation complete — {} valid combos", len(result))
        return result

    # ------------------------------------------------------------------
    # EMA optimisation
    # ------------------------------------------------------------------

    def optimize_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sweep EMA crossover parameters and rank by profit factor.

        Parameter grid
        ~~~~~~~~~~~~~~
        * ``fast_ema``: 20, 50, 100
        * ``slow_ema``: 100, 200

        Only combinations where ``fast < slow`` are evaluated.

        Returns:
            :class:`pd.DataFrame` sorted by ``profit_factor`` descending.
        """
        _require_vbt()

        fast_values = [20, 50, 100]
        slow_values = [100, 200]
        rows: list[dict] = []

        combos = [
            (f, s)
            for f, s in itertools.product(fast_values, slow_values)
            if f < s
        ]
        logger.info("EMA optimisation — {} parameter combinations", len(combos))

        for fast, slow in combos:
            try:
                metrics = self._runner.run_ema_cross_strategy(
                    df, fast=fast, slow=slow
                )
                row = {"fast_ema": fast, "slow_ema": slow}
                row.update(metrics)
                rows.append(row)
            except Exception as exc:
                logger.warning(
                    "EMA combo (fast={}, slow={}) failed: {}", fast, slow, exc
                )

        result = pd.DataFrame(rows)
        if not result.empty:
            result.sort_values("profit_factor", ascending=False, inplace=True)
            result.reset_index(drop=True, inplace=True)
        logger.info("EMA optimisation complete — {} valid combos", len(result))
        return result

    # ------------------------------------------------------------------
    # RSI optimisation
    # ------------------------------------------------------------------

    def optimize_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sweep RSI threshold parameters and rank by profit factor.

        Parameter grid
        ~~~~~~~~~~~~~~
        * ``oversold``: 25, 30, 35
        * ``overbought``: 65, 70, 75

        Returns:
            :class:`pd.DataFrame` sorted by ``profit_factor`` descending.
        """
        _require_vbt()

        oversold_values = [25, 30, 35]
        overbought_values = [65, 70, 75]
        rows: list[dict] = []

        total_combos = len(oversold_values) * len(overbought_values)
        logger.info("RSI optimisation — {} parameter combinations", total_combos)

        for os_val, ob_val in itertools.product(oversold_values, overbought_values):
            try:
                metrics = self._runner.run_rsi_strategy(
                    df, rsi_period=14, oversold=os_val, overbought=ob_val
                )
                row = {"oversold": os_val, "overbought": ob_val}
                row.update(metrics)
                rows.append(row)
            except Exception as exc:
                logger.warning(
                    "RSI combo (os={}, ob={}) failed: {}", os_val, ob_val, exc
                )

        result = pd.DataFrame(rows)
        if not result.empty:
            result.sort_values("profit_factor", ascending=False, inplace=True)
            result.reset_index(drop=True, inplace=True)
        logger.info("RSI optimisation complete — {} valid combos", len(result))
        return result

    # ------------------------------------------------------------------
    # Run all optimisations
    # ------------------------------------------------------------------

    def optimize_all(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Run all parameter sweeps and return a combined results dict.

        Returns:
            Dictionary mapping strategy name (``"orb"``, ``"ema"``,
            ``"rsi"``) to their respective results DataFrames.
        """
        _require_vbt()

        results: dict[str, pd.DataFrame] = {}

        logger.info("Starting full optimisation sweep")

        try:
            results["orb"] = self.optimize_orb(df)
        except Exception as exc:
            logger.error("ORB optimisation failed entirely: {}", exc)

        try:
            results["ema"] = self.optimize_ema(df)
        except Exception as exc:
            logger.error("EMA optimisation failed entirely: {}", exc)

        try:
            results["rsi"] = self.optimize_rsi(df)
        except Exception as exc:
            logger.error("RSI optimisation failed entirely: {}", exc)

        logger.info(
            "Full optimisation sweep complete — {} strategies produced results",
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: dict[str, pd.DataFrame],
        output_path: str = "data/backtest_results.csv",
    ) -> None:
        """Save all optimisation results to a single CSV file.

        Each strategy's DataFrame is tagged with a ``strategy`` column
        before concatenation so the source can be identified in the
        combined output.

        Args:
            results: Dictionary returned by :meth:`optimize_all`.
            output_path: Destination CSV path (directories created
                automatically).
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []
        for strategy_name, df_result in results.items():
            if df_result.empty:
                continue
            tagged = df_result.copy()
            tagged.insert(0, "strategy", strategy_name)
            frames.append(tagged)

        if not frames:
            logger.warning("No optimisation results to save")
            return

        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(out, index=False)
        logger.info(
            "Saved {} optimisation rows to {}", len(combined), out.resolve()
        )
