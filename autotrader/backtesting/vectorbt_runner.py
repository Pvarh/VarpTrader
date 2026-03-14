"""Vectorbt-based backtesting engine for all signal strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

try:
    import vectorbt as vbt
except ImportError:
    vbt = None
    logger.warning("vectorbt not installed — backtesting unavailable")

import yfinance as yf


def _require_vbt() -> None:
    """Raise ImportError if vectorbt is not available."""
    if vbt is None:
        raise ImportError(
            "vectorbt is required for backtesting. "
            "Install it with: pip install vectorbt"
        )


class VectorBTRunner:
    """Run signal strategies through vectorbt's vectorized backtesting engine."""

    def __init__(self, initial_capital: float = 100_000.0, fees: float = 0.001):
        self._initial_capital = initial_capital
        self._fees = fees

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "5m",
    ) -> dict[str, pd.DataFrame]:
        """Load historical OHLCV from yfinance for multiple symbols.

        Args:
            symbols: List of ticker symbols (e.g. ``["AAPL", "TSLA"]``).
            start: Start date string accepted by yfinance (e.g. ``"2024-01-01"``).
            end: End date string.
            interval: Bar size — ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"1d"``, etc.

        Returns:
            Dictionary mapping each symbol to a DataFrame with columns
            ``Open, High, Low, Close, Volume``.
        """
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            logger.info("Downloading {} data ({} — {}, {})", sym, start, end, interval)
            try:
                df = yf.download(
                    sym,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    logger.warning("No data returned for {}", sym)
                    continue

                # Flatten MultiIndex columns that yfinance sometimes returns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Normalize column names to title case
                df.columns = [c.title() for c in df.columns]
                required = {"Open", "High", "Low", "Close", "Volume"}
                if not required.issubset(set(df.columns)):
                    logger.warning(
                        "Missing columns for {}. Got: {}", sym, list(df.columns)
                    )
                    continue

                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.dropna(inplace=True)
                result[sym] = df
                logger.info("Loaded {} bars for {}", len(df), sym)
            except Exception as exc:
                logger.error("Failed to download {}: {}", sym, exc)
        return result

    # ------------------------------------------------------------------
    # Strategy: RSI mean-reversion
    # ------------------------------------------------------------------

    def run_rsi_strategy(
        self,
        df: pd.DataFrame,
        rsi_period: int = 14,
        oversold: int = 32,
        overbought: int = 68,
    ) -> dict:
        """Run RSI mean-reversion strategy.

        * Entry long  — RSI crosses below *oversold*.
        * Exit long   — RSI crosses above *overbought*.

        Returns:
            Dictionary with standardized performance metrics.
        """
        _require_vbt()
        close = df["Close"]

        rsi = vbt.RSI.run(close, window=rsi_period).rsi

        # Entry when RSI crosses below oversold (long entry on dip)
        entries = rsi.vbt.crossed_below(oversold)
        # Exit when RSI crosses above overbought
        exits = rsi.vbt.crossed_above(overbought)

        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=self._initial_capital,
            fees=self._fees,
            freq="5m",
        )
        logger.info(
            "RSI strategy complete — {} trades",
            pf.trades.count(),
        )
        return self._extract_metrics(pf)

    # ------------------------------------------------------------------
    # Strategy: EMA crossover
    # ------------------------------------------------------------------

    def run_ema_cross_strategy(
        self, df: pd.DataFrame, fast: int = 50, slow: int = 200
    ) -> dict:
        """Run EMA crossover (trend-following) strategy.

        * Entry — fast EMA crosses above slow EMA.
        * Exit  — fast EMA crosses below slow EMA.

        Returns:
            Dictionary with standardized performance metrics.
        """
        _require_vbt()
        close = df["Close"]

        fast_ema = vbt.MA.run(close, window=fast, ewm=True).ma
        slow_ema = vbt.MA.run(close, window=slow, ewm=True).ma

        entries = fast_ema.vbt.crossed_above(slow_ema)
        exits = fast_ema.vbt.crossed_below(slow_ema)

        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=self._initial_capital,
            fees=self._fees,
            freq="5m",
        )
        logger.info(
            "EMA cross ({}/{}) strategy complete — {} trades",
            fast,
            slow,
            pf.trades.count(),
        )
        return self._extract_metrics(pf)

    # ------------------------------------------------------------------
    # Strategy: Bollinger Band fade
    # ------------------------------------------------------------------

    def run_bollinger_strategy(
        self, df: pd.DataFrame, period: int = 20, std: float = 2.0
    ) -> dict:
        """Run Bollinger Band fade strategy.

        * Entry long — price touches the lower band.
        * Exit long  — price touches the middle band (SMA).

        Returns:
            Dictionary with standardized performance metrics.
        """
        _require_vbt()
        close = df["Close"]

        bb = vbt.BBANDS.run(close, window=period, alpha=std)
        lower = bb.lower
        middle = bb.middle

        # Entry: close crosses below or touches the lower band
        entries = close.vbt.crossed_below(lower)
        # Exit: close crosses above or touches the middle band
        exits = close.vbt.crossed_above(middle)

        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=self._initial_capital,
            fees=self._fees,
            freq="5m",
        )
        logger.info(
            "Bollinger Band strategy (period={}, std={}) complete — {} trades",
            period,
            std,
            pf.trades.count(),
        )
        return self._extract_metrics(pf)

    # ------------------------------------------------------------------
    # Strategy: VWAP mean reversion
    # ------------------------------------------------------------------

    def run_vwap_strategy(
        self, df: pd.DataFrame, deviation_pct: float = 0.003
    ) -> dict:
        """Run VWAP mean-reversion strategy.

        * Entry long — price < VWAP * (1 - deviation).
        * Exit long  — price >= VWAP.

        If the DataFrame does not contain a pre-computed ``Vwap`` column the
        VWAP is calculated from the typical price and volume on a per-day basis.

        Returns:
            Dictionary with standardized performance metrics.
        """
        _require_vbt()
        close = df["Close"].copy()

        # Compute intraday VWAP if not already present
        if "Vwap" in df.columns:
            vwap_series = df["Vwap"].copy()
        else:
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
            cum_vol = df["Volume"].copy()
            cum_tp_vol = (typical_price * df["Volume"]).copy()

            # Group by trading day so VWAP resets each session
            if hasattr(df.index, "date"):
                day_groups = pd.Series(df.index.date, index=df.index)
            else:
                day_groups = pd.Series(0, index=df.index)

            cum_vol = cum_vol.groupby(day_groups).cumsum()
            cum_tp_vol = cum_tp_vol.groupby(day_groups).cumsum()
            vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)
            vwap_series.ffill(inplace=True)

        lower_band = vwap_series * (1.0 - deviation_pct)

        entries = close < lower_band
        exits = close >= vwap_series

        # Avoid entering and exiting on the same bar — clear entry where exit is also true
        overlap = entries & exits
        entries[overlap] = False

        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=self._initial_capital,
            fees=self._fees,
            freq="5m",
        )
        logger.info(
            "VWAP reversion strategy (dev={:.4f}) complete — {} trades",
            deviation_pct,
            pf.trades.count(),
        )
        return self._extract_metrics(pf)

    # ------------------------------------------------------------------
    # Strategy: Opening Range Breakout (ORB)
    # ------------------------------------------------------------------

    def run_orb_strategy(
        self,
        df: pd.DataFrame,
        orb_minutes: int = 60,
        volume_multiplier: float = 1.5,
    ) -> dict:
        """Run Opening Range Breakout strategy.

        * Capture the high/low of the first *orb_minutes* bars per session.
        * Entry long — close > ORB high **and** bar volume > average volume
          of ORB window * *volume_multiplier*.
        * Exit (stop) — close < ORB low.

        Returns:
            Dictionary with standardized performance metrics.
        """
        _require_vbt()
        close = df["Close"].copy()
        high = df["High"].copy()
        low = df["Low"].copy()
        volume = df["Volume"].copy()

        # Determine bar duration from index
        if len(df) >= 2 and hasattr(df.index, "freq"):
            bar_freq = df.index.freq
            if bar_freq is not None:
                bar_minutes = int(bar_freq.delta.total_seconds() / 60)
            else:
                bar_minutes = int(
                    (df.index[1] - df.index[0]).total_seconds() / 60
                )
        elif len(df) >= 2:
            bar_minutes = max(
                1, int((df.index[1] - df.index[0]).total_seconds() / 60)
            )
        else:
            bar_minutes = 5

        bars_in_orb = max(1, orb_minutes // bar_minutes)

        # Build per-day ORB high / low / avg-volume arrays
        orb_high = pd.Series(np.nan, index=df.index, dtype=float)
        orb_low = pd.Series(np.nan, index=df.index, dtype=float)
        orb_avg_vol = pd.Series(np.nan, index=df.index, dtype=float)

        if hasattr(df.index, "date"):
            dates = pd.Series(df.index.date, index=df.index)
        else:
            # If no datetime index, treat entire df as one session
            dates = pd.Series(0, index=df.index)

        for day, idx in dates.groupby(dates).groups.items():
            idx_sorted = idx.sort_values() if hasattr(idx, "sort_values") else sorted(idx)
            orb_slice = idx_sorted[:bars_in_orb]
            oh = high.loc[orb_slice].max()
            ol = low.loc[orb_slice].min()
            av = volume.loc[orb_slice].mean()
            # Only apply to bars AFTER the opening range
            post_orb = idx_sorted[bars_in_orb:]
            orb_high.loc[post_orb] = oh
            orb_low.loc[post_orb] = ol
            orb_avg_vol.loc[post_orb] = av

        # Entry: close breaks above ORB high with volume confirmation
        vol_threshold = orb_avg_vol * volume_multiplier
        entries = (close > orb_high) & (volume > vol_threshold)

        # Exit: close drops below ORB low (stop hit)
        exits = close < orb_low

        # Drop NaN rows (opening-range bars with no ORB reference)
        valid = orb_high.notna()
        entries = entries & valid
        exits = exits & valid

        # Avoid entering and exiting on the same bar
        overlap = entries & exits
        entries[overlap] = False

        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            init_cash=self._initial_capital,
            fees=self._fees,
            freq="5m",
        )
        logger.info(
            "ORB strategy (window={}min, vol_mult={}) complete — {} trades",
            orb_minutes,
            volume_multiplier,
            pf.trades.count(),
        )
        return self._extract_metrics(pf)

    # ------------------------------------------------------------------
    # Run all strategies
    # ------------------------------------------------------------------

    def run_all(self, df: pd.DataFrame, config: dict) -> dict[str, dict]:
        """Run all five strategies on a single DataFrame.

        Args:
            config: The ``strategies`` section of ``config.json``.

        Returns:
            Dictionary mapping strategy name to its metrics dictionary.
        """
        _require_vbt()
        results: dict[str, dict] = {}

        # RSI momentum
        rsi_cfg = config.get("rsi_momentum", {})
        try:
            results["rsi_momentum"] = self.run_rsi_strategy(
                df,
                rsi_period=14,
                oversold=rsi_cfg.get("rsi_oversold", 32),
                overbought=rsi_cfg.get("rsi_overbought", 68),
            )
        except Exception as exc:
            logger.error("RSI strategy failed: {}", exc)

        # EMA cross
        ema_cfg = config.get("ema_cross", {})
        try:
            results["ema_cross"] = self.run_ema_cross_strategy(
                df,
                fast=ema_cfg.get("fast_ema", 50),
                slow=ema_cfg.get("slow_ema", 200),
            )
        except Exception as exc:
            logger.error("EMA cross strategy failed: {}", exc)

        # Bollinger fade
        bb_cfg = config.get("bollinger_fade", {})
        try:
            results["bollinger_fade"] = self.run_bollinger_strategy(
                df,
                period=bb_cfg.get("bb_period", 20),
                std=bb_cfg.get("bb_std", 2.0),
            )
        except Exception as exc:
            logger.error("Bollinger fade strategy failed: {}", exc)

        # VWAP reversion
        vwap_cfg = config.get("vwap_reversion", {})
        try:
            # config stores as 0.3 (percent); strategy expects fraction
            dev_pct = vwap_cfg.get("vwap_deviation_pct", 0.3)
            if dev_pct > 0.1:
                dev_pct = dev_pct / 100.0
            results["vwap_reversion"] = self.run_vwap_strategy(
                df, deviation_pct=dev_pct
            )
        except Exception as exc:
            logger.error("VWAP reversion strategy failed: {}", exc)

        # First candle / ORB
        orb_cfg = config.get("first_candle", {})
        try:
            results["first_candle"] = self.run_orb_strategy(
                df,
                orb_minutes=orb_cfg.get("orb_window_minutes", 60),
                volume_multiplier=orb_cfg.get("volume_multiplier", 1.5),
            )
        except Exception as exc:
            logger.error("ORB strategy failed: {}", exc)

        logger.info(
            "Completed {} / 5 strategies", len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metrics(portfolio) -> dict:
        """Extract standardized metrics from a ``vbt.Portfolio`` object.

        Returns:
            Dictionary containing ``total_return``, ``profit_factor``,
            ``win_rate``, ``max_drawdown``, ``max_drawdown_pct``,
            ``sharpe_ratio``, ``total_trades``, ``avg_win``, ``avg_loss``,
            and ``calmar_ratio``.
        """
        stats = portfolio.stats()

        total_return = float(portfolio.total_return())
        total_trades = int(portfolio.trades.count())

        # Win rate
        if total_trades > 0:
            winning = int((portfolio.trades.pnl.values > 0).sum())
            win_rate = winning / total_trades
        else:
            win_rate = 0.0

        # Profit factor: gross wins / gross losses
        if total_trades > 0:
            pnl_values = portfolio.trades.pnl.values
            gross_wins = float(pnl_values[pnl_values > 0].sum())
            gross_losses = float(abs(pnl_values[pnl_values < 0].sum()))
            profit_factor = (
                gross_wins / gross_losses if gross_losses > 0 else float("inf")
            )
        else:
            profit_factor = 0.0

        # Drawdown
        max_dd = float(portfolio.max_drawdown())
        max_dd_pct = max_dd  # vbt returns this as a ratio already

        # Sharpe ratio
        try:
            sharpe = float(stats.get("Sharpe Ratio", 0.0))
        except (KeyError, TypeError):
            sharpe = 0.0

        # Calmar ratio
        try:
            calmar = float(stats.get("Calmar Ratio", 0.0))
        except (KeyError, TypeError):
            calmar = 0.0

        # Average win / average loss
        if total_trades > 0:
            pnl_values = portfolio.trades.pnl.values
            wins = pnl_values[pnl_values > 0]
            losses = pnl_values[pnl_values < 0]
            avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        else:
            avg_win = 0.0
            avg_loss = 0.0

        return {
            "total_return": round(total_return, 6),
            "profit_factor": round(profit_factor, 4),
            "win_rate": round(win_rate, 4),
            "max_drawdown": round(max_dd, 6),
            "max_drawdown_pct": round(max_dd_pct, 6),
            "sharpe_ratio": round(sharpe, 4),
            "total_trades": total_trades,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "calmar_ratio": round(calmar, 4),
        }
