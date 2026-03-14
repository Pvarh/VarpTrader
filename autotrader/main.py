"""AutoTrader v2 -- main entry point and scheduler.

Orchestrates all trading subsystems: data feeds, signals, whale monitoring,
risk management, order execution, journaling, alerts, nightly analysis,
swing bias advisor, backtesting, optimisation, and the live dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
import signal as signal_mod
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

# -- Logging setup (loguru) -------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(
    LOG_DIR / "autotrader.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
)

# -- Local imports -----------------------------------------------------------
from journal.db import TradeDatabase
from journal.models import Trade
from data.feed_stocks import StockFeed
from data.feed_crypto import CryptoFeed
from signals.first_candle import FirstCandleSignal
from signals.ema_cross import EMACrossSignal
from signals.vwap_reversion import VWAPReversionSignal
from signals.rsi_momentum import RSIMomentumSignal
from signals.bollinger_fade import BollingerFadeSignal
from signals.base_signal import SignalResult, SignalDirection
from signals.indicators import Indicators
from whale.block_trades import BlockTradeDetector
from whale.onchain import OnChainWhaleDetector
from whale.orderbook import OrderBookScanner
from risk.position_sizer import PositionSizer
from risk.kill_switch import KillSwitch
from risk.reward_ratio import RewardRatioGate
from execution.alpaca_executor import AlpacaExecutor
from execution.crypto_executor import CryptoExecutor
from execution.order_validator import OrderValidator
from alerts.telegram_bot import TelegramAlert
from analysis.analyzer import PerformanceAnalyzer
from analysis.llm_advisor import LLMAdvisor
from analysis.config_updater import ConfigUpdater
from analysis.report_builder import ReportBuilder
from analysis.swing_advisor import SwingAdvisor
from execution.position_monitor import PositionMonitor
from execution.paper_engine import PaperPortfolio, PaperExecutor
from execution.trade_lifecycle import TradeManager, TradeContext, TradeState

# -- FastAPI app setup -------------------------------------------------------
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from dashboard.router import router as dashboard_router, init as dashboard_init

app = FastAPI(title="AutoTrader")
app.include_router(dashboard_router)
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "dashboard" / "static")),
    name="static",
)

# -- Environment & config ---------------------------------------------------
load_dotenv()

CONFIG_PATH = Path("config.json")


def load_config() -> dict:
    """Load and return the config.json file."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# AutoTrader orchestrator
# ============================================================================

class AutoTrader:
    """Central orchestrator for the automated trading system."""

    def __init__(self) -> None:
        """Initialize all subsystems from config and environment variables."""
        self.config = load_config()
        self.paper_trade = os.getenv("PAPER_TRADE", "true").lower() == "true"
        logger.info("initializing_autotrader | paper_trade={}", self.paper_trade)

        # -- Database --------------------------------------------------------
        db_path = os.getenv("DB_PATH", "data/trades.db")
        self.db = TradeDatabase(db_path)

        # -- Data feeds ------------------------------------------------------
        self.stock_feed = StockFeed(
            alpaca_api_key=os.getenv("ALPACA_API_KEY"),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY"),
            polygon_api_key=os.getenv("POLYGON_API_KEY"),
        )
        self.crypto_feed = CryptoFeed(
            exchange_id="binance",
            api_key=os.getenv("BINANCE_API_KEY"),
            secret=os.getenv("BINANCE_SECRET_KEY"),
        )

        # -- Signals (from config["strategies"]) -----------------------------
        strat_cfg = self.config["strategies"]
        self.signals = [
            FirstCandleSignal(strat_cfg["first_candle"]),
            EMACrossSignal(strat_cfg["ema_cross"]),
            VWAPReversionSignal(strat_cfg["vwap_reversion"]),
            RSIMomentumSignal(strat_cfg["rsi_momentum"]),
            BollingerFadeSignal(strat_cfg["bollinger_fade"]),
        ]

        # -- Whale monitoring (new config keys) ------------------------------
        whale_cfg = self.config["whale"]
        self.block_detector = BlockTradeDetector(
            api_key=os.getenv("POLYGON_API_KEY", ""),
            min_shares=whale_cfg.get("stock_block_shares", 50000),
            min_value=whale_cfg.get("stock_block_usd", 2_000_000),
            flag_duration_min=whale_cfg.get("flag_ttl_minutes", 15),
        )
        self.onchain_detector = OnChainWhaleDetector(
            api_key=os.getenv("WHALE_ALERT_API_KEY", ""),
            min_usd=whale_cfg.get("crypto_transfer_usd", 1_000_000),
            flag_duration_min=whale_cfg.get("flag_ttl_minutes", 15),
        )
        self.orderbook_scanner = OrderBookScanner()

        # -- Risk ------------------------------------------------------------
        risk_cfg = self.config["risk"]
        self.position_sizer = PositionSizer(risk_cfg)
        self.kill_switch = KillSwitch(
            {"daily_loss_limit_pct": risk_cfg.get("daily_loss_limit_pct", 0.03)}
        )
        self.reward_gate = RewardRatioGate(
            min_ratio=risk_cfg.get("min_reward_ratio", 2.0)
        )
        self.order_validator = OrderValidator(risk_cfg)

        # -- Execution -------------------------------------------------------
        self.stock_executor = AlpacaExecutor(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            base_url=os.getenv(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ),
            paper=self.paper_trade,
        )
        self.crypto_executor = CryptoExecutor(
            exchange_id="binance",
            api_key=os.getenv("BINANCE_API_KEY"),
            secret=os.getenv("BINANCE_SECRET_KEY"),
        )

        # -- Paper trading engine --------------------------------------------
        if self.paper_trade:
            self.paper_portfolio = PaperPortfolio(initial_capital=100_000.0)
            self.paper_executor = PaperExecutor(
                portfolio=self.paper_portfolio, db=self.db, slippage_pct=0.001
            )
            logger.info("paper_trading_engine_initialized")
        else:
            self.paper_portfolio = None
            self.paper_executor = None

        # -- Alerts ----------------------------------------------------------
        self.telegram = TelegramAlert(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

        # -- Analysis --------------------------------------------------------
        self.analyzer = PerformanceAnalyzer(self.db)
        analysis_cfg = self.config.get("analysis", {})
        self.llm_advisor = LLMAdvisor(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=analysis_cfg.get("model", "claude-sonnet-4-6"),
        )
        self.config_updater = ConfigUpdater(str(CONFIG_PATH))
        self.report_builder = ReportBuilder()

        # -- Swing advisor ---------------------------------------------------
        swing_cfg = self.config.get("swing_advisor", {})
        self.swing_advisor = SwingAdvisor(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            news_api_key=os.getenv("NEWS_API_KEY", ""),
            db=self.db,
            model=swing_cfg.get("model", "claude-sonnet-4-6"),
            min_confidence=swing_cfg.get("min_confidence_to_block", 60),
        )

        # -- Position monitor ------------------------------------------------
        self.position_monitor = PositionMonitor(
            db=self.db,
            stock_feed=self.stock_feed,
            crypto_feed=self.crypto_feed,
            stock_executor=self.stock_executor,
            crypto_executor=self.crypto_executor,
            telegram=self.telegram,
            paper_trade=self.paper_trade,
        )

        # -- Trade lifecycle -------------------------------------------------
        self.trade_manager = TradeManager()

        # -- State -----------------------------------------------------------
        self._daily_trade_count = 0
        self._running = True

    # ====================================================================
    # Config hot-reload
    # ====================================================================
    def reload_config(self) -> None:
        """Reload config.json from disk and re-apply volatile settings."""
        try:
            self.config = load_config()
            logger.debug("config_reloaded")
        except Exception:
            logger.exception("config_reload_error")

    # ====================================================================
    # Swing advisor
    # ====================================================================
    def run_swing_advisor(self) -> None:
        """Run the weekly swing bias advisor for all watchlist symbols."""
        logger.info("swing_advisor_starting")
        try:
            stocks = self.config.get("watchlist", {}).get("stocks", [])
            crypto = self.config.get("watchlist", {}).get("crypto", [])
            symbols = stocks + crypto
            if not symbols:
                logger.warning("swing_advisor_no_symbols")
                return
            summary = self.swing_advisor.run(symbols)

            lines = ["Weekly Swing Bias Update"]
            biases = summary.get("biases", {})
            for sym, bias in biases.items():
                lines.append(f"  {sym}: {bias}")
            self.telegram.send_message("\n".join(lines))
            logger.info("swing_advisor_complete | biases={}", biases)
        except Exception:
            logger.exception("swing_advisor_error")

    # ====================================================================
    # Stock scanning loop
    # ====================================================================
    def scan_stocks(self) -> None:
        """Run signal evaluation across all configured stock symbols."""
        if not self._running or self.kill_switch.halted:
            return
        logger.info("scanning_stocks")
        account = self.stock_executor.get_account()
        account_value = float(account.get("equity", 0)) if account else 0.0
        if account_value <= 0:
            logger.warning("no_account_equity")
            return

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl = self.db.get_daily_pnl(today_str)
        if not self.kill_switch.is_trading_allowed(daily_pnl, account_value):
            logger.warning("kill_switch_triggered | daily_pnl={}", daily_pnl)
            self.telegram.send_kill_switch_alert()
            return

        positions = self.stock_executor.get_positions()
        current_positions = len(positions) if positions else 0

        for symbol in self.config["watchlist"]["stocks"]:
            try:
                self._evaluate_stock_symbol(
                    symbol, account_value, current_positions
                )
            except Exception:
                logger.exception("stock_scan_error | symbol={}", symbol)

    def _evaluate_stock_symbol(
        self, symbol: str, account_value: float, current_positions: int
    ) -> None:
        """Evaluate all enabled signals for a single stock symbol."""
        candles_5m = self.stock_feed.get_historical_bars(
            symbol, period="2d", interval="5m"
        )
        if not candles_5m:
            return

        current_price = candles_5m[-1].close
        avg_volume = self.stock_feed.get_average_volume(symbol)
        first_candle = self.stock_feed.get_first_candle(symbol)

        candles_1h = self.stock_feed.get_historical_bars(
            symbol, period="1mo", interval="1h"
        )

        whale_flag = 0
        if self.block_detector.has_sell_flag(symbol):
            logger.info("whale_sell_flag_active | symbol={}", symbol)
            whale_flag = 1

        for sig in self.signals:
            if not sig.is_enabled():
                continue
            try:
                result = self._run_signal(
                    sig, symbol, candles_5m, current_price, "stock",
                    first_candle, avg_volume, candles_1h=candles_1h,
                )
                if not result.triggered:
                    continue

                if (
                    result.direction == SignalDirection.LONG
                    and self.block_detector.has_sell_flag(symbol)
                ):
                    logger.info(
                        "whale_suppressed_long | symbol={} strategy={}",
                        symbol, sig.name,
                    )
                    continue

                self._process_signal(
                    result, symbol, "stock", account_value,
                    current_positions, whale_flag,
                )
            except Exception:
                logger.exception(
                    "signal_eval_error | symbol={} strategy={}",
                    symbol, sig.name,
                )

    # ====================================================================
    # Crypto scanning loop
    # ====================================================================
    def scan_crypto(self) -> None:
        """Run signal evaluation across all configured crypto symbols."""
        if not self._running or self.kill_switch.halted:
            return
        logger.info("scanning_crypto")
        balance = self.crypto_executor.get_balance()
        account_value = (
            float(balance.get("total", {}).get("USDT", 0)) if balance else 0.0
        )
        if account_value <= 0:
            logger.warning("no_crypto_balance")
            return

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl = self.db.get_daily_pnl(today_str)
        if not self.kill_switch.is_trading_allowed(daily_pnl, account_value):
            logger.warning("kill_switch_triggered_crypto | daily_pnl={}", daily_pnl)
            self.telegram.send_kill_switch_alert()
            return

        crypto_positions = self.crypto_executor.get_positions()
        current_positions = len(crypto_positions) if crypto_positions else 0

        for symbol in self.config["watchlist"]["crypto"]:
            try:
                self._evaluate_crypto_symbol(
                    symbol, account_value, current_positions
                )
            except Exception:
                logger.exception("crypto_scan_error | symbol={}", symbol)

    def _evaluate_crypto_symbol(
        self, symbol: str, account_value: float, current_positions: int
    ) -> None:
        """Evaluate all enabled signals for a single crypto symbol."""
        candles_5m = self.crypto_feed.get_historical_bars(
            symbol, timeframe="5m", limit=200
        )
        if not candles_5m:
            return

        current_price = candles_5m[-1].close
        base_symbol = symbol.split("/")[0].upper()

        candles_1h = self.crypto_feed.get_historical_bars(
            symbol, timeframe="1h", limit=250
        )

        whale_flag = 0
        if self.onchain_detector.has_sell_pressure(base_symbol):
            logger.info("onchain_sell_pressure | symbol={}", symbol)
            whale_flag = 1

        for sig in self.signals:
            if not sig.is_enabled():
                continue
            try:
                result = self._run_signal(
                    sig, symbol, candles_5m, current_price, "crypto",
                    None, 0.0, candles_1h=candles_1h,
                )
                if not result.triggered:
                    continue

                if (
                    result.direction == SignalDirection.LONG
                    and self.onchain_detector.has_sell_pressure(base_symbol)
                ):
                    logger.info(
                        "whale_suppressed_long_crypto | symbol={} strategy={}",
                        symbol, sig.name,
                    )
                    continue

                self._process_signal(
                    result, symbol, "crypto", account_value,
                    current_positions, whale_flag,
                )
            except Exception:
                logger.exception(
                    "signal_eval_error_crypto | symbol={} strategy={}",
                    symbol, sig.name,
                )

    # ====================================================================
    # Signal evaluation helpers
    # ====================================================================
    def _run_signal(
        self,
        sig,
        symbol: str,
        candles: list,
        current_price: float,
        market: str,
        first_candle=None,
        avg_volume: float = 0.0,
        candles_1h: list | None = None,
    ) -> SignalResult:
        """Run a specific signal's evaluation method."""
        if isinstance(sig, FirstCandleSignal) and first_candle:
            now = datetime.now(timezone.utc)
            market_open = now.replace(hour=13, minute=30, second=0, microsecond=0)
            minutes_since = max(
                0, int((now - market_open).total_seconds() / 60)
            )
            return sig.evaluate_with_context(
                symbol=symbol,
                first_candle=first_candle,
                current_candle=candles[-1],
                avg_volume=avg_volume,
                minutes_since_open=minutes_since,
            )
        elif isinstance(sig, RSIMomentumSignal):
            closes_5m = [c.close for c in candles]
            rsi_5m_series = Indicators.rsi(closes_5m, 14)
            rsi_5m = rsi_5m_series[-1] if rsi_5m_series else 50.0

            if candles_1h and len(candles_1h) >= 15:
                closes_1h = [c.close for c in candles_1h]
                rsi_1h_series = Indicators.rsi(closes_1h, 14)
                rsi_1h = rsi_1h_series[-1] if rsi_1h_series else 50.0
            else:
                rsi_1h = 50.0

            return sig.evaluate_from_rsi(
                symbol=symbol, rsi_5m=rsi_5m, rsi_1h=rsi_1h,
                current_price=current_price,
            )
        elif isinstance(sig, VWAPReversionSignal):
            vwap_series = Indicators.vwap(candles)
            vwap = vwap_series[-1] if vwap_series else current_price

            now = datetime.now(timezone.utc)
            market_open = now.replace(hour=13, minute=30)
            market_close = now.replace(hour=20, minute=0)
            mins_since = max(
                0, int((now - market_open).total_seconds() / 60)
            )
            mins_before = max(
                0, int((market_close - now).total_seconds() / 60)
            )
            return sig.evaluate_from_vwap(
                symbol=symbol, current_price=current_price, vwap=vwap,
                recent_candles=candles[-3:], minutes_since_open=mins_since,
                minutes_before_close=mins_before,
            )
        elif isinstance(sig, BollingerFadeSignal):
            closes = [c.close for c in candles]
            if len(closes) >= 20:
                upper_series, middle_series, lower_series = (
                    Indicators.bollinger_bands(closes, period=20, std_dev=2.0)
                )
                upper = upper_series[-1]
                middle = middle_series[-1]
                lower = lower_series[-1]

                rsi_series = Indicators.rsi(closes, 14)
                rsi = rsi_series[-1] if rsi_series else 50.0

                return sig.evaluate_from_bands(
                    symbol=symbol, current_price=current_price,
                    lower_band=lower, upper_band=upper, middle_band=middle,
                    rsi=rsi, prev_candle=candles[-2],
                )
        elif isinstance(sig, EMACrossSignal):
            ema_candles = (
                candles_1h
                if candles_1h and len(candles_1h) >= 200
                else candles
            )
            closes = [c.close for c in ema_candles]

            fast_period = sig.config.get("fast_period", 50)
            slow_period = sig.config.get("slow_period", 200)

            fast_ema = Indicators.ema(closes, fast_period)
            slow_ema = Indicators.ema(closes, slow_period)

            if (
                len(fast_ema) >= 2
                and len(slow_ema) >= 2
                and not (fast_ema[-1] != fast_ema[-1])
                and not (slow_ema[-1] != slow_ema[-1])
                and not (fast_ema[-2] != fast_ema[-2])
                and not (slow_ema[-2] != slow_ema[-2])
            ):
                prev_fast = fast_ema[-2]
                prev_slow = slow_ema[-2]
                curr_fast = fast_ema[-1]
                curr_slow = slow_ema[-1]
            else:
                return SignalResult(triggered=False, strategy_name=sig.name)

            rsi_series = Indicators.rsi(closes, 14)
            rsi = rsi_series[-1] if rsi_series else 50.0

            return sig.evaluate_from_emas(
                symbol=symbol,
                prev_fast_ema=prev_fast,
                prev_slow_ema=prev_slow,
                curr_fast_ema=curr_fast,
                curr_slow_ema=curr_slow,
                rsi=rsi, current_price=current_price,
            )

        return sig.evaluate(symbol, candles, current_price, market)

    @staticmethod
    def _compute_rsi(candles: list, period: int = 14) -> float:
        """Compute RSI from a list of OHLCV candles."""
        if not candles:
            return 50.0
        closes = [c.close for c in candles]
        rsi_series = Indicators.rsi(closes, period)
        return rsi_series[-1] if rsi_series else 50.0

    # ====================================================================
    # Order processing
    # ====================================================================
    def _process_signal(
        self,
        result: SignalResult,
        symbol: str,
        market: str,
        account_value: float,
        current_positions: int,
        whale_flag: int,
    ) -> None:
        """Validate, size, execute, and journal a triggered signal."""
        if result.direction is None:
            return

        if not self.reward_gate.check(
            result.entry_price, result.stop_loss, result.take_profit
        ):
            logger.info(
                "reward_ratio_rejected | symbol={} strategy={}",
                symbol, result.strategy_name,
            )
            return

        candles = (
            self.stock_feed.get_historical_bars(symbol)
            if market == "stock"
            else self.crypto_feed.get_historical_bars(symbol)
        )
        atr = (
            self._compute_atr(candles)
            if candles
            else result.entry_price * 0.01
        )
        quantity = self.position_sizer.calculate_size(
            account_value, result.entry_price, atr
        )

        side = "buy" if result.direction == SignalDirection.LONG else "sell"

        valid, reason = self.order_validator.validate(
            symbol, side, quantity, current_positions,
            self._daily_trade_count, self.kill_switch.halted,
        )
        if not valid:
            logger.info("order_rejected | symbol={} reason={}", symbol, reason)
            return

        order_result = None
        if self.paper_trade and self.paper_executor:
            order_result = self.paper_executor.submit_market_order(
                symbol=symbol, side=side, quantity=float(quantity),
                market_price=result.entry_price, market=market,
                strategy=result.strategy_name,
            )
        else:
            if market == "stock":
                order_result = self.stock_executor.submit_market_order(
                    symbol, quantity, side
                )
            else:
                order_result = self.crypto_executor.submit_market_order(
                    symbol, side, float(quantity)
                )

        trade = Trade(
            symbol=symbol,
            market=market,
            strategy=result.strategy_name,
            direction=result.direction.value,
            entry_price=result.entry_price,
            quantity=quantity,
            stop_loss=result.stop_loss,
            take_profit=result.take_profit,
            whale_flag=whale_flag,
            paper_trade=1 if self.paper_trade else 0,
        )
        trade_id = self.db.insert_trade(trade)
        self._daily_trade_count += 1

        alert_data = trade.to_dict()
        alert_data["action"] = "ENTRY"
        alert_data["trade_id"] = trade_id
        self.telegram.send_trade_alert(alert_data)

        logger.info(
            "trade_opened | trade_id={} symbol={} strategy={} direction={} "
            "entry={} paper={}",
            trade_id, symbol, result.strategy_name,
            result.direction.value, result.entry_price, self.paper_trade,
        )

    @staticmethod
    def _compute_atr(candles: list, period: int = 14) -> float:
        """Compute Average True Range from OHLCV candles."""
        if not candles or len(candles) < 2:
            return 0.0
        atr_series = Indicators.atr(candles, period)
        return atr_series[-1] if atr_series else 0.0

    # ====================================================================
    # Whale polling
    # ====================================================================
    def poll_whale_stocks(self) -> None:
        """Poll for stock block trades."""
        try:
            self.block_detector.poll(self.config["watchlist"]["stocks"])
        except Exception:
            logger.exception("whale_stock_poll_error")

    def poll_whale_crypto(self) -> None:
        """Poll for on-chain whale movements."""
        try:
            self.onchain_detector.poll()
        except Exception:
            logger.exception("whale_crypto_poll_error")

    # ====================================================================
    # Nightly analysis
    # ====================================================================
    def run_nightly_analysis(self) -> None:
        """Execute the AI-powered nightly analysis pipeline."""
        logger.info("nightly_analysis_starting")
        try:
            analysis_cfg = self.config.get("analysis", {})
            lookback = analysis_cfg.get("lookback_days", 30)
            report_data = self.analyzer.build_full_report(lookback_days=lookback)

            min_trades = analysis_cfg.get("min_trades_for_suggestion", 15)
            if report_data.get("total_trades", 0) < min_trades:
                logger.info(
                    "insufficient_trades_for_analysis | count={}",
                    report_data.get("total_trades", 0),
                )
                return

            recommendations = self.llm_advisor.get_recommendations(report_data)
            approved, rejected = self.config_updater.validate_changes(
                recommendations
            )
            if approved:
                auto_apply = analysis_cfg.get("auto_apply_changes", False)
                if auto_apply:
                    self.config_updater.apply_changes(approved)
                    self.config = load_config()
                    logger.info("config_updated | changes={}", approved)
                else:
                    logger.info(
                        "config_changes_pending_approval | changes={}", approved
                    )

            report_md = self.report_builder.build_daily_report(
                report_data, config_changes=approved, rejected=rejected
            )
            self.telegram.send_daily_report(report_md)

            from journal.models import AnalysisRun

            run = AnalysisRun(
                trades_analyzed=report_data.get("total_trades", 0),
                report_markdown=report_md,
                config_changes_json=json.dumps(approved) if approved else None,
                approved=1 if (approved and auto_apply) else 0,
            )
            self.db.insert_analysis_run(run)
            logger.info("nightly_analysis_complete")

        except Exception:
            logger.exception("nightly_analysis_error")

    # ====================================================================
    # Daily reset
    # ====================================================================
    def daily_reset(self) -> None:
        """Reset daily counters and kill switch at market open."""
        logger.info("daily_reset")
        self._daily_trade_count = 0
        self.kill_switch.reset()
        self.config = load_config()

    # ====================================================================
    # Shutdown
    # ====================================================================
    def shutdown(self) -> None:
        """Graceful shutdown handler."""
        logger.info("shutting_down")
        self._running = False
        self.stock_feed.stop_stream()
        self.db.close()


# ============================================================================
# Dashboard helpers
# ============================================================================

def _start_dashboard(host: str, port: int) -> None:
    """Run uvicorn in the calling thread (intended for daemon threads)."""
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ============================================================================
# CLI commands
# ============================================================================

def run_live(args) -> None:
    """Run the live/paper trading scheduler with background dashboard."""
    trader = AutoTrader()
    scheduler = BlockingScheduler()

    # -- Initialize dashboard with shared state ----------------------------
    dashboard_init(db=trader.db, analyzer=trader.analyzer, config=trader.config)

    # -- Start dashboard in background thread ------------------------------
    dashboard_cfg = trader.config.get("dashboard", {})
    dash_host = dashboard_cfg.get("host", "0.0.0.0")
    dash_port = dashboard_cfg.get("port", 8000)
    dash_thread = threading.Thread(
        target=_start_dashboard,
        args=(dash_host, dash_port),
        daemon=True,
    )
    dash_thread.start()
    logger.info(
        "dashboard_started | host={} port={}",
        dash_host, dash_port,
    )

    # -- Start Alpaca WebSocket for real-time stock prices -----------------
    stock_symbols = trader.config.get("watchlist", {}).get("stocks", [])
    if stock_symbols:
        trader.stock_feed.start_stream(stock_symbols)

    # -- Parse nightly analysis time from config["analysis"]["nightly_run_time"]
    nightly_time = trader.config.get("analysis", {}).get(
        "nightly_run_time", "23:30"
    )
    nightly_hour, nightly_minute = (
        int(nightly_time.split(":")[0]),
        int(nightly_time.split(":")[1]),
    )

    tz = "US/Eastern"

    # -- Stock scanning every 5 minutes ------------------------------------
    scheduler.add_job(
        trader.scan_stocks,
        IntervalTrigger(minutes=5),
        id="scan_stocks",
        max_instances=1,
    )

    # -- Crypto scanning every 3 minutes (24/7) ----------------------------
    scheduler.add_job(
        trader.scan_crypto,
        IntervalTrigger(minutes=3),
        id="scan_crypto",
        max_instances=1,
    )

    # -- Whale polling -----------------------------------------------------
    whale_cfg = trader.config.get("whale", {})
    scheduler.add_job(
        trader.poll_whale_stocks,
        IntervalTrigger(
            seconds=whale_cfg.get("poll_interval_stocks_sec", 60)
        ),
        id="whale_stocks",
        max_instances=1,
    )
    scheduler.add_job(
        trader.poll_whale_crypto,
        IntervalTrigger(
            seconds=whale_cfg.get("poll_interval_crypto_sec", 30)
        ),
        id="whale_crypto",
        max_instances=1,
    )

    # -- Position monitor every 30 seconds ---------------------------------
    scheduler.add_job(
        trader.position_monitor.check_open_positions,
        IntervalTrigger(seconds=30),
        id="position_monitor",
        max_instances=1,
    )

    # -- Config hot-reload every 60 seconds --------------------------------
    scheduler.add_job(
        trader.reload_config,
        IntervalTrigger(seconds=60),
        id="config_reload",
        max_instances=1,
    )

    # -- Daily reset at 09:30 ET ------------------------------------------
    scheduler.add_job(
        trader.daily_reset,
        CronTrigger(hour=9, minute=30, timezone=tz),
        id="daily_reset",
    )

    # -- Nightly analysis --------------------------------------------------
    scheduler.add_job(
        trader.run_nightly_analysis,
        CronTrigger(
            hour=nightly_hour,
            minute=nightly_minute,
            timezone=tz,
        ),
        id="nightly_analysis",
    )

    # -- Swing advisor every Monday at 06:00 -------------------------------
    scheduler.add_job(
        trader.run_swing_advisor,
        CronTrigger(day_of_week="mon", hour=6, minute=0, timezone=tz),
        id="swing_advisor",
    )

    # -- Signal handlers ---------------------------------------------------
    def _signal_handler(sig, frame):
        trader.shutdown()
        scheduler.shutdown(wait=False)

    signal_mod.signal(signal_mod.SIGINT, _signal_handler)
    signal_mod.signal(signal_mod.SIGTERM, _signal_handler)

    logger.info("autotrader_started | paper_trade={}", trader.paper_trade)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        trader.shutdown()


def run_backtest(args) -> None:
    """Run backtest on historical data (vectorbt preferred, fallback to custom)."""
    config = load_config()

    try:
        from backtesting.vectorbt_runner import VectorBTRunner
        from backtesting.performance_report import PerformanceReport

        runner = VectorBTRunner(initial_capital=args.capital)
        data = runner.load_data([args.symbol], args.start, args.end)

        if args.symbol not in data:
            print(f"No data loaded for {args.symbol}")
            return

        results = runner.run_all(data[args.symbol], config["strategies"])

        report = PerformanceReport(config)
        print(report.build_report(results))

    except ImportError:
        logger.info("vectorbt_unavailable -- falling back to custom engine")

        from backtest.engine import BacktestEngine
        from backtest.data_loader import HistoricalDataLoader

        loader = HistoricalDataLoader()
        symbol = args.symbol
        market = "crypto" if "/" in symbol else "stock"

        if hasattr(args, "csv") and args.csv:
            candles = loader.load_from_csv(args.csv, symbol, market, "5m")
        else:
            candles = loader.load_from_yfinance(
                symbol, args.start, args.end, interval="5m"
            )

        if not candles:
            print(f"No data loaded for {symbol}")
            return

        candles_1h = (
            loader.resample(candles, "1h") if len(candles) > 12 else None
        )

        strat_cfg = config["strategies"]
        signals_list = []
        if strat_cfg["first_candle"]["enabled"]:
            signals_list.append(FirstCandleSignal(strat_cfg["first_candle"]))
        if strat_cfg["ema_cross"]["enabled"]:
            signals_list.append(EMACrossSignal(strat_cfg["ema_cross"]))
        if strat_cfg["vwap_reversion"]["enabled"]:
            signals_list.append(
                VWAPReversionSignal(strat_cfg["vwap_reversion"])
            )
        if strat_cfg["rsi_momentum"]["enabled"]:
            signals_list.append(RSIMomentumSignal(strat_cfg["rsi_momentum"]))
        if strat_cfg["bollinger_fade"]["enabled"]:
            signals_list.append(
                BollingerFadeSignal(strat_cfg["bollinger_fade"])
            )

        engine = BacktestEngine(config, initial_capital=args.capital)
        result = engine.run(candles, signals_list, candles_1h=candles_1h)
        print(result.summary())


def run_analyze(args) -> None:
    """Run the nightly analysis pipeline immediately."""
    trader = AutoTrader()
    trader.run_nightly_analysis()


def run_optimize(args) -> None:
    """Run vectorbt parameter optimizer for a symbol."""
    config = load_config()

    from backtesting.vectorbt_runner import VectorBTRunner
    from backtesting.optimizer import StrategyOptimizer

    runner = VectorBTRunner(initial_capital=args.capital)
    data = runner.load_data([args.symbol], args.start, args.end)

    if args.symbol not in data:
        print(f"No data loaded for {args.symbol}")
        return

    df = data[args.symbol]
    optimizer = StrategyOptimizer(
        initial_capital=args.capital, fees=0.001
    )
    results = optimizer.optimize_all(df)

    for strategy_name, result_df in results.items():
        print(f"\n{'=' * 60}")
        print(f"  {strategy_name.upper()} -- Top 5 parameter combinations")
        print(f"{'=' * 60}")
        if result_df.empty:
            print("  No valid results.")
        else:
            print(result_df.head(5).to_string(index=False))

    optimizer.save_results(results)
    print(f"\nFull results saved to data/backtest_results.csv")


def run_bias(args) -> None:
    """Run the swing advisor immediately for all watchlist symbols."""
    trader = AutoTrader()
    trader.run_swing_advisor()


def run_dashboard_only(args) -> None:
    """Run the dashboard server without any trading logic."""
    config = load_config()
    db_path = os.getenv("DB_PATH", "data/trades.db")
    db = TradeDatabase(db_path)
    analyzer = PerformanceAnalyzer(db)

    dashboard_init(db=db, analyzer=analyzer, config=config)

    dashboard_cfg = config.get("dashboard", {})
    host = dashboard_cfg.get("host", "0.0.0.0")
    port = dashboard_cfg.get("port", 8000)

    logger.info("dashboard_only_mode | host={} port={}", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> None:
    """Start the AutoTrader system with CLI interface."""
    parser = argparse.ArgumentParser(
        description="AutoTrader v2 -- Automated Trading System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- trade (live / paper) ----------------------------------------------
    subparsers.add_parser(
        "trade", help="Run live or paper trading with dashboard"
    )

    # -- backtest ----------------------------------------------------------
    bt_parser = subparsers.add_parser(
        "backtest", help="Run backtest on historical data"
    )
    bt_parser.add_argument(
        "symbol", help="Symbol to backtest (e.g. AAPL, BTC/USDT)"
    )
    bt_parser.add_argument(
        "--start", default="2024-06-01", help="Start date (YYYY-MM-DD)"
    )
    bt_parser.add_argument(
        "--end", default="2025-01-01", help="End date (YYYY-MM-DD)"
    )
    bt_parser.add_argument(
        "--capital", type=float, default=100_000.0, help="Starting capital"
    )
    bt_parser.add_argument(
        "--csv", default=None, help="Path to CSV file instead of yfinance"
    )

    # -- analyze -----------------------------------------------------------
    subparsers.add_parser("analyze", help="Run nightly analysis now")

    # -- optimize ----------------------------------------------------------
    opt_parser = subparsers.add_parser(
        "optimize", help="Run parameter optimizer"
    )
    opt_parser.add_argument(
        "symbol", help="Symbol to optimize (e.g. AAPL, BTC/USDT)"
    )
    opt_parser.add_argument(
        "--start", default="2024-06-01", help="Start date (YYYY-MM-DD)"
    )
    opt_parser.add_argument(
        "--end", default="2025-01-01", help="End date (YYYY-MM-DD)"
    )
    opt_parser.add_argument(
        "--capital", type=float, default=100_000.0, help="Starting capital"
    )

    # -- bias (swing advisor) ----------------------------------------------
    subparsers.add_parser("bias", help="Run swing advisor now")

    # -- dashboard (UI only) -----------------------------------------------
    subparsers.add_parser(
        "dashboard", help="Run dashboard only (no trading)"
    )

    args = parser.parse_args()

    if args.command == "trade" or args.command is None:
        run_live(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "analyze":
        run_analyze(args)
    elif args.command == "optimize":
        run_optimize(args)
    elif args.command == "bias":
        run_bias(args)
    elif args.command == "dashboard":
        run_dashboard_only(args)


if __name__ == "__main__":
    main()
