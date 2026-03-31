"""AutoTrader v2 -- main entry point and scheduler.

Orchestrates all trading subsystems: data feeds, signals, whale monitoring,
risk management, order execution, journaling, alerts, nightly analysis,
swing bias advisor, backtesting, optimisation, and the live dashboard.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import signal as signal_mod
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

# -- Logging setup (loguru) -------------------------------------------------
IS_PYTEST_PROCESS = "pytest" in sys.modules
LOG_DIR = Path(os.getenv("LOG_PATH", "logs"))
if IS_PYTEST_PROCESS:
    LOG_DIR = LOG_DIR / "pytest"
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
from signals.ema_pullback import EMAPullbackSignal
from signals.vwap_reversion import VWAPReversionSignal
from signals.vpoc_bounce import VPOCBounceSignal
from signals.macd_divergence import MACDDivergenceSignal
from signals.rsi_momentum import RSIMomentumSignal
from signals.bollinger_fade import BollingerFadeSignal
from signals.base_signal import SignalResult, SignalDirection
from signals.indicators import Indicators
from signals.regime_detector import RegimeDetector
from signals.polymarket_sentiment import PolymarketSentiment
from signals.session_bias import SessionBias
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
OVERSEER_TRIGGER_PATH = LOG_DIR / "overseer_trigger.json"
NIGHTLY_ANALYSIS_TRIGGER_PATH = LOG_DIR / "nightly_analysis_trigger.json"
STARVATION_OVERSEER_MODEL = "claude-sonnet-4-6"
DRAWDOWN_HALT_PCT = float(os.getenv("DRAWDOWN_HALT_PCT", "0.05"))
DRAWDOWN_RESUME_PCT = float(os.getenv("DRAWDOWN_RESUME_PCT", "0.03"))
STRATEGY_COOLDOWN_LOSSES = int(os.getenv("STRATEGY_COOLDOWN_LOSSES", "3"))
STRATEGY_COOLDOWN_MINUTES = int(os.getenv("STRATEGY_COOLDOWN_MINUTES", "60"))
WINRATE_LOOKBACK = int(os.getenv("WINRATE_LOOKBACK", "10"))
WINRATE_SIZE_MIN = float(os.getenv("WINRATE_SIZE_MIN", "0.5"))
WINRATE_SIZE_MAX = float(os.getenv("WINRATE_SIZE_MAX", "1.5"))
STRATEGY_DISABLE_AFTER_COOLDOWNS = int(os.getenv("STRATEGY_DISABLE_AFTER_COOLDOWNS", "3"))
SIGNAL_RETRY_COOLDOWN_SECONDS = int(
    os.getenv("SIGNAL_RETRY_COOLDOWN_SECONDS", "900")
)
LOSS_COOLDOWN_MINUTES = int(os.getenv("LOSS_COOLDOWN_MINUTES", "180"))
LOSS_COOLDOWN_STREAK = int(os.getenv("LOSS_COOLDOWN_STREAK", "2"))
SIGNAL_RETRY_PRICE_TOLERANCE_PCT = float(
    os.getenv("SIGNAL_RETRY_PRICE_TOLERANCE_PCT", "0.005")
)
RECENTLY_EXITED_COOLDOWN_SECONDS = int(
    os.getenv("RECENTLY_EXITED_COOLDOWN_SECONDS", "1800")
)
_US_EASTERN = ZoneInfo("US/Eastern")


_config_lock = threading.Lock()


def load_config() -> dict:
    """Load and return the config.json file."""
    with _config_lock, open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _lookup_nested_value(data: dict, path: list[str]) -> object:
    """Safely resolve a nested config value."""
    node: object = data
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _format_message_value(value: object, max_len: int = 48) -> str:
    """Render compact values for Telegram summaries."""
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
    """Build a compact Telegram summary for nightly analysis changes."""
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
        if len(approved) > 5:
            lines.append(f"- ... and {len(approved) - 5} more")
    elif min_trades is None:
        lines.append("")
        lines.append("No config changes proposed.")

    if rejected:
        lines.append("")
        lines.append("Rejected:")
        for item in rejected[:3]:
            param = item.get("param", "?")
            value = _format_message_value(item.get("value"))
            reason = str(item.get("reason", "unknown"))[:90]
            lines.append(f"- {param} -> {value} ({reason})")
        if len(rejected) > 3:
            lines.append(f"- ... and {len(rejected) - 3} more")

    return "\n".join(lines)


def _truncate_quantity(quantity: float, decimals: int = 6) -> float:
    """Truncate a floating quantity without rounding up."""
    scale = 10 ** decimals
    return int(quantity * scale) / scale


def _resolve_claude_auth_mode() -> str:
    """Return the configured Claude auth mode."""
    mode = os.getenv("VARPTRADER_CLAUDE_AUTH_MODE", "login").strip().lower()
    return mode if mode in {"login", "api", "auto"} else "login"


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
        self.signals = self._build_signals(self.config)

        # -- Whale monitoring (new config keys) ------------------------------
        whale_cfg = self.config["whale"]
        self.block_detector = BlockTradeDetector(
            api_key=os.getenv("POLYGON_API_KEY", ""),
            stock_block_shares=whale_cfg.get("stock_block_shares", 50000),
            stock_block_usd=whale_cfg.get("stock_block_usd", 2_000_000),
            flag_ttl_minutes=whale_cfg.get("flag_ttl_minutes", 15),
            enabled=whale_cfg.get("stock_block_trades_enabled", False),
        )
        self.onchain_detector = OnChainWhaleDetector(
            api_key=os.getenv("WHALE_ALERT_API_KEY", ""),
            crypto_transfer_usd=whale_cfg.get("crypto_transfer_usd", 1_000_000),
            flag_ttl_minutes=whale_cfg.get("flag_ttl_minutes", 15),
            min_sell_events=whale_cfg.get("sell_pressure_confirmations", 3),
            max_block_minutes=whale_cfg.get("max_sell_pressure_block_minutes", 30),
            monitored_symbols=self._watchlist_crypto_base_symbols(self.config),
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
            paper_executor=self.paper_executor,
        )

        # -- Regime / sentiment / session bias ------------------------------
        self.regime_detector = RegimeDetector(
            adx_threshold=self.config.get("regime", {}).get("adx_threshold", 25.0),
        )
        self.polymarket = PolymarketSentiment(
            block_short_threshold=self.config.get("polymarket", {}).get(
                "block_short_threshold", 0.65
            ),
        )
        self.session_bias = SessionBias()
        self._session_bias_by_symbol: dict[str, SessionBias] = {}
        self._last_regime_by_symbol: dict[str, str] = {}

        # -- Trade lifecycle -------------------------------------------------
        self.trade_manager = TradeManager()

        # -- State -----------------------------------------------------------
        self._daily_trade_count = 0
        self._in_flight_symbols: set[str] = set()  # Track symbols being processed in current scan cycle
        self._peak_equity: float = 0.0
        self._drawdown_halted: bool = False
        self._strategy_cooldown_until: dict[str, float] = {}  # strategy -> timestamp
        self._strategy_cooldown_count: dict[str, int] = {}  # strategy -> cooldown trigger count
        self._running = True
        self._start_time = time.time()
        self._last_signal_time = time.time()
        self._starvation_alerted = False
        self._starvation_overseer_triggered = False
        self._overseer_run_in_progress = False
        self._recent_signal_attempts: dict[
            tuple[str, str, str],
            dict[str, float],
        ] = {}

    # ====================================================================
    # Config hot-reload
    # ====================================================================
    @staticmethod
    def _watchlist_crypto_base_symbols(config: dict) -> set[str]:
        """Return the monitored crypto base assets from the watchlist."""
        crypto_symbols = config.get("watchlist", {}).get("crypto", [])
        return {
            str(symbol).split("/")[0].upper()
            for symbol in crypto_symbols
            if str(symbol).strip()
        }

    @staticmethod
    def _build_signals(config: dict) -> list:
        """Instantiate all strategy signal objects from config."""
        strat_cfg = config["strategies"]
        return [
            FirstCandleSignal(strat_cfg["first_candle"]),
            EMACrossSignal(strat_cfg["ema_cross"]),
            EMAPullbackSignal(strat_cfg["ema_pullback"]),
            VWAPReversionSignal(strat_cfg["vwap_reversion"]),
            RSIMomentumSignal(strat_cfg["rsi_momentum"]),
            BollingerFadeSignal(strat_cfg["bollinger_fade"]),
            VPOCBounceSignal(strat_cfg["vpoc_bounce"]),
            MACDDivergenceSignal(strat_cfg["macd_divergence"]),
        ]

    def _refresh_runtime_components(self) -> None:
        """Apply the current config to in-memory runtime components."""
        self.signals = self._build_signals(self.config)

        risk_cfg = self.config["risk"]
        self.position_sizer = PositionSizer(risk_cfg)
        self.reward_gate = RewardRatioGate(
            min_ratio=risk_cfg.get("min_reward_ratio", 2.0)
        )
        self.order_validator = OrderValidator(risk_cfg)

        previous_halt = getattr(self.kill_switch, "halted", False)
        self.kill_switch = KillSwitch(
            {"daily_loss_limit_pct": risk_cfg.get("daily_loss_limit_pct", 0.03)}
        )
        self.kill_switch._halted = previous_halt

        self.regime_detector = RegimeDetector(
            adx_threshold=self.config.get("regime", {}).get("adx_threshold", 25.0),
        )
        self.polymarket = PolymarketSentiment(
            block_short_threshold=self.config.get("polymarket", {}).get(
                "block_short_threshold", 0.65
            ),
        )

        whale_cfg = self.config.get("whale", {})
        self.block_detector = BlockTradeDetector(
            api_key=os.getenv("POLYGON_API_KEY", ""),
            stock_block_shares=whale_cfg.get("stock_block_shares", 50000),
            stock_block_usd=whale_cfg.get("stock_block_usd", 2_000_000),
            flag_ttl_minutes=whale_cfg.get("flag_ttl_minutes", 15),
            enabled=whale_cfg.get("stock_block_trades_enabled", False),
        )
        self.onchain_detector = OnChainWhaleDetector(
            api_key=os.getenv("WHALE_ALERT_API_KEY", ""),
            crypto_transfer_usd=whale_cfg.get("crypto_transfer_usd", 1_000_000),
            flag_ttl_minutes=whale_cfg.get("flag_ttl_minutes", 15),
            min_sell_events=whale_cfg.get("sell_pressure_confirmations", 3),
            max_block_minutes=whale_cfg.get("max_sell_pressure_block_minutes", 30),
            monitored_symbols=self._watchlist_crypto_base_symbols(self.config),
        )

        dashboard_init(
            db=self.db,
            analyzer=self.analyzer,
            config=self.config,
            kill_switch=self.kill_switch,
            paper_executor=self.paper_executor,
            stock_feed=self.stock_feed,
            crypto_feed=self.crypto_feed,
            order_validator=self.order_validator,
        )

    def reload_config(self) -> None:
        """Reload config.json from disk and re-apply volatile settings."""
        try:
            new_config = load_config()
            if new_config == self.config:
                logger.debug("config_reload_skipped | reason=no_change")
                return
            self.config = new_config
            self._refresh_runtime_components()
            logger.info("config_reloaded | runtime_components_refreshed=true")
        except Exception:
            logger.exception("config_reload_error")

    @staticmethod
    def _is_us_market_open() -> bool:
        """Return True only during regular US equity trading hours.

        Regular session: Monday–Friday 09:30–16:00 Eastern.
        """
        now_et = datetime.now(_US_EASTERN)
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et < market_close

    def _recently_exited_trade(self, symbol: str, strategy: str) -> bool:
        """Return True if a trade for symbol+strategy closed within the cooldown."""
        recent = self.db.get_recent_closed_trades(
            symbol=symbol, strategy=strategy, limit=1,
            paper_trade_only=self.paper_trade,
        )
        if not recent:
            return False
        last = recent[0]
        exit_ts = last.get("exit_timestamp")
        if not exit_ts:
            return False
        try:
            exit_dt = datetime.fromisoformat(exit_ts)
            elapsed = (datetime.now(timezone.utc) - exit_dt).total_seconds()
            return elapsed < RECENTLY_EXITED_COOLDOWN_SECONDS
        except (ValueError, TypeError):
            return False

    def _record_signal_activity(self) -> None:
        """Mark that at least one strategy emitted a signal this cycle."""
        self._last_signal_time = time.time()
        self._starvation_alerted = False
        self._starvation_overseer_triggered = False

    def _prune_recent_signal_attempts(self) -> None:
        """Drop stale duplicate-suppression keys."""
        now = time.time()
        stale = [
            key
            for key, payload in self._recent_signal_attempts.items()
            if now - payload.get("timestamp", 0.0) > SIGNAL_RETRY_COOLDOWN_SECONDS
        ]
        for key in stale:
            self._recent_signal_attempts.pop(key, None)

    def _signal_attempt_key(
        self,
        symbol: str,
        result: SignalResult,
    ) -> tuple[str, str, str]:
        """Build a stable in-memory key for one signal idea."""
        direction = result.direction.value if result.direction else "none"
        return symbol.upper(), result.strategy_name, direction

    def _is_duplicate_signal_attempt(
        self,
        symbol: str,
        result: SignalResult,
    ) -> bool:
        """Return whether a near-identical signal was attempted recently."""
        self._prune_recent_signal_attempts()
        key = self._signal_attempt_key(symbol, result)
        attempt = self._recent_signal_attempts.get(key)
        if not attempt:
            return False

        elapsed = time.time() - float(attempt.get("timestamp", 0.0))
        if elapsed > SIGNAL_RETRY_COOLDOWN_SECONDS:
            return False

        last_price = float(attempt.get("entry_price", 0.0) or 0.0)
        if last_price <= 0:
            return False

        price_delta_pct = abs(result.entry_price - last_price) / last_price
        if price_delta_pct > SIGNAL_RETRY_PRICE_TOLERANCE_PCT:
            return False

        logger.info(
            "signal_duplicate_suppressed | symbol={} strategy={} direction={} elapsed_sec={:.0f} price_delta_pct={:.4f}",
            symbol,
            result.strategy_name,
            result.direction.value if result.direction else "none",
            elapsed,
            price_delta_pct,
        )
        return True

    def _remember_signal_attempt(
        self,
        symbol: str,
        result: SignalResult,
    ) -> None:
        """Remember a signal attempt to suppress identical retries."""
        key = self._signal_attempt_key(symbol, result)
        self._recent_signal_attempts[key] = {
            "timestamp": time.time(),
            "entry_price": float(result.entry_price),
        }

    def _loss_cooldown_remaining_seconds(
        self,
        symbol: str,
        strategy_name: str,
    ) -> float:
        """Return remaining cooldown seconds after repeated losses."""
        recent = self.db.get_recent_closed_trades(
            symbol=symbol,
            strategy=strategy_name,
            limit=LOSS_COOLDOWN_STREAK,
            paper_trade_only=self.paper_trade,
        )
        if len(recent) < LOSS_COOLDOWN_STREAK:
            return 0.0
        if any(str(trade.get("outcome")) != "loss" for trade in recent):
            return 0.0

        last_exit = recent[0].get("exit_timestamp")
        if not last_exit:
            return 0.0

        try:
            last_dt = datetime.fromisoformat(str(last_exit))
        except ValueError:
            return 0.0

        elapsed = datetime.now(timezone.utc) - last_dt.astimezone(timezone.utc)
        remaining = LOSS_COOLDOWN_MINUTES * 60 - elapsed.total_seconds()
        return max(0.0, remaining)

    def _session_bias_enabled(self) -> bool:
        """Return whether session-bias gating is enabled in config."""
        return bool(self.config.get("session_bias", {}).get("enabled", True))

    def _session_bias_state(self, symbol: str) -> SessionBias:
        """Return the bias state object for one symbol."""
        symbol_key = symbol.upper()
        if not hasattr(self, "_session_bias_by_symbol"):
            self._session_bias_by_symbol = {}
        return self._session_bias_by_symbol.setdefault(symbol_key, SessionBias())

    def _update_session_bias(self, symbol: str, regime: str, candles_1h: list) -> SessionBias:
        """Update and return the session-bias state for one symbol."""
        symbol_key = symbol.upper()
        bias_state = self._session_bias_state(symbol_key)
        if not hasattr(self, "_last_regime_by_symbol"):
            self._last_regime_by_symbol = {}

        previous_regime = self._last_regime_by_symbol.get(symbol_key)
        if regime != previous_regime:
            self._last_regime_by_symbol[symbol_key] = regime
            bias_state.force_reevaluate()
            logger.info(
                "regime_changed | symbol={} old={} new={}",
                symbol,
                previous_regime or "?",
                regime,
            )
        bias_state.evaluate(candles_1h)
        return bias_state

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
        if not self._is_us_market_open():
            logger.debug("scan_stocks_skipped | reason=market_closed")
            return
        logger.info("scanning_stocks")

        # Clear in-flight symbols at start of scan cycle
        self._in_flight_symbols.clear()

        if self.paper_trade and self.paper_portfolio:
            account_value = self.paper_portfolio.equity
        else:
            account = self.stock_executor.get_account()
            account_value = float(account.get("equity", 0)) if account else 0.0

        if account_value <= 0:
            logger.warning("no_account_equity")
            return

        if not self._check_drawdown(account_value):
            logger.info("scan_stocks_skipped | reason=drawdown_breaker")
            return

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl = self.db.get_daily_pnl(today_str)
        if not self.kill_switch.is_trading_allowed(daily_pnl, account_value):
            logger.warning("kill_switch_triggered | daily_pnl={}", daily_pnl)
            self.telegram.send_kill_switch_alert()
            return

        if self.paper_trade and self.paper_executor:
            positions = self.paper_executor.get_positions()
        else:
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

        # -- Regime detection & session bias (use 1h candles) ----------------
        regime = self.regime_detector.detect(candles_1h or candles_5m)
        bias_state = self._session_bias_state(symbol)
        if candles_1h and len(candles_1h) >= 50:
            if self._session_bias_enabled():
                bias_state = self._update_session_bias(symbol, regime, candles_1h)

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
                self._record_signal_activity()

                # -- Fast VWAP pre-check ----------------------------------------
                # Block longs below 1h VWAP, shorts above 1h VWAP
                if (
                    candles_1h
                    and len(candles_1h) >= 10
                    and result.direction
                    and self._uses_directional_vwap_filter(sig.name)
                ):
                    vwap_series = Indicators.vwap(candles_1h)
                    if vwap_series:
                        vwap_1h = vwap_series[-1]
                        if result.direction == SignalDirection.LONG and current_price < vwap_1h:
                            logger.info(
                                "vwap_blocked_long | symbol={} strategy={} price={:.4f} vwap={:.4f}",
                                symbol, sig.name, current_price, vwap_1h,
                            )
                            continue
                        if result.direction == SignalDirection.SHORT and current_price > vwap_1h:
                            logger.info(
                                "vwap_blocked_short | symbol={} strategy={} price={:.4f} vwap={:.4f}",
                                symbol, sig.name, current_price, vwap_1h,
                            )
                            continue

                # -- Trend filter gate --------------------------------------
                if not self._regime_allows(sig.name, result.direction, regime):
                    logger.info(
                        "regime_blocked | symbol={} strategy={} direction={} regime={}",
                        symbol, sig.name, result.direction.value, regime,
                    )
                    continue

                # -- Session bias gate --------------------------------------
                if (
                    self._session_bias_enabled()
                    and result.direction
                    and bias_state.should_block(result.direction.value)
                ):
                    logger.info(
                        "session_bias_blocked | symbol={} strategy={} direction={} bias={}",
                        symbol, sig.name, result.direction.value,
                        bias_state.bias,
                    )
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

        # Clear in-flight symbols at start of scan cycle
        self._in_flight_symbols.clear()

        if self.paper_trade and self.paper_portfolio:
            account_value = self.paper_portfolio.equity
        else:
            balance = self.crypto_executor.get_balance()
            if not balance:
                logger.warning("no_crypto_balance")
                return

            balances = balance.get("free", {})
            usdc_free = float(balances.get("USDC", 0))
            usdt_free = float(balances.get("USDT", 0))
            stable_balance = usdc_free + usdt_free

            if stable_balance <= 1.0:
                logger.warning("no_crypto_balance | usdc={} usdt={}", usdc_free, usdt_free)
                return

            if usdc_free > 0:
                logger.info("crypto_balance_found | currency=USDC amount={}", round(usdc_free, 2))
            if usdt_free > 0:
                logger.info("crypto_balance_found | currency=USDT amount={}", round(usdt_free, 2))

            total = balance.get("total", {})
            account_value = float(total.get("USDT", 0)) + float(total.get("USDC", 0))
            if account_value <= 0:
                account_value = stable_balance

        if account_value <= 0:
            logger.warning("no_crypto_account_value")
            return

        if not self._check_drawdown(account_value):
            logger.info("scan_crypto_skipped | reason=drawdown_breaker")
            return

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl = self.db.get_daily_pnl(today_str)
        if not self.kill_switch.is_trading_allowed(daily_pnl, account_value):
            logger.warning("kill_switch_triggered_crypto | daily_pnl={}", daily_pnl)
            self.telegram.send_kill_switch_alert()
            return

        if self.paper_trade and self.paper_executor:
            crypto_positions = self.paper_executor.get_positions()
        else:
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
        # Prefer WebSocket candle buffer (no REST call), fall back to REST
        ws = getattr(self.crypto_feed, "_ws", None)
        if ws is not None and ws.is_ready(symbol, min_candles=55):
            candles_5m = ws.get_recent_candles(symbol, 200)
        else:
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

        # -- Regime detection & session bias ---------------------------------
        regime = self.regime_detector.detect(candles_1h or candles_5m)
        bias_state = self._session_bias_state(symbol)
        if candles_1h and len(candles_1h) >= 50:
            if self._session_bias_enabled():
                bias_state = self._update_session_bias(symbol, regime, candles_1h)

        whale_flag = 0
        if self.onchain_detector.has_sell_pressure(base_symbol):
            logger.info(
                "onchain_sell_pressure | symbol={} -- skipping after confirmed sell-pressure cluster",
                symbol,
            )
            return

        for sig in self.signals:
            if not sig.is_enabled():
                continue
            if not self._strategy_allowed_for_market(sig.name, "crypto"):
                continue
            try:
                result = self._run_signal(
                    sig, symbol, candles_5m, current_price, "crypto",
                    None, 0.0, candles_1h=candles_1h,
                )
                if not result.triggered:
                    continue
                self._record_signal_activity()

                # -- Fast VWAP pre-check ----------------------------------------
                # Block longs below 1h VWAP, shorts above 1h VWAP
                if (
                    candles_1h
                    and len(candles_1h) >= 10
                    and result.direction
                    and self._uses_directional_vwap_filter(sig.name)
                ):
                    vwap_series = Indicators.vwap(candles_1h)
                    if vwap_series:
                        vwap_1h = vwap_series[-1]
                        if result.direction == SignalDirection.LONG and current_price < vwap_1h:
                            logger.info(
                                "vwap_blocked_long | symbol={} strategy={} price={:.4f} vwap={:.4f}",
                                symbol, sig.name, current_price, vwap_1h,
                            )
                            continue
                        if result.direction == SignalDirection.SHORT and current_price > vwap_1h:
                            logger.info(
                                "vwap_blocked_short | symbol={} strategy={} price={:.4f} vwap={:.4f}",
                                symbol, sig.name, current_price, vwap_1h,
                            )
                            continue

                # -- Trend filter gate --------------------------------------
                if not self._regime_allows(sig.name, result.direction, regime):
                    logger.info(
                        "regime_blocked | symbol={} strategy={} direction={} regime={}",
                        symbol, sig.name, result.direction.value, regime,
                    )
                    continue

                # -- Session bias gate --------------------------------------
                if (
                    self._session_bias_enabled()
                    and result.direction
                    and bias_state.should_block(result.direction.value)
                ):
                    logger.info(
                        "session_bias_blocked | symbol={} strategy={} direction={} bias={}",
                        symbol, sig.name, result.direction.value,
                        bias_state.bias,
                    )
                    continue

                # -- Polymarket sentiment (BTC/ETH only) --------------------
                if (
                    result.direction == SignalDirection.SHORT
                    and self.polymarket.should_block_short(base_symbol)
                ):
                    logger.info(
                        "polymarket_blocked_short | symbol={} strategy={} asset={}",
                        symbol, sig.name, base_symbol,
                    )
                    continue

                self._process_signal(
                    result, symbol, "crypto", account_value,
                    current_positions, 0,
                )
            except Exception:
                logger.exception(
                    "signal_eval_error_crypto | symbol={} strategy={}",
                    symbol, sig.name,
                )

    # ====================================================================
    # Signal evaluation helpers
    # ====================================================================
    # Strategies that are excluded from crypto markets
    _STOCK_ONLY_STRATEGIES: set[str] = {"bollinger_fade", "first_candle"}

    # Correlation groups: symbols that move together. Max 2 same-direction
    # positions per group to limit overlapping exposure.
    _CORRELATION_GROUPS: list[set[str]] = [
        {"SPY", "QQQ", "AAPL", "NVDA"},  # US large-cap tech / index
        {"BTC/USDT", "BTC/USDC"},        # BTC pairs
        {"ETH/USDT", "ETH/USDC"},        # ETH pairs
        {"SOL/USDT", "SOL/USDC"},        # SOL pairs
    ]
    _MAX_CORRELATED_POSITIONS: int = 2

    @staticmethod
    def _strategy_allowed_for_market(strategy_name: str, market: str) -> bool:
        """Return False if *strategy_name* is excluded from *market*."""
        if market == "crypto" and strategy_name in AutoTrader._STOCK_ONLY_STRATEGIES:
            return False
        return True

    def _correlation_allows(self, symbol: str, direction: str) -> bool:
        """Check if opening this position would exceed correlation limits."""
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return True
        for group in self._CORRELATION_GROUPS:
            if symbol not in group:
                continue
            same_dir_count = sum(
                1 for t in open_trades
                if t["symbol"] in group and t["direction"] == direction
            )
            if same_dir_count >= self._MAX_CORRELATED_POSITIONS:
                logger.info(
                    "correlation_blocked | symbol={} direction={} group_count={}",
                    symbol, direction, same_dir_count,
                )
                return False
        return True

    def _check_drawdown(self, current_equity: float) -> bool:
        """Check equity drawdown from peak. Returns True if trading is allowed."""
        if current_equity <= 0:
            return True

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity <= 0:
            return True

        drawdown = (self._peak_equity - current_equity) / self._peak_equity

        if self._drawdown_halted:
            if drawdown <= DRAWDOWN_RESUME_PCT:
                self._drawdown_halted = False
                logger.info(
                    "drawdown_breaker_resumed | equity={} peak={} drawdown_pct={:.2%}",
                    round(current_equity, 2), round(self._peak_equity, 2), drawdown,
                )
                self.telegram.send_message(
                    f"DRAWDOWN BREAKER RESUMED\n\nEquity: ${current_equity:,.2f}\nPeak: ${self._peak_equity:,.2f}\nDrawdown: {drawdown:.1%}",
                    parse_mode="",
                )
                return True
            return False

        if drawdown >= DRAWDOWN_HALT_PCT:
            self._drawdown_halted = True
            logger.warning(
                "drawdown_breaker_triggered | equity={} peak={} drawdown_pct={:.2%}",
                round(current_equity, 2), round(self._peak_equity, 2), drawdown,
            )
            self.telegram.send_message(
                f"DRAWDOWN BREAKER TRIGGERED\n\nEquity: ${current_equity:,.2f}\nPeak: ${self._peak_equity:,.2f}\nDrawdown: {drawdown:.1%}\n\nTrading paused until drawdown recovers to {DRAWDOWN_RESUME_PCT:.0%}",
                parse_mode="",
            )
            return False

        return True

    def _strategy_cooldown_allows(self, strategy: str) -> bool:
        """Return False if this strategy is on cooldown after consecutive losses."""
        until = self._strategy_cooldown_until.get(strategy, 0.0)
        if time.time() < until:
            remaining = int(until - time.time())
            logger.info(
                "strategy_cooldown_active | strategy={} remaining_sec={}",
                strategy, remaining,
            )
            return False

        recent = self.db.get_recent_closed_by_strategy(
            strategy, limit=STRATEGY_COOLDOWN_LOSSES,
        )
        if len(recent) < STRATEGY_COOLDOWN_LOSSES:
            return True

        if all(t["outcome"] == "loss" for t in recent):
            self._strategy_cooldown_count[strategy] = (
                self._strategy_cooldown_count.get(strategy, 0) + 1
            )
            count = self._strategy_cooldown_count[strategy]

            # Auto-disable after too many cooldown triggers
            if count >= STRATEGY_DISABLE_AFTER_COOLDOWNS:
                self._auto_disable_strategy(strategy, count)
                return False

            cooldown_until = time.time() + STRATEGY_COOLDOWN_MINUTES * 60
            self._strategy_cooldown_until[strategy] = cooldown_until
            logger.warning(
                "strategy_cooldown_triggered | strategy={} consecutive_losses={} cooldown_min={} cooldown_count={}",
                strategy, STRATEGY_COOLDOWN_LOSSES, STRATEGY_COOLDOWN_MINUTES, count,
            )
            self.telegram.send_message(
                f"STRATEGY COOLDOWN ({count}/{STRATEGY_DISABLE_AFTER_COOLDOWNS})\n\n{strategy}: {STRATEGY_COOLDOWN_LOSSES} consecutive losses\nPaused for {STRATEGY_COOLDOWN_MINUTES} minutes\n\n⚠️ Will auto-disable after {STRATEGY_DISABLE_AFTER_COOLDOWNS} cooldowns",
                parse_mode="",
            )
            return False

        # A win resets the cooldown count
        if self._strategy_cooldown_count.get(strategy, 0) > 0:
            if any(t["outcome"] == "win" for t in recent):
                self._strategy_cooldown_count[strategy] = 0
        return True

    def _auto_disable_strategy(self, strategy: str, cooldown_count: int) -> None:
        """Auto-disable a strategy in config.json after repeated cooldowns."""
        try:
            cfg = load_config()
            if strategy in cfg.get("strategies", {}):
                cfg["strategies"][strategy]["enabled"] = False
                try:
                    from main import _config_lock
                except ImportError:
                    _config_lock = threading.Lock()
                with _config_lock, open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                    json.dump(cfg, fh, indent=2)
                    fh.write("\n")

            logger.warning(
                "strategy_auto_disabled | strategy={} cooldown_count={}",
                strategy, cooldown_count,
            )
            self.telegram.send_message(
                f"🚫 STRATEGY AUTO-DISABLED\n\n{strategy} has been disabled after {cooldown_count} consecutive cooldown triggers.\n\nReview performance and manually re-enable in config when ready.",
                parse_mode="",
            )
        except Exception:
            logger.exception("strategy_auto_disable_failed | strategy={}", strategy)

    def _winrate_size_multiplier(self, strategy: str) -> float:
        """Scale position size based on recent win rate for this strategy.

        Returns a multiplier between WINRATE_SIZE_MIN and WINRATE_SIZE_MAX.
        With 0% win rate -> MIN, 50% -> 1.0, 100% -> MAX.
        Falls back to 1.0 when there's no history.
        """
        recent = self.db.get_recent_closed_by_strategy(
            strategy, limit=WINRATE_LOOKBACK,
        )
        if len(recent) < 3:
            return 1.0

        wins = sum(1 for t in recent if t["outcome"] == "win")
        win_rate = wins / len(recent)

        # Linear interpolation: 0% -> MIN, 50% -> 1.0, 100% -> MAX
        if win_rate <= 0.5:
            multiplier = WINRATE_SIZE_MIN + (1.0 - WINRATE_SIZE_MIN) * (win_rate / 0.5)
        else:
            multiplier = 1.0 + (WINRATE_SIZE_MAX - 1.0) * ((win_rate - 0.5) / 0.5)

        logger.info(
            "winrate_size_adjustment | strategy={} win_rate={:.0%} trades={} multiplier={:.2f}",
            strategy, win_rate, len(recent), multiplier,
        )
        return multiplier

    @staticmethod
    def _regime_allows(
        strategy_name: str,
        direction: SignalDirection | None,
        regime: str,
    ) -> bool:
        """Check whether *regime* permits *direction* for *strategy_name*.

        Rules:
        - Longs are ONLY allowed in trending_up regime (not ranging)
        - rsi_momentum  shorts only in trending_down or ranging
        - bollinger_fade fires freely in ranging, but in trends it only
          takes pullbacks in the direction of the trend
        - ema_cross      always allowed (it IS trend-following)
        - first_candle   always allowed
        """
        if direction is None:
            return True

        # Global long gate: only allow longs in confirmed uptrend
        if direction == SignalDirection.LONG and regime != "trending_up":
            return False

        if strategy_name == "rsi_momentum":
            if direction == SignalDirection.SHORT:
                return regime in ("trending_down", "ranging")

        if strategy_name == "bollinger_fade":
            if regime == "ranging":
                return True
            if regime == "trending_up":
                return direction == SignalDirection.LONG
            if regime == "trending_down":
                return direction == SignalDirection.SHORT

        return True

    @staticmethod
    def _uses_directional_vwap_filter(strategy_name: str) -> bool:
        """Return whether a strategy should respect the directional VWAP gate.

        ema_cross is excluded: it is trend-following and the crossover
        itself is a stronger directional signal than intraday VWAP.
        """
        return strategy_name in {"first_candle", "ema_pullback"}

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
            atr_series = Indicators.atr(candles, 20)
            atr_20d = atr_series[-1] if atr_series else first_candle.high - first_candle.low
            return sig.evaluate_with_context(
                symbol=symbol,
                orb_high=first_candle.high,
                orb_low=first_candle.low,
                current_candle=candles[-1],
                avg_volume=avg_volume,
                minutes_since_open=minutes_since,
                atr_20d=atr_20d,
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
            vwap_val = vwap_series[-1] if vwap_series else current_price
            atr_vals = Indicators.atr(candles, period=sig.config.get("atr_period", 14))
            atr_val = atr_vals[-1] if atr_vals else 0.0
            slope_lb = sig.config.get("slope_lookback", 20)
            slope_val = Indicators.vwap_slope(vwap_series, lookback=slope_lb)
            cur_vol = candles[-1].volume if candles else 0.0
            vol_window = candles[-20:] if len(candles) >= 20 else candles
            avg_vol = sum(c.volume for c in vol_window) / len(vol_window) if vol_window else 1.0
            return sig.evaluate_from_vwap(
                symbol=symbol, current_price=current_price,
                vwap=vwap_val, atr=atr_val, vwap_slope=slope_val,
                current_volume=cur_vol, avg_volume=avg_vol, market=market,
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
                if candles_1h and len(candles_1h) >= 50
                else candles
            )
            closes = [c.close for c in ema_candles]

            fast_period = sig.config.get("fast_ema", 20)
            slow_period = sig.config.get("slow_ema", 50)

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

            # Volume confirmation: current bar volume > 20-period avg
            if sig.config.get("volume_confirmation", True):
                volumes = [c.volume for c in ema_candles]
                if len(volumes) >= 20:
                    avg_vol = sum(volumes[-20:]) / 20
                    if volumes[-1] <= avg_vol:
                        return SignalResult(
                            triggered=False, strategy_name=sig.name,
                            reason="Volume below 20-period average",
                        )

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
        elif isinstance(sig, EMAPullbackSignal):
            ema_candles = (
                candles_1h
                if candles_1h and len(candles_1h) >= 50
                else candles
            )
            closes = [c.close for c in ema_candles]

            fast_period = sig.config.get("fast_ema", 20)
            slow_period = sig.config.get("slow_ema", 50)

            fast_ema_series = Indicators.ema(closes, fast_period)
            slow_ema_series = Indicators.ema(closes, slow_period)

            if not fast_ema_series or not slow_ema_series:
                return SignalResult(triggered=False, strategy_name=sig.name)

            ema_fast = fast_ema_series[-1]
            ema_slow = slow_ema_series[-1]

            if sig.config.get("volume_confirmation", True):
                volumes = [c.volume for c in ema_candles]
                if len(volumes) >= 20:
                    avg_vol = sum(volumes[-20:]) / 20
                    if volumes[-1] <= avg_vol:
                        return SignalResult(
                            triggered=False, strategy_name=sig.name,
                            reason="Volume below 20-period average",
                        )

            rsi_series = Indicators.rsi(closes, 14)
            rsi = rsi_series[-1] if rsi_series else 50.0

            return sig.evaluate_pullback(
                symbol=symbol,
                current_price=current_price,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                rsi=rsi,
            )

        elif isinstance(sig, VPOCBounceSignal):
            bounce_n = sig.config.get("bounce_candles", 2)
            return sig.evaluate_from_profile(
                symbol=symbol, current_price=current_price,
                session_candles=candles, recent_candles=candles[-bounce_n:],
                market=market,
            )

        elif isinstance(sig, MACDDivergenceSignal):
            return sig.evaluate_from_macd(
                symbol=symbol, current_price=current_price,
                candles=candles, market=market,
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

        cooldown_remaining = self._loss_cooldown_remaining_seconds(
            symbol=symbol,
            strategy_name=result.strategy_name,
        )
        if cooldown_remaining > 0:
            logger.info(
                "signal_blocked_recent_losses | symbol={} strategy={} cooldown_remaining_min={:.1f}",
                symbol,
                result.strategy_name,
                cooldown_remaining / 60.0,
            )
            return

        if self._is_duplicate_signal_attempt(symbol, result):
            return

        if self._recently_exited_trade(symbol, result.strategy_name):
            logger.info(
                "signal_blocked_recently_exited | symbol={} strategy={} cooldown_sec={}",
                symbol, result.strategy_name, RECENTLY_EXITED_COOLDOWN_SECONDS,
            )
            return

        # In-memory duplicate guard: prevent multiple strategies from
        # opening the same symbol in the same scan cycle
        if symbol in self._in_flight_symbols:
            logger.info(
                "duplicate_blocked_in_memory | symbol={} strategy={}",
                symbol, result.strategy_name,
            )
            return
        # Mark this symbol as in-flight immediately
        self._in_flight_symbols.add(symbol)

        if not self._correlation_allows(symbol, result.direction.value):
            return

        if not self._strategy_cooldown_allows(result.strategy_name):
            return

        if not self.reward_gate.check(
            result.entry_price, result.stop_loss, result.take_profit
        ):
            logger.info(
                "reward_ratio_rejected | symbol={} strategy={}",
                symbol, result.strategy_name,
            )
            return

        self._remember_signal_attempt(symbol, result)

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
        # Determine available cash for position sizing
        if self.paper_trade and self.paper_portfolio:
            available_cash = self.paper_portfolio.cash
        elif market == "stock":
            account = self.stock_executor.get_account()
            available_cash = float(account.get("buying_power", 0)) if account else 0.0
        else:
            available_cash = account_value  # crypto: account_value already from stablecoin balance

        quantity = self.position_sizer.calculate_size(
            account_value, result.entry_price, atr,
            available_cash=available_cash, market=market,
        )

        # Scale quantity by recent win rate for this strategy
        size_mult = self._winrate_size_multiplier(result.strategy_name)
        if size_mult != 1.0:
            raw_qty = quantity
            quantity = quantity * size_mult
            # Re-apply market rounding
            if market == "crypto":
                quantity = float(int(quantity * 1e6) / 1e6)
            else:
                quantity = float(max(1, int(quantity)))
            logger.info(
                "quantity_scaled_by_winrate | strategy={} raw={} scaled={} multiplier={:.2f}",
                result.strategy_name, raw_qty, quantity, size_mult,
            )

        if quantity <= 0:
            logger.warning(
                "order_rejected_insufficient_cash | symbol={} cash={} price={}",
                symbol, round(available_cash, 2), result.entry_price,
            )
            return

        side = "buy" if result.direction == SignalDirection.LONG else "sell"

        open_trades = self.db.get_open_trades()
        open_symbols = {t["symbol"] for t in open_trades} if open_trades else set()
        current_positions = len(open_trades)

        valid, reason = self.order_validator.validate(
            symbol, side, quantity, current_positions,
            self._daily_trade_count, self.kill_switch.halted,
            open_symbols=open_symbols,
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
                stop_loss=result.stop_loss,
                take_profit=result.take_profit,
                whale_flag=whale_flag,
            )
            if not order_result:
                logger.warning(
                    "paper_order_failed | symbol={} strategy={} side={}",
                    symbol, result.strategy_name, side,
                )
                return
            trade_id = order_result.get("trade_id")
            fill_price = float(order_result.get("fill_price", result.entry_price))
            slippage = abs(fill_price - result.entry_price) / result.entry_price if result.entry_price else 0.0
            if slippage > 0.001:  # Log when slippage > 0.1%
                logger.warning(
                    "trade_slippage | symbol={} strategy={} expected={} fill={} slippage_pct={:.4%}",
                    symbol, result.strategy_name, result.entry_price, fill_price, slippage,
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

            fill_price = float(order_result.get("fill_price", result.entry_price)) if order_result else result.entry_price
            slippage = abs(fill_price - result.entry_price) / result.entry_price if result.entry_price else 0.0
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
                paper_trade=0,
                slippage=slippage,
            )
            trade_id = self.db.insert_trade(trade)
            if slippage > 0.001:  # Log when slippage > 0.1%
                logger.warning(
                    "trade_slippage | symbol={} strategy={} expected={} fill={} slippage_pct={:.4%}",
                    symbol, result.strategy_name, result.entry_price, fill_price, slippage,
                )

        self._daily_trade_count += 1
        self._last_signal_time = time.time()

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
            whale_cfg = self.config.get("whale", {})
            if not whale_cfg.get("enabled", True):
                return
            if not whale_cfg.get("stock_block_trades_enabled", False):
                return
            self.block_detector.poll(self.config["watchlist"]["stocks"])
        except Exception:
            logger.exception("whale_stock_poll_error")

    def poll_whale_crypto(self) -> None:
        """Poll for on-chain whale movements."""
        try:
            if not self.config.get("whale", {}).get("enabled", True):
                return
            self.onchain_detector.poll()
        except Exception:
            logger.exception("whale_crypto_poll_error")

    # ====================================================================
    # Nightly analysis
    # ====================================================================
    def run_nightly_analysis(self) -> None:
        """Execute the AI-powered nightly analysis pipeline."""
        if _resolve_claude_auth_mode() == "login" and shutil.which("claude") is None:
            payload = {
                "trigger_reason": "nightly_analysis",
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "model": self.config.get("analysis", {}).get("model", "claude-sonnet-4-6"),
            }
            NIGHTLY_ANALYSIS_TRIGGER_PATH.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info(
                "nightly_analysis_queued_for_host | path={}",
                NIGHTLY_ANALYSIS_TRIGGER_PATH,
            )
            self.telegram.send_message(
                "NIGHTLY ANALYSIS QUEUED\n\nClaude nightly analysis will run on the VPS host using the logged-in subscription.",
                parse_mode="",
            )
            return

        logger.info("nightly_analysis_starting")
        try:
            analysis_cfg = self.config.get("analysis", {})
            lookback = analysis_cfg.get("lookback_days", 30)
            auto_apply = analysis_cfg.get("auto_apply_changes", False)
            config_before = self.config
            report_data = self.analyzer.build_full_report(lookback_days=lookback)

            min_trades = analysis_cfg.get("min_trades_for_suggestion", 15)
            if report_data.get("total_trades", 0) < min_trades:
                logger.info(
                    "insufficient_trades_for_analysis | count={}",
                    report_data.get("total_trades", 0),
                )
                # Still build and send a basic stats report (no AI recommendations)
                report_md = self.report_builder.build_daily_report(
                    report_data, config_changes=[], rejected=[],
                )
                self.telegram.send_daily_report(report_md)
                self.telegram.send_message(
                    _build_nightly_change_summary(
                        config_before,
                        report_data,
                        {},
                        [],
                        auto_apply=auto_apply,
                        min_trades=min_trades,
                    ),
                    parse_mode="",
                )

                from journal.models import AnalysisRun

                run = AnalysisRun(
                    trades_analyzed=report_data.get("total_trades", 0),
                    report_markdown=report_md,
                )
                self.db.insert_analysis_run(run)
                logger.info("nightly_basic_report_sent | trades={}", report_data.get("total_trades", 0))
                return

            recommendations = self.llm_advisor.get_recommendations(report_data)
            approved, rejected = self.config_updater.validate_changes(
                recommendations
            )
            # Config changes are handled by the overseer (runs 30 min later
            # with full context).  The nightly analysis only reports
            # recommendations -- it never writes to config.json.
            if approved:
                logger.info(
                    "nightly_config_recommendations | changes={} (not applied, overseer handles config)",
                    approved,
                )

            report_md = self.report_builder.build_daily_report(
                report_data, config_changes=approved, rejected=rejected
            )
            self.telegram.send_daily_report(report_md)
            self.telegram.send_message(
                _build_nightly_change_summary(
                    config_before,
                    report_data,
                    approved,
                    rejected,
                    auto_apply=False,  # Never auto-apply; overseer handles config
                ),
                parse_mode="",
            )

            from journal.models import AnalysisRun

            run = AnalysisRun(
                trades_analyzed=report_data.get("total_trades", 0),
                report_markdown=report_md,
                config_changes_json=json.dumps(approved) if approved else None,
                approved=0,  # Recommendations only; overseer applies
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
    # Heartbeat
    # ====================================================================
    def send_heartbeat(self) -> None:
        """Send a periodic heartbeat message to Telegram."""
        try:
            uptime_sec = time.time() - self._start_time
            hours = int(uptime_sec // 3600)
            minutes = int((uptime_sec % 3600) // 60)
            if hours > 0:
                uptime_str = f"{hours}h {minutes}m"
            else:
                uptime_str = f"{minutes}m"

            positions = []
            cash = 0.0
            equity = 0.0
            unrealized = 0.0
            if self.paper_trade and self.paper_executor:
                pos_list = self.paper_executor.get_positions()
                portfolio = self.paper_executor._portfolio
                cash = portfolio.cash
                for p in pos_list:
                    sym = p.get("symbol", "?")
                    side = p.get("side", "?")
                    entry = p.get("avg_entry_price", 0)
                    pnl = p.get("unrealized_pl", 0.0)
                    unrealized += pnl
                    pnl_str = f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}"
                    positions.append(f"  {sym} {side} @ {entry:,.2f} ({pnl_str})")
                invested = sum(
                    abs(p.get("qty", 0)) * p.get("avg_entry_price", 0)
                    for p in pos_list
                )
                equity = cash + invested + unrealized

            pos_lines = "\n".join(positions) if positions else "  None"

            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_pnl = self.db.get_daily_pnl(today_str)
            daily_trade_count = self.db.get_daily_trade_count(today_str)

            text = (
                f"\U0001f493 VarpTrader Heartbeat\n"
                f"Equity: ${equity:,.0f} | Cash: ${cash:,.0f}\n"
                f"Trades Today: {daily_trade_count}\n"
                f"Positions: {len(positions)}\n"
                f"{pos_lines}\n"
                f"Daily P&L: ${daily_pnl:+,.0f} | Unrealized: ${unrealized:+,.0f}\n"
                f"Uptime: {uptime_str}"
            )
            self.telegram.send_message(text, parse_mode="")
            logger.info("heartbeat_sent | equity={} positions={} uptime={}", round(equity, 2), len(positions), uptime_str)
        except Exception:
            logger.exception("heartbeat_error")

    def _run_overseer_job(
        self,
        trigger_reason: str,
        *,
        deep: bool = False,
        model: str | None = None,
    ) -> None:
        """Execute the overseer in a background thread."""
        try:
            from overseer.run_overseer import run_overseer

            report = run_overseer(
                deep=deep,
                model=model,
                trigger_reason=trigger_reason,
            )
            logger.info(
                "overseer_trigger_complete | reason={} deep={} chars={}",
                trigger_reason,
                deep,
                len(report),
            )
        except Exception as exc:
            logger.exception(
                "overseer_trigger_error | reason={} deep={} err={}",
                trigger_reason,
                deep,
                exc,
            )
            self.telegram.send_message(
                f"OVERSEER ERROR\n\nTrigger: {trigger_reason}\nError: {exc}",
                parse_mode="",
            )
        finally:
            self._overseer_run_in_progress = False

    def trigger_overseer_async(
        self,
        trigger_reason: str,
        *,
        deep: bool = False,
        model: str | None = None,
    ) -> str:
        """Trigger overseer locally when available, otherwise queue a host trigger."""
        if trigger_reason == "signal_starvation" and not model:
            model = STARVATION_OVERSEER_MODEL

        if self._overseer_run_in_progress:
            logger.info(
                "overseer_trigger_skipped | reason={} status=already_running",
                trigger_reason,
            )
            return "skipped"

        if shutil.which("claude") is None:
            payload = {
                "trigger_reason": trigger_reason,
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "deep": deep,
                "model": model,
            }
            OVERSEER_TRIGGER_PATH.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info(
                "overseer_trigger_queued | reason={} path={}",
                trigger_reason,
                OVERSEER_TRIGGER_PATH,
            )
            return "queued"

        self._overseer_run_in_progress = True
        threading.Thread(
            target=self._run_overseer_job,
            kwargs={
                "trigger_reason": trigger_reason,
                "deep": deep,
                "model": model,
            },
            daemon=True,
        ).start()
        logger.info(
            "overseer_trigger_started | reason={} deep={}",
            trigger_reason,
            deep,
        )
        return "running"

    # ====================================================================
    # Signal starvation detection
    # ====================================================================
    def check_signal_starvation(self) -> None:
        """Alert if no signals have fired in the last 4 hours during market hours."""
        try:
            now = datetime.now(timezone.utc)
            hour_utc = now.hour

            # Only check during active hours: stocks 13:30-20:00 UTC, crypto 24/7
            # Use a simple check: always monitor since crypto is 24/7
            elapsed = time.time() - self._last_signal_time
            starvation_hours = 4

            if elapsed > starvation_hours * 3600:
                hours_since = elapsed / 3600
                if not self._starvation_alerted:
                    logger.warning(
                        "signal_starvation | hours_since_last_signal={:.1f}",
                        hours_since,
                    )
                    self.telegram.send_message(
                        f"SIGNAL STARVATION\n\n"
                        f"No signals have fired in {hours_since:.1f} hours.\n"
                        f"Filters may be too aggressive.\n"
                        f"Check regime detector, session bias, and RSI thresholds.",
                        parse_mode="",
                    )
                    self._starvation_alerted = True
                if not self._starvation_overseer_triggered:
                    status = self.trigger_overseer_async("signal_starvation")
                    if status == "running":
                        self.telegram.send_message(
                            f"OVERSEER TRIGGERED\n\n"
                            f"Reason: signal starvation ({hours_since:.1f}h without signal).",
                            parse_mode="",
                        )
                    elif status == "queued":
                        self.telegram.send_message(
                            f"OVERSEER QUEUED\n\n"
                            f"Reason: signal starvation ({hours_since:.1f}h without signal).\n"
                            f"Host watcher will execute the overseer run.",
                            parse_mode="",
                        )
                    self._starvation_overseer_triggered = True
            else:
                self._starvation_alerted = False
                self._starvation_overseer_triggered = False
        except Exception:
            logger.exception("starvation_check_error")

    # ====================================================================
    # Telegram commands
    # ====================================================================
    @staticmethod
    def _parse_telegram_instruction(text: str) -> tuple[str, str] | None:
        """Parse slash commands and plain-text buy/sell instructions."""
        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("/"):
            parts = stripped.split(maxsplit=1)
            command = parts[0].lower().split("@")[0]
            argument = parts[1].strip() if len(parts) > 1 else ""
            return command, argument

        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            return None
        action = parts[0].lower()
        if action not in {"buy", "sell"}:
            return None
        return f"/{action}", parts[1].strip()

    def _resolve_manual_symbol(self, raw_symbol: str) -> tuple[str, str]:
        """Resolve a Telegram-entered symbol to a known market."""
        symbol = raw_symbol.strip().upper()
        if not symbol:
            raise ValueError("Missing symbol. Use /buy NVDA or /sell NVDA.")

        watchlist = self.config.get("watchlist", {})
        stock_symbols = [str(item).upper() for item in watchlist.get("stocks", [])]
        crypto_symbols = [str(item).upper() for item in watchlist.get("crypto", [])]
        crypto_aliases = {item.replace("/", ""): item for item in crypto_symbols}

        if symbol in crypto_symbols:
            return symbol, "crypto"
        if symbol in crypto_aliases:
            return crypto_aliases[symbol], "crypto"
        if "/" in symbol:
            return symbol, "crypto"
        if symbol in stock_symbols:
            return symbol, "stock"

        match = difflib.get_close_matches(symbol, stock_symbols, n=1, cutoff=0.75)
        if match:
            return match[0], "stock"
        return symbol, "stock"

    def _manual_account_snapshot(self) -> tuple[float, float]:
        """Return current account equity and available cash for manual orders."""
        if self.paper_trade and self.paper_portfolio:
            return self.paper_portfolio.equity, self.paper_portfolio.cash
        raise RuntimeError("Manual Telegram trading is enabled only in PAPER mode.")

    def _get_manual_market_price(self, symbol: str, market: str) -> float:
        """Fetch the latest price for a manual Telegram order."""
        price = (
            self.stock_feed.get_latest_price(symbol)
            if market == "stock"
            else self.crypto_feed.get_latest_price(symbol)
        )
        return float(price or 0.0)

    def _calculate_manual_quantity(
        self,
        symbol: str,
        market: str,
        market_price: float,
        account_value: float,
        available_cash: float,
    ) -> float:
        """Size a manual Telegram order from configured position risk."""
        risk_pct = float(self.config.get("risk", {}).get("position_size_pct", 0.005) or 0.005)
        notional = min(available_cash, max(0.0, account_value * risk_pct))
        if market == "stock":
            quantity = int(notional / market_price) if market_price > 0 else 0
            if quantity <= 0 and available_cash >= market_price > 0:
                quantity = 1
            return float(quantity)

        if market_price <= 0:
            return 0.0
        return _truncate_quantity(notional / market_price, decimals=6)

    def _cmd_buy(self, raw_symbol: str) -> None:
        """Open a manual long position from Telegram."""
        try:
            symbol, market = self._resolve_manual_symbol(raw_symbol)
            account_value, available_cash = self._manual_account_snapshot()
            market_price = self._get_manual_market_price(symbol, market)
            if market_price <= 0:
                self.telegram.send_message(
                    f"Manual buy failed: no live price available for {symbol}.",
                    parse_mode="",
                )
                return

            quantity = self._calculate_manual_quantity(
                symbol, market, market_price, account_value, available_cash,
            )
            if quantity <= 0:
                self.telegram.send_message(
                    f"Manual buy failed: size for {symbol} would be zero at ${market_price:,.2f}.",
                    parse_mode="",
                )
                return

            open_trades = self.db.get_open_trades()
            open_symbols = {trade["symbol"] for trade in open_trades} if open_trades else set()
            current_positions = len(open_symbols)
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_trade_count = max(self._daily_trade_count, self.db.get_daily_trade_count(today_str))

            valid, reason = self.order_validator.validate(
                symbol,
                "buy",
                quantity,
                current_positions,
                daily_trade_count,
                self.kill_switch.halted,
                open_symbols=open_symbols,
            )
            if not valid:
                self.telegram.send_message(
                    f"Manual buy rejected for {symbol}: {reason}",
                    parse_mode="",
                )
                return

            order = self.paper_executor.submit_market_order(
                symbol=symbol,
                side="buy",
                quantity=quantity,
                market_price=market_price,
                market=market,
                strategy="manual_telegram",
                stop_loss=0.0,
                take_profit=0.0,
                whale_flag=0,
            )
            if not order:
                self.telegram.send_message(
                    f"Manual buy failed for {symbol}.",
                    parse_mode="",
                )
                return

            self._daily_trade_count = daily_trade_count + 1
            self.telegram.send_message(
                "MANUAL BUY EXECUTED\n\n"
                f"Symbol: {symbol}\n"
                f"Qty: {float(order.get('qty', quantity)):.6f}".rstrip("0").rstrip(".") + "\n"
                f"Fill: ${float(order.get('fill_price', market_price)):,.2f}\n"
                "Exit: sell it manually with /sell SYMBOL.",
                parse_mode="",
            )
            logger.info("manual_buy_executed | symbol={} market={} qty={}", symbol, market, quantity)
        except Exception as exc:
            logger.exception("manual_buy_error | symbol={} err={}", raw_symbol, exc)
            self.telegram.send_message(
                f"Manual buy failed: {exc}",
                parse_mode="",
            )

    def _cmd_sell(self, raw_symbol: str) -> None:
        """Close an open position manually from Telegram."""
        try:
            symbol, market = self._resolve_manual_symbol(raw_symbol)
            if not (self.paper_trade and self.paper_executor):
                raise RuntimeError("Manual Telegram trading is enabled only in PAPER mode.")

            open_positions = self.paper_executor.get_positions()
            open_symbols = {str(pos.get('symbol', '')).upper() for pos in open_positions}
            if symbol not in open_symbols:
                match = difflib.get_close_matches(symbol, list(open_symbols), n=1, cutoff=0.75)
                if match:
                    symbol = match[0]
                    market = "crypto" if "/" in symbol else "stock"
                else:
                    self.telegram.send_message(
                        f"No open position found for {symbol}.",
                        parse_mode="",
                    )
                    return

            market_price = self._get_manual_market_price(symbol, market)
            if market_price <= 0:
                self.telegram.send_message(
                    f"Manual sell failed: no live price available for {symbol}.",
                    parse_mode="",
                )
                return

            close_result = self.paper_executor.close_position(
                symbol=symbol,
                market_price=market_price,
                market=market,
            )
            if not close_result:
                self.telegram.send_message(
                    f"Manual sell failed for {symbol}.",
                    parse_mode="",
                )
                return

            self.telegram.send_message(
                "MANUAL SELL EXECUTED\n\n"
                f"Symbol: {symbol}\n"
                f"Fill: ${float(close_result.get('fill_price', market_price)):,.2f}\n"
                f"PnL: ${float(close_result.get('pnl', 0.0)):+,.2f}",
                parse_mode="",
            )
            logger.info("manual_sell_executed | symbol={} market={}", symbol, market)
        except Exception as exc:
            logger.exception("manual_sell_error | symbol={} err={}", raw_symbol, exc)
            self.telegram.send_message(
                f"Manual sell failed: {exc}",
                parse_mode="",
            )

    def poll_telegram_commands(self) -> None:
        """Poll for incoming Telegram bot commands and respond."""
        try:
            updates = self.telegram.get_updates()
            if not updates:
                return

            for update in updates:
                msg = update.get("message", {})
                text = (msg.get("text") or "").strip()
                chat_id = str(msg.get("chat", {}).get("id", ""))

                if chat_id != self.telegram._chat_id:
                    continue

                parsed = self._parse_telegram_instruction(text)
                if not parsed:
                    continue

                cmd, argument = parsed

                if cmd == "/status":
                    self._cmd_status()
                elif cmd == "/positions":
                    self._cmd_positions()
                elif cmd == "/pnl":
                    self._cmd_pnl()
                elif cmd == "/config":
                    self._cmd_config()
                elif cmd == "/kill":
                    self._cmd_kill()
                elif cmd == "/resume":
                    self._cmd_resume()
                elif cmd == "/buy":
                    self._cmd_buy(argument)
                elif cmd == "/sell":
                    self._cmd_sell(argument)
                elif cmd == "/help":
                    self._cmd_help()
                else:
                    self.telegram.send_message(
                        f"Unknown command: {cmd}\nType /help for available commands.",
                        parse_mode="",
                    )
        except Exception:
            logger.debug("telegram_poll_error")

    def _cmd_status(self) -> None:
        equity = 0.0
        cash = 0.0
        positions_count = 0
        if self.paper_trade and self.paper_executor:
            portfolio = self.paper_executor._portfolio
            cash = portfolio.cash
            pos = self.paper_executor.get_positions()
            positions_count = len(pos)
            invested = sum(abs(p.get("qty", 0)) * p.get("avg_entry_price", 0) for p in pos)
            unrealized = sum(p.get("unrealized_pl", 0.0) for p in pos)
            equity = cash + invested + unrealized

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl = self.db.get_daily_pnl(today_str)
        uptime = time.time() - self._start_time
        h, m = int(uptime // 3600), int((uptime % 3600) // 60)
        mode = "PAPER" if self.paper_trade else "LIVE"
        halted = "YES" if self.kill_switch.halted else "No"
        hours_since_signal = (time.time() - self._last_signal_time) / 3600

        self.telegram.send_message(
            f"Status: {mode} mode\n"
            f"Equity: ${equity:,.0f} | Cash: ${cash:,.0f}\n"
            f"Positions: {positions_count}\n"
            f"Daily P&L: ${daily_pnl:+,.0f}\n"
            f"Kill switch: {halted}\n"
            f"Last signal: {hours_since_signal:.1f}h ago\n"
            f"Uptime: {h}h {m}m",
            parse_mode="",
        )

    def _cmd_positions(self) -> None:
        if not (self.paper_trade and self.paper_executor):
            self.telegram.send_message("No paper executor active.", parse_mode="")
            return
        pos = self.paper_executor.get_positions()
        if not pos:
            self.telegram.send_message("No open positions.", parse_mode="")
            return
        lines = ["Open Positions:"]
        for p in pos:
            sym = p.get("symbol", "?")
            side = p.get("side", "?")
            entry = p.get("avg_entry_price", 0)
            pnl = p.get("unrealized_pl", 0.0)
            lines.append(f"  {sym} {side} @ {entry:,.2f} (P&L: ${pnl:+,.0f})")
        self.telegram.send_message("\n".join(lines), parse_mode="")

    def _cmd_pnl(self) -> None:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_pnl = self.db.get_daily_pnl(today_str)
        trades = self.db.get_closed_trades(limit=10)
        lines = [f"Daily P&L: ${daily_pnl:+,.2f}", "", "Last 10 trades:"]
        for t in (trades or []):
            sym = t.get("symbol", "?")
            pnl = t.get("pnl", 0.0)
            outcome = t.get("outcome", "?")
            strategy = t.get("strategy", "?")
            lines.append(f"  {sym} {strategy} {outcome} ${pnl:+,.2f}")
        self.telegram.send_message("\n".join(lines), parse_mode="")

    def _cmd_config(self) -> None:
        strats = self.config.get("strategies", {})
        lines = ["Strategy configs:"]
        for name, cfg in strats.items():
            enabled = cfg.get("enabled", True)
            status = "ON" if enabled else "OFF"
            lines.append(f"  {name}: {status}")
        risk = self.config.get("risk", {})
        lines.append(f"\nRisk:")
        lines.append(f"  position_size: {risk.get('position_size_pct', 'N/A')}")
        lines.append(f"  stop_loss: {risk.get('stop_loss_pct', 'N/A')}")
        lines.append(f"  max_positions: {risk.get('max_positions', 'N/A')}")
        self.telegram.send_message("\n".join(lines), parse_mode="")

    def _cmd_kill(self) -> None:
        self.kill_switch._halted = True
        self.telegram.send_message(
            "Kill switch ACTIVATED manually.\nAll trading halted.",
            parse_mode="",
        )
        logger.warning("kill_switch_manual_activate")

    def _cmd_resume(self) -> None:
        self.kill_switch.reset()
        self.telegram.send_message(
            "Kill switch RESET. Trading resumed.",
            parse_mode="",
        )
        logger.info("kill_switch_manual_reset")

    def _cmd_help(self) -> None:
        self.telegram.send_message(
            "VarpTrader Commands:\n"
            "/status - Bot status, equity, uptime\n"
            "/positions - Open positions with P&L\n"
            "/pnl - Daily P&L and last 10 trades\n"
            "/config - Strategy and risk config\n"
            "/buy SYMBOL - Open a manual paper long (also supports 'buy SYMBOL')\n"
            "/sell SYMBOL - Close an open paper position (also supports 'sell SYMBOL')\n"
            "/kill - Activate kill switch\n"
            "/resume - Reset kill switch\n"
            "/help - This message",
            parse_mode="",
        )

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
    dashboard_init(
        db=trader.db,
        analyzer=trader.analyzer,
        config=trader.config,
        kill_switch=trader.kill_switch,
        paper_executor=trader.paper_executor,
        stock_feed=trader.stock_feed,
        crypto_feed=trader.crypto_feed,
        order_validator=trader.order_validator,
    )

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

    # -- Start Binance WebSocket for real-time crypto prices ---------------
    crypto_symbols = trader.config.get("watchlist", {}).get("crypto", [])
    if crypto_symbols:
        trader.crypto_feed.start_stream(crypto_symbols, timeframe="5m")

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

    # -- Heartbeat every 2 hours ------------------------------------------
    scheduler.add_job(
        trader.send_heartbeat,
        IntervalTrigger(hours=2),
        id="heartbeat",
        max_instances=1,
    )

    # -- Signal starvation check every 30 minutes -------------------------
    scheduler.add_job(
        trader.check_signal_starvation,
        IntervalTrigger(minutes=30),
        id="starvation_check",
        max_instances=1,
    )

    # -- Telegram command polling every 10 seconds -------------------------
    scheduler.add_job(
        trader.poll_telegram_commands,
        IntervalTrigger(seconds=10),
        id="telegram_commands",
        max_instances=1,
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
        if strat_cfg.get("vpoc_bounce", {}).get("enabled", False):
            signals_list.append(VPOCBounceSignal(strat_cfg["vpoc_bounce"]))
        if strat_cfg.get("macd_divergence", {}).get("enabled", False):
            signals_list.append(MACDDivergenceSignal(strat_cfg["macd_divergence"]))

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
    paper_executor = None
    if os.getenv("PAPER_TRADE", "true").lower() == "true":
        portfolio = PaperPortfolio(initial_capital=100_000.0)
        paper_executor = PaperExecutor(
            portfolio=portfolio, db=db, slippage_pct=0.001
        )

    dashboard_init(
        db=db,
        analyzer=analyzer,
        config=config,
        paper_executor=paper_executor,
    )

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
