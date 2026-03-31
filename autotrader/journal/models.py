"""Trade data models and schema definitions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Trade:
    """Represents a single trade record."""

    symbol: str
    market: str  # 'stock' or 'crypto'
    strategy: str
    direction: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    outcome: str = "open"  # 'win', 'loss', 'breakeven', 'open'
    whale_flag: int = 0
    day_of_week: Optional[str] = None
    hour_of_day: Optional[int] = None
    market_condition: Optional[str] = None  # 'trending', 'ranging', 'volatile'
    paper_trade: int = 0
    slippage: float = 0.0
    swing_bias: Optional[str] = None
    swing_confidence: Optional[int] = None

    def __post_init__(self) -> None:
        """Derive day_of_week and hour_of_day from timestamp if not set."""
        if self.day_of_week is None or self.hour_of_day is None:
            dt = datetime.fromisoformat(self.timestamp)
            self.day_of_week = dt.strftime("%A")
            self.hour_of_day = dt.hour

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return asdict(self)

    def compute_pnl(self) -> None:
        """Calculate PnL fields when exit_price is set."""
        if self.exit_price is None:
            return
        if self.direction == "long":
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
            self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
            self.pnl_pct = (self.entry_price - self.exit_price) / self.entry_price
        if self.pnl > 0:
            self.outcome = "win"
        elif self.pnl < 0:
            self.outcome = "loss"
        else:
            self.outcome = "breakeven"


@dataclass
class AnalysisRun:
    """Represents a single AI analysis run."""

    run_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: Optional[int] = None
    trades_analyzed: int = 0
    report_markdown: Optional[str] = None
    config_changes_json: Optional[str] = None
    approved: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return asdict(self)


@dataclass
class OHLCV:
    """Normalized OHLCV bar used across all data feeds."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str  # '1m', '5m', '15m', '1h', '1d'
    market: str  # 'stock' or 'crypto'
    vwap: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class SwingBias:
    """Weekly swing bias for a single symbol."""

    symbol: str
    bias: str  # 'bullish', 'bearish', 'neutral'
    confidence: int  # 0-100
    reason: Optional[str] = None
    week_start: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    id: Optional[int] = None
    claude_raw_response: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
