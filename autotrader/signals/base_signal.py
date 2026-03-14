"""Abstract base class for all trading signals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from journal.models import OHLCV


class SignalDirection(Enum):
    """Direction of a trading signal."""
    LONG = "long"
    SHORT = "short"


@dataclass
class SignalResult:
    """Output of a signal evaluation."""

    triggered: bool
    direction: Optional[SignalDirection] = None
    symbol: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    strategy_name: str = ""
    reason: str = ""


class BaseSignal(ABC):
    """Abstract base class that all signal strategies must implement."""

    def __init__(self, config: dict) -> None:
        """Initialize with strategy-specific configuration.

        Args:
            config: Dictionary of strategy parameters from config.json.
        """
        self.config = config
        self.enabled = config.get("enabled", True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this signal strategy."""
        ...

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_price: float,
        market: str,
    ) -> SignalResult:
        """Evaluate market data and return a signal result.

        Args:
            symbol: Trading symbol (e.g., 'AAPL' or 'BTC/USDT').
            candles: List of OHLCV bars, most recent last.
            current_price: The latest market price.
            market: 'stock' or 'crypto'.

        Returns:
            SignalResult indicating whether a trade should be taken.
        """
        ...

    def is_enabled(self) -> bool:
        """Check if this signal is enabled in config."""
        return self.enabled

    def check_swing_bias(self, symbol: str, direction: SignalDirection) -> bool:
        """Check if swing bias allows this trade direction.

        Returns True if trade is allowed, False if blocked.
        """
        from analysis.swing_advisor import SwingAdvisor
        min_conf = 60  # default
        return not SwingAdvisor.should_block_trade(symbol, direction.value, min_conf)
