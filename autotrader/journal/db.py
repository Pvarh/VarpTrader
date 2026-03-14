"""SQLite database setup, schema management, and parameterized query operations."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from loguru import logger

from journal.models import Trade, AnalysisRun, SwingBias

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    strategy TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    pnl REAL,
    pnl_pct REAL,
    outcome TEXT,
    whale_flag INTEGER DEFAULT 0,
    day_of_week TEXT,
    hour_of_day INTEGER,
    market_condition TEXT,
    paper_trade INTEGER DEFAULT 0,
    swing_bias TEXT,
    swing_confidence INTEGER
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    trades_analyzed INTEGER,
    report_markdown TEXT,
    config_changes_json TEXT,
    approved INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS swing_bias_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start TEXT NOT NULL,
    symbol TEXT NOT NULL,
    bias TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    reason TEXT,
    claude_raw_response TEXT
);
"""


class TradeDatabase:
    """Thread-safe SQLite database for trade journaling."""

    def __init__(self, db_path: str = "data/trades.db") -> None:
        """Initialize database connection and create schema.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info("trade_database_initialized | db_path={db_path}", db_path=str(self._db_path))

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path))
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    @contextmanager
    def _cursor(self):
        """Context manager providing a database cursor with auto-commit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    def insert_trade(self, trade: Trade) -> int:
        """Insert a new trade record.

        Args:
            trade: Trade dataclass instance to insert.

        Returns:
            The row ID of the inserted trade.
        """
        sql = """
            INSERT INTO trades (
                timestamp, symbol, market, strategy, direction,
                entry_price, exit_price, quantity, stop_loss, take_profit,
                pnl, pnl_pct, outcome, whale_flag, day_of_week,
                hour_of_day, market_condition, paper_trade,
                swing_bias, swing_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            trade.timestamp, trade.symbol, trade.market, trade.strategy,
            trade.direction, trade.entry_price, trade.exit_price,
            trade.quantity, trade.stop_loss, trade.take_profit,
            trade.pnl, trade.pnl_pct, trade.outcome, trade.whale_flag,
            trade.day_of_week, trade.hour_of_day, trade.market_condition,
            trade.paper_trade, trade.swing_bias, trade.swing_confidence,
        )
        with self._cursor() as cur:
            cur.execute(sql, params)
            trade_id = cur.lastrowid
        logger.info(
            "trade_inserted | trade_id={trade_id} symbol={symbol} strategy={strategy}",
            trade_id=trade_id, symbol=trade.symbol, strategy=trade.strategy,
        )
        return trade_id

    def update_trade_exit(
        self, trade_id: int, exit_price: float, pnl: float,
        pnl_pct: float, outcome: str
    ) -> None:
        """Update a trade with exit information.

        Args:
            trade_id: Database ID of the trade to update.
            exit_price: The price at which the position was closed.
            pnl: Realized profit/loss in dollar terms.
            pnl_pct: Realized profit/loss as a percentage.
            outcome: Trade outcome ('win', 'loss', 'breakeven').
        """
        sql = """
            UPDATE trades
            SET exit_price = ?, pnl = ?, pnl_pct = ?, outcome = ?
            WHERE id = ?
        """
        with self._cursor() as cur:
            cur.execute(sql, (exit_price, pnl, pnl_pct, outcome, trade_id))
        logger.info(
            "trade_exit_updated | trade_id={trade_id} outcome={outcome} pnl={pnl}",
            trade_id=trade_id, outcome=outcome, pnl=pnl,
        )

    def get_trade_by_id(self, trade_id: int) -> Optional[dict]:
        """Fetch a single trade by its ID.

        Args:
            trade_id: Database ID of the trade.

        Returns:
            Dictionary of trade data or None if not found.
        """
        with self._cursor() as cur:
            cur.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def get_open_trades(self) -> list[dict]:
        """Fetch all trades with outcome = 'open'.

        Returns:
            List of dictionaries representing open trades.
        """
        with self._cursor() as cur:
            cur.execute("SELECT * FROM trades WHERE outcome = ?", ("open",))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_trades_since(self, since_iso: str) -> list[dict]:
        """Fetch all trades after a given timestamp.

        Args:
            since_iso: ISO format timestamp string.

        Returns:
            List of trade dictionaries.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp",
                (since_iso,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_daily_pnl(self, date_str: str) -> float:
        """Calculate total PnL for a specific date.

        Args:
            date_str: Date string in YYYY-MM-DD format.

        Returns:
            Sum of PnL for the date, or 0.0 if no trades.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(pnl), 0.0) FROM trades WHERE timestamp LIKE ?",
                (f"{date_str}%",),
            )
            result = cur.fetchone()
        return result[0] if result else 0.0

    def get_closed_trades_for_period(self, start_iso: str, end_iso: str) -> list[dict]:
        """Fetch closed trades within a date range.

        Args:
            start_iso: Start timestamp (inclusive).
            end_iso: End timestamp (inclusive).

        Returns:
            List of closed trade dictionaries.
        """
        with self._cursor() as cur:
            cur.execute(
                """SELECT * FROM trades
                   WHERE timestamp BETWEEN ? AND ?
                   AND outcome != 'open'
                   ORDER BY timestamp""",
                (start_iso, end_iso),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def insert_analysis_run(self, run: AnalysisRun) -> int:
        """Insert an analysis run record.

        Args:
            run: AnalysisRun dataclass instance.

        Returns:
            The row ID of the inserted record.
        """
        sql = """
            INSERT INTO analysis_runs (
                run_timestamp, trades_analyzed, report_markdown,
                config_changes_json, approved
            ) VALUES (?, ?, ?, ?, ?)
        """
        params = (
            run.run_timestamp, run.trades_analyzed, run.report_markdown,
            run.config_changes_json, run.approved,
        )
        with self._cursor() as cur:
            cur.execute(sql, params)
            run_id = cur.lastrowid
        logger.info("analysis_run_inserted | run_id={run_id}", run_id=run_id)
        return run_id

    def get_all_strategies(self) -> list[str]:
        """Get a list of all unique strategies used in trades.

        Returns:
            List of strategy name strings.
        """
        with self._cursor() as cur:
            cur.execute("SELECT DISTINCT strategy FROM trades")
            rows = cur.fetchall()
        return [r["strategy"] for r in rows]

    def insert_swing_bias(self, bias: SwingBias) -> int:
        """Insert a swing bias log record.

        Args:
            bias: SwingBias dataclass instance.

        Returns:
            The row ID of the inserted record.
        """
        sql = """
            INSERT INTO swing_bias_log (
                week_start, symbol, bias, confidence, reason,
                claude_raw_response
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            bias.week_start, bias.symbol, bias.bias, bias.confidence,
            bias.reason, bias.claude_raw_response,
        )
        with self._cursor() as cur:
            cur.execute(sql, params)
            bias_id = cur.lastrowid
        logger.info(
            "swing_bias_inserted | bias_id={bias_id} symbol={symbol} bias={bias} confidence={confidence}",
            bias_id=bias_id, symbol=bias.symbol, bias=bias.bias,
            confidence=bias.confidence,
        )
        return bias_id

    def get_swing_biases_for_week(self, week_start: str) -> list[dict]:
        """Get all swing biases for a given week.

        Args:
            week_start: Week start date string in YYYY-MM-DD format.

        Returns:
            List of swing bias dictionaries.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM swing_bias_log WHERE week_start = ? ORDER BY symbol",
                (week_start,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the thread-local database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
