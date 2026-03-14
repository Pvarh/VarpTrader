"""Order book wall detection and imbalance analysis.

Scans a CCXT-formatted order book for significant bid/ask walls and
computes an imbalance ratio to gauge short-term directional bias.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class OrderBookScanner:
    """Scans order book for large walls using CCXT order-book data.

    A *wall* is defined as a single resting order whose size exceeds a
    configurable percentage of the total visible depth on that side.
    The scanner also computes a bid/ask imbalance ratio that strategies
    can use as a momentum filter.

    Parameters
    ----------
    wall_threshold_pct:
        Minimum fraction of total side depth for a single level to be
        classified as a wall.  For example, ``0.05`` means a single
        order must represent at least 5 % of all visible bid (or ask)
        volume.
    """

    def __init__(self, wall_threshold_pct: float = 0.05) -> None:
        self._wall_threshold: float = wall_threshold_pct

        logger.info(
            "orderbook_scanner_initialised | wall_threshold_pct={wall_threshold_pct}",
            wall_threshold_pct=wall_threshold_pct,
        )

    # ------------------------------------------------------------------
    # Wall scanning
    # ------------------------------------------------------------------
    def scan_for_walls(
        self,
        order_book: dict[str, Any],
        current_price: float,
    ) -> dict[str, Any]:
        """Analyse an order book for significant bid/ask walls.

        Parameters
        ----------
        order_book:
            A CCXT-formatted order book dictionary containing at least
            ``"bids"`` and ``"asks"`` keys, each being a list of
            ``[price, amount]`` pairs sorted by price (best first).
        current_price:
            The current market price, used for proximity calculations.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``bid_wall`` (bool): Whether a significant bid wall was
              found.
            - ``ask_wall`` (bool): Whether a significant ask wall was
              found.
            - ``bid_wall_price`` (float | None): Price level of the
              largest bid wall, or *None*.
            - ``ask_wall_price`` (float | None): Price level of the
              largest ask wall, or *None*.
            - ``bid_wall_size`` (float | None): Size (quantity) at the
              bid wall level, or *None*.
            - ``ask_wall_size`` (float | None): Size (quantity) at the
              ask wall level, or *None*.
            - ``imbalance_ratio`` (float): Ratio of total bid depth to
              total ask depth.  Values > 1.0 indicate stronger bids;
              values < 1.0 indicate stronger asks.

            Returns default empty-state values on error.
        """
        default_result: dict[str, Any] = {
            "bid_wall": False,
            "ask_wall": False,
            "bid_wall_price": None,
            "ask_wall_price": None,
            "bid_wall_size": None,
            "ask_wall_size": None,
            "imbalance_ratio": 1.0,
        }

        try:
            bids: list[list[float]] = order_book.get("bids", [])
            asks: list[list[float]] = order_book.get("asks", [])

            if not bids or not asks:
                logger.warning(
                    "order_book_empty | current_price={current_price} bids={bids_count} asks={asks_count}",
                    current_price=current_price,
                    bids_count=len(bids),
                    asks_count=len(asks),
                )
                return default_result

            # ----------------------------------------------------------
            # Total visible depth
            # ----------------------------------------------------------
            total_bid_volume: float = sum(level[1] for level in bids)
            total_ask_volume: float = sum(level[1] for level in asks)

            # ----------------------------------------------------------
            # Imbalance ratio
            # ----------------------------------------------------------
            imbalance_ratio: float = (
                total_bid_volume / total_ask_volume
                if total_ask_volume > 0
                else 1.0
            )

            # ----------------------------------------------------------
            # Detect bid wall (largest single bid level)
            # ----------------------------------------------------------
            bid_wall: bool = False
            bid_wall_price: float | None = None
            bid_wall_size: float | None = None

            if total_bid_volume > 0:
                largest_bid = max(bids, key=lambda lvl: lvl[1])
                largest_bid_pct: float = largest_bid[1] / total_bid_volume
                if largest_bid_pct >= self._wall_threshold:
                    bid_wall = True
                    bid_wall_price = largest_bid[0]
                    bid_wall_size = largest_bid[1]

            # ----------------------------------------------------------
            # Detect ask wall (largest single ask level)
            # ----------------------------------------------------------
            ask_wall: bool = False
            ask_wall_price: float | None = None
            ask_wall_size: float | None = None

            if total_ask_volume > 0:
                largest_ask = max(asks, key=lambda lvl: lvl[1])
                largest_ask_pct: float = largest_ask[1] / total_ask_volume
                if largest_ask_pct >= self._wall_threshold:
                    ask_wall = True
                    ask_wall_price = largest_ask[0]
                    ask_wall_size = largest_ask[1]

            result: dict[str, Any] = {
                "bid_wall": bid_wall,
                "ask_wall": ask_wall,
                "bid_wall_price": bid_wall_price,
                "ask_wall_price": ask_wall_price,
                "bid_wall_size": bid_wall_size,
                "ask_wall_size": ask_wall_size,
                "imbalance_ratio": round(imbalance_ratio, 4),
            }

            logger.info(
                "order_book_scanned | current_price={current_price} bid_wall={bid_wall} ask_wall={ask_wall} bid_wall_price={bid_wall_price} ask_wall_price={ask_wall_price} imbalance_ratio={imbalance_ratio} total_bid_volume={total_bid_volume} total_ask_volume={total_ask_volume}",
                current_price=current_price,
                bid_wall=bid_wall,
                ask_wall=ask_wall,
                bid_wall_price=bid_wall_price,
                ask_wall_price=ask_wall_price,
                imbalance_ratio=round(imbalance_ratio, 4),
                total_bid_volume=round(total_bid_volume, 4),
                total_ask_volume=round(total_ask_volume, 4),
            )

            return result

        except Exception:
            logger.exception("order_book_scan_error | current_price={current_price}", current_price=current_price)
            return default_result
