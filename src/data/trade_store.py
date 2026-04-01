"""SQLite persistence for trade/order history."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from loguru import logger

from src.orders.manager import OrderRecord


class TradeStore:
    """Stores and retrieves trade order history in SQLite."""

    def __init__(self, db_path: str = "data/trades.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._create_tables()
        logger.info(f"TradeStore initialized at {db_path}")

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                volume REAL NOT NULL,
                entry_price REAL NOT NULL,
                sl REAL NOT NULL,
                tp REAL NOT NULL,
                confidence REAL NOT NULL,
                reasons TEXT NOT NULL,
                trade_id INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                exit_price REAL NOT NULL DEFAULT 0.0,
                pnl REAL NOT NULL DEFAULT 0.0,
                contract_size REAL NOT NULL DEFAULT 100000
            )
        """)
        self._conn.commit()
        # Add contract_size column to existing databases
        try:
            self._conn.execute(
                "ALTER TABLE trades ADD COLUMN contract_size REAL NOT NULL DEFAULT 100000"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    def save_order(self, record: OrderRecord) -> int:
        """Save an order record and return its database row id."""
        cursor = self._conn.execute(
            """INSERT INTO trades
               (timestamp, symbol, side, volume, entry_price, sl, tp,
                confidence, reasons, trade_id, status, exit_price, pnl,
                contract_size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.timestamp,
                record.symbol,
                record.side,
                record.volume,
                record.entry_price,
                record.sl,
                record.tp,
                record.confidence,
                "|".join(record.reasons),
                record.trade_id,
                record.status,
                record.exit_price,
                record.pnl,
                record.contract_size,
            ),
        )
        self._conn.commit()
        row_id = cursor.lastrowid
        logger.debug(f"Saved trade record #{row_id} for {record.symbol}")
        return row_id

    def update_order(self, trade_id: int, status: str,
                     exit_price: float = 0.0, pnl: float = 0.0) -> None:
        """Update an existing trade record by its broker trade_id."""
        self._conn.execute(
            """UPDATE trades SET status = ?, exit_price = ?, pnl = ?
               WHERE trade_id = ? AND status != 'closed'""",
            (status, exit_price, pnl, trade_id),
        )
        self._conn.commit()

    def load_trades(self, symbol: str | None = None,
                    limit: int = 100) -> pd.DataFrame:
        """Load trade history as a DataFrame.

        Args:
            symbol: Filter by symbol, or None for all.
            limit: Max number of records to return (most recent first).
        """
        if symbol:
            query = """
                SELECT * FROM trades WHERE symbol = ?
                ORDER BY id DESC LIMIT ?
            """
            df = pd.read_sql_query(query, self._conn, params=(symbol, limit))
        else:
            query = "SELECT * FROM trades ORDER BY id DESC LIMIT ?"
            df = pd.read_sql_query(query, self._conn, params=(limit,))

        if not df.empty:
            df = df.sort_values("id").reset_index(drop=True)
        return df

    def get_stats(self, symbol: str | None = None) -> dict:
        """Compute summary statistics from persisted trades."""
        df = self.load_trades(symbol=symbol, limit=10000)
        if df.empty:
            return {"total": 0}

        total = len(df)
        opened = len(df[df["status"] == "opened"])
        closed = len(df[df["status"] == "closed"])
        errors = len(df[df["status"] == "error"])

        closed_df = df[df["status"] == "closed"]
        winning = len(closed_df[closed_df["pnl"] > 0])
        losing = len(closed_df[closed_df["pnl"] <= 0])
        win_rate = (winning / len(closed_df) * 100) if len(closed_df) > 0 else 0

        total_pnl = closed_df["pnl"].sum() if not closed_df.empty else 0

        return {
            "total": total,
            "opened": opened,
            "closed": closed,
            "errors": errors,
            "winning": winning,
            "losing": losing,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
        }

    def close(self) -> None:
        self._conn.close()
