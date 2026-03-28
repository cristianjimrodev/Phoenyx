"""Local SQLite cache for historical candle data."""

import sqlite3
from pathlib import Path

import pandas as pd
from loguru import logger


class DataStore:
    """Stores and retrieves historical candle data locally."""

    def __init__(self, db_path: str = "data/candles.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._create_tables()
        logger.info(f"DataStore initialized at {db_path}")

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                period INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, period, timestamp)
            )
        """)
        self._conn.commit()

    def save_candles(self, symbol: str, period: int, df: pd.DataFrame) -> None:
        if df.empty:
            return

        records = []
        for _, row in df.iterrows():
            records.append((
                symbol, period,
                int(row["timestamp"]), row["open"], row["high"],
                row["low"], row["close"], row["volume"],
            ))

        self._conn.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )
        self._conn.commit()
        logger.debug(f"Saved {len(records)} candles for {symbol}/{period}")

    def load_candles(self, symbol: str, period: int, limit: int = 500) -> pd.DataFrame:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND period = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, self._conn, params=(symbol, period, limit))
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("datetime")
        return df

    def close(self) -> None:
        self._conn.close()
