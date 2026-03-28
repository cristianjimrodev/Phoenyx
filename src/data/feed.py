"""Data feed combining real-time and historical data."""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.broker.base import BrokerClient, TickPrice
from src.broker.xtb.models import PERIODS
from src.data.store import DataStore


class DataFeed:
    """Manages market data: fetches history, caches locally, and streams live prices."""

    def __init__(self, broker: BrokerClient, store: DataStore):
        self._broker = broker
        self._store = store
        self._live_prices: dict[str, TickPrice] = {}

    async def get_candles(self, symbol: str, timeframe: str,
                          count: int = 500) -> pd.DataFrame:
        period = PERIODS.get(timeframe, 60)

        # Try local cache first
        df = self._store.load_candles(symbol, period, limit=count)

        # Fetch from broker if insufficient data
        if len(df) < count:
            logger.info(f"Fetching {count} candles for {symbol}/{timeframe} from broker")
            df = await self._broker.get_candles(symbol, period, count)
            if not df.empty:
                self._store.save_candles(symbol, period, df)

        return df

    async def subscribe(self, symbol: str) -> None:
        async def on_tick(tick: TickPrice) -> None:
            self._live_prices[symbol] = tick

        await self._broker.subscribe_prices(symbol, on_tick)

    def get_latest_price(self, symbol: str) -> TickPrice | None:
        return self._live_prices.get(symbol)

    async def unsubscribe(self, symbol: str) -> None:
        await self._broker.unsubscribe_prices(symbol)
        self._live_prices.pop(symbol, None)
