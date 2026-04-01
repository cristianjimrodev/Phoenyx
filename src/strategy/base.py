"""Base strategy interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.analysis.indicators import Signal


@dataclass
class TradeSignal:
    symbol: str
    signal: Signal
    confidence: float       # 0-100
    suggested_sl: float     # suggested stop loss price
    suggested_tp: float     # suggested take profit price
    details: list[str]      # reasoning breakdown


class Strategy(ABC):
    """Abstract base for trading strategies."""

    @abstractmethod
    async def evaluate(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        """Evaluate market data and return a trade signal."""

    async def evaluate_mtf(self, symbol: str, dataframes: dict[str, pd.DataFrame]) -> TradeSignal:
        """Multi-timeframe evaluation. Default: use first available DataFrame."""
        if dataframes:
            first_df = next(iter(dataframes.values()))
            return await self.evaluate(symbol, first_df)
        return TradeSignal(symbol, Signal.HOLD, 0, 0, 0, ["No data"])
