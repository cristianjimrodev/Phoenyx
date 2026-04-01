"""Correlation matrix computation for portfolio risk management."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from loguru import logger


class CorrelationMatrix:
    """Computes and caches pairwise correlation between symbols."""

    def __init__(self, lookback: int = 100, update_interval: int = 3600):
        self._lookback = lookback
        self._update_interval = update_interval
        self._matrix: pd.DataFrame | None = None
        self._last_update: float = 0

    def update(self, candle_data: dict[str, pd.DataFrame]) -> None:
        """Recompute correlation matrix from close-price returns."""
        if len(candle_data) < 2:
            self._matrix = None
            return

        returns = {}
        for symbol, df in candle_data.items():
            if df.empty or len(df) < self._lookback:
                continue
            close = df["close"].tail(self._lookback)
            returns[symbol] = close.pct_change().dropna()

        if len(returns) < 2:
            self._matrix = None
            return

        # Align all return series by index
        returns_df = pd.DataFrame(returns)
        self._matrix = returns_df.corr()
        self._last_update = time.time()
        logger.info(f"Correlation matrix updated for {len(returns)} symbols")

    @property
    def needs_update(self) -> bool:
        return time.time() - self._last_update > self._update_interval

    def get_correlation(self, sym_a: str, sym_b: str) -> float:
        if self._matrix is None:
            return 0.0
        if sym_a not in self._matrix.index or sym_b not in self._matrix.columns:
            return 0.0
        return float(self._matrix.loc[sym_a, sym_b])

    def get_correlated_symbols(self, symbol: str, threshold: float = 0.7) -> list[tuple[str, float]]:
        if self._matrix is None or symbol not in self._matrix.index:
            return []
        row = self._matrix.loc[symbol]
        result = []
        for sym, corr in row.items():
            if sym != symbol and abs(corr) >= threshold:
                result.append((sym, float(corr)))
        return sorted(result, key=lambda x: abs(x[1]), reverse=True)
