"""Shared fixtures for Phoenyx test suite."""
from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from src.broker.base import AccountInfo, BrokerClient, Side, Trade, OrderStatus


def make_ohlcv_df(
    length: int = 200,
    trend: str = "up",
    base_price: float = 1.1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame.

    Args:
        length: Number of bars.
        trend: One of "up", "down", "flat", "volatile".
        base_price: Starting price.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    drift_map = {
        "up": 0.0002,
        "down": -0.0002,
        "flat": 0.0,
        "volatile": 0.0,
    }
    noise_scale = 0.005 if trend == "volatile" else 0.001

    drift = drift_map[trend]
    returns = drift + rng.normal(0, noise_scale, size=length)
    close = base_price + np.cumsum(returns)
    # Ensure prices stay positive
    close = np.maximum(close, 0.01)

    open_ = np.roll(close, 1)
    open_[0] = base_price

    noise_hl = np.abs(rng.normal(0, noise_scale * 0.5, size=length))
    high = np.maximum(open_, close) + noise_hl
    low = np.minimum(open_, close) - noise_hl
    volume = rng.uniform(1000, 5000, size=length)

    timestamps = np.arange(length) * 3600  # 1-hour bars from epoch

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.index = pd.to_datetime(df["timestamp"], unit="s")
    df.index.name = "datetime"
    return df


def make_price_sequence(prices: list[float], seed: int = 42) -> pd.DataFrame:
    """Build an OHLCV DataFrame from an explicit close-price sequence."""
    rng = np.random.default_rng(seed)
    n = len(prices)
    close = np.array(prices, dtype=float)

    open_ = np.roll(close, 1)
    open_[0] = close[0]

    noise = np.abs(rng.normal(0, 0.0005, size=n))
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = rng.uniform(1000, 5000, size=n)

    timestamps = np.arange(n) * 3600

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.index = pd.to_datetime(df["timestamp"], unit="s")
    df.index.name = "datetime"
    return df


# --------------- Config fixtures ---------------

@pytest.fixture
def indicator_config():
    return {
        "rsi": {"period": 14, "overbought": 70, "oversold": 30},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "moving_averages": {"fast": 20, "slow": 50, "long": 200},
        "bollinger": {"period": 20, "std_dev": 2},
        "atr": {"period": 14},
    }


@pytest.fixture
def sr_config():
    return {
        "lookback_periods": 100,
        "min_touches": 2,
        "tolerance_pct": 0.001,
        "trend_line_min_points": 3,
    }


@pytest.fixture
def pattern_config():
    return {
        "zigzag_threshold": 0.03,
        "min_pattern_bars": 10,
        "max_pattern_bars": 100,
        "enabled": [],
    }


@pytest.fixture
def risk_config():
    return {
        "max_risk_per_trade": 0.02,
        "max_daily_drawdown": 0.05,
        "max_open_positions": 5,
        "default_rr_ratio": 2.0,
    }


@pytest.fixture
def mock_account():
    return AccountInfo(
        balance=10000,
        equity=10000,
        margin=0,
        free_margin=10000,
        margin_level=0,
        currency="USD",
    )


@pytest.fixture
def mock_broker():
    broker = AsyncMock(spec=BrokerClient)
    broker.open_trade.return_value = Trade(
        trade_id=1,
        symbol="EURUSD",
        side=Side.BUY,
        volume=0.1,
        open_price=1.1000,
        open_time=0,
        status=OrderStatus.OPENED,
    )
    broker.close_trade.return_value = True
    return broker
