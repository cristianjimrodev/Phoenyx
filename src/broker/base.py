from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable

import pandas as pd


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    OPENED = "opened"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class Symbol:
    name: str
    description: str
    category: str
    currency: str
    lot_min: float
    lot_max: float
    lot_step: float
    pip_size: float
    contract_size: float
    leverage: float
    swap_long: float = 0.0
    swap_short: float = 0.0


@dataclass
class Candle:
    timestamp: int      # milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TickPrice:
    symbol: str
    ask: float
    bid: float
    spread: float
    timestamp: int


@dataclass
class Trade:
    trade_id: int
    symbol: str
    side: Side
    volume: float
    open_price: float
    open_time: int
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    close_price: float = 0.0
    close_time: int = 0
    status: OrderStatus = OrderStatus.OPENED


@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str


PriceCallback = Callable[[TickPrice], Awaitable[None]]


class BrokerClient(ABC):
    """Abstract base class for broker API clients."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect and authenticate with the broker."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""

    @abstractmethod
    async def get_symbols(self) -> list[Symbol]:
        """Get all available trading symbols."""

    @abstractmethod
    async def get_symbol(self, symbol: str) -> Symbol:
        """Get info for a specific symbol."""

    @abstractmethod
    async def get_candles(self, symbol: str, period: int, count: int) -> pd.DataFrame:
        """Get historical candles as a DataFrame with columns: timestamp, open, high, low, close, volume."""

    @abstractmethod
    async def subscribe_prices(self, symbol: str, callback: PriceCallback) -> None:
        """Subscribe to real-time price updates for a symbol."""

    @abstractmethod
    async def unsubscribe_prices(self, symbol: str) -> None:
        """Unsubscribe from price updates."""

    @abstractmethod
    async def open_trade(self, symbol: str, side: Side, volume: float,
                         sl: float = 0, tp: float = 0) -> Trade:
        """Open a new trade position."""

    @abstractmethod
    async def close_trade(self, trade_id: int) -> bool:
        """Close an open trade by ID."""

    @abstractmethod
    async def modify_trade(self, trade_id: int, sl: float = 0, tp: float = 0) -> bool:
        """Modify SL/TP of an open trade."""

    @abstractmethod
    async def get_open_trades(self) -> list[Trade]:
        """Get all currently open trades."""

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get current account balance and margin info."""
