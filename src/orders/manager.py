"""Order execution and tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from src.broker.base import BrokerClient, Side, Trade, OrderStatus
from src.strategy.base import TradeSignal
from src.analysis.indicators import Signal


@dataclass
class OrderRecord:
    timestamp: str
    symbol: str
    side: str
    volume: float
    entry_price: float
    sl: float
    tp: float
    confidence: float
    reasons: list[str]
    trade_id: int = 0
    status: str = "pending"
    exit_price: float = 0.0
    pnl: float = 0.0


class OrderManager:
    """Executes trade signals and tracks order history."""

    def __init__(self, broker: BrokerClient):
        self._broker = broker
        self._history: list[OrderRecord] = []

    async def execute_signal(self, signal: TradeSignal, volume: float,
                             sl: float, tp: float) -> Trade | None:
        """Execute a trade signal through the broker."""
        if signal.signal == Signal.HOLD:
            return None

        side = Side.BUY if signal.signal == Signal.BUY else Side.SELL

        record = OrderRecord(
            timestamp=datetime.now().isoformat(),
            symbol=signal.symbol,
            side=side.value,
            volume=volume,
            entry_price=0,
            sl=sl,
            tp=tp,
            confidence=signal.confidence,
            reasons=signal.details,
        )

        try:
            trade = await self._broker.open_trade(
                symbol=signal.symbol,
                side=side,
                volume=volume,
                sl=sl,
                tp=tp,
            )

            record.trade_id = trade.trade_id
            record.entry_price = trade.open_price
            record.status = "opened"
            self._history.append(record)

            logger.info(
                f"ORDER EXECUTED: {side.value} {volume} {signal.symbol} "
                f"@ {trade.open_price} SL={sl} TP={tp} "
                f"(confidence={signal.confidence:.0f}%)"
            )
            return trade

        except Exception as e:
            record.status = "error"
            self._history.append(record)
            logger.error(f"Order execution failed: {e}")
            return None

    async def close_position(self, trade_id: int) -> bool:
        result = await self._broker.close_trade(trade_id)
        if result:
            logger.info(f"Position {trade_id} closed")
        return result

    def get_history(self) -> list[OrderRecord]:
        return self._history.copy()

    def get_stats(self) -> dict:
        if not self._history:
            return {"total": 0}

        total = len(self._history)
        opened = sum(1 for o in self._history if o.status == "opened")
        errors = sum(1 for o in self._history if o.status == "error")

        return {
            "total": total,
            "opened": opened,
            "errors": errors,
        }
