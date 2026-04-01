"""Tests for src/orders/manager.py"""
from __future__ import annotations

import pytest

from src.broker.base import Side, Trade, OrderStatus
from src.strategy.base import TradeSignal
from src.analysis.indicators import Signal
from src.orders.manager import OrderManager, OrderRecord


class TestExecuteSignal:
    @pytest.mark.asyncio
    async def test_buy_signal(self, mock_broker):
        om = OrderManager(mock_broker)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=1.0950, suggested_tp=1.1100,
            details=["test reason"],
        )
        trade = await om.execute_signal(signal, volume=0.1, sl=1.0950, tp=1.1100)

        assert trade is not None
        assert trade.trade_id == 1
        mock_broker.open_trade.assert_called_once_with(
            symbol="EURUSD", side=Side.BUY, volume=0.1, sl=1.0950, tp=1.1100,
        )
        history = om.get_history()
        assert len(history) == 1
        assert history[0].status == "opened"
        assert history[0].side == "buy"

    @pytest.mark.asyncio
    async def test_sell_signal(self, mock_broker):
        mock_broker.open_trade.return_value = Trade(
            trade_id=2, symbol="EURUSD", side=Side.SELL,
            volume=0.1, open_price=1.1000, open_time=0,
        )
        om = OrderManager(mock_broker)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.SELL, confidence=75,
            suggested_sl=1.1100, suggested_tp=1.0900,
            details=["sell reason"],
        )
        trade = await om.execute_signal(signal, volume=0.1, sl=1.1100, tp=1.0900)

        assert trade is not None
        mock_broker.open_trade.assert_called_once_with(
            symbol="EURUSD", side=Side.SELL, volume=0.1, sl=1.1100, tp=1.0900,
        )

    @pytest.mark.asyncio
    async def test_hold_returns_none(self, mock_broker):
        om = OrderManager(mock_broker)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.HOLD, confidence=0,
            suggested_sl=0, suggested_tp=0,
            details=[],
        )
        result = await om.execute_signal(signal, volume=0.1, sl=0, tp=0)
        assert result is None
        mock_broker.open_trade.assert_not_called()
        assert om.get_history() == []

    @pytest.mark.asyncio
    async def test_broker_error(self, mock_broker):
        mock_broker.open_trade.side_effect = Exception("Connection lost")
        om = OrderManager(mock_broker)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=1.0950, suggested_tp=1.1100,
            details=["test"],
        )
        result = await om.execute_signal(signal, volume=0.1, sl=1.0950, tp=1.1100)
        assert result is None
        history = om.get_history()
        assert len(history) == 1
        assert history[0].status == "error"


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_delegates_to_broker(self, mock_broker):
        om = OrderManager(mock_broker)
        result = await om.close_position(trade_id=1)
        assert result is True
        mock_broker.close_trade.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, mock_broker):
        mock_broker.close_trade.return_value = False
        om = OrderManager(mock_broker)
        result = await om.close_position(trade_id=99)
        assert result is False


class TestGetHistory:
    @pytest.mark.asyncio
    async def test_returns_copy(self, mock_broker):
        om = OrderManager(mock_broker)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=1.0950, suggested_tp=1.1100,
            details=["test"],
        )
        await om.execute_signal(signal, volume=0.1, sl=1.0950, tp=1.1100)

        history = om.get_history()
        history.clear()
        # Internal state should not be affected
        assert len(om.get_history()) == 1


class TestGetStats:
    def test_empty_history(self, mock_broker):
        om = OrderManager(mock_broker)
        stats = om.get_stats()
        assert stats == {"total": 0}

    @pytest.mark.asyncio
    async def test_mixed_orders(self, mock_broker):
        om = OrderManager(mock_broker)

        buy_signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=1.0950, suggested_tp=1.1100,
            details=["test"],
        )

        # 3 successful + 1 error
        await om.execute_signal(buy_signal, volume=0.1, sl=1.0950, tp=1.1100)
        await om.execute_signal(buy_signal, volume=0.1, sl=1.0950, tp=1.1100)
        await om.execute_signal(buy_signal, volume=0.1, sl=1.0950, tp=1.1100)

        mock_broker.open_trade.side_effect = Exception("fail")
        await om.execute_signal(buy_signal, volume=0.1, sl=1.0950, tp=1.1100)

        stats = om.get_stats()
        assert stats["total"] == 4
        assert stats["opened"] == 3
        assert stats["errors"] == 1
