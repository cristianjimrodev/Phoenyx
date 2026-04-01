"""Tests for src/broker/paper.py"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.broker.base import (
    AccountInfo, BrokerClient, Side, Symbol, Trade, OrderStatus,
)
from src.broker.paper import PaperBroker
from tests.conftest import make_ohlcv_df


@pytest.fixture
def mock_data_broker():
    broker = AsyncMock(spec=BrokerClient)
    broker.connect.return_value = True
    broker.get_candles.return_value = make_ohlcv_df(length=10, base_price=1.1000)
    broker.get_symbol.return_value = Symbol(
        name="EURUSD", description="Euro vs USD", category="CASH",
        currency="USD", lot_min=0.01, lot_max=100, lot_step=0.01,
        pip_size=0.0001, contract_size=100000, leverage=100,
    )
    return broker


@pytest.fixture
def paper(mock_data_broker, tmp_path):
    state_file = str(tmp_path / "paper_state.json")
    return PaperBroker(mock_data_broker, initial_balance=10000, currency="USD",
                       state_file=state_file)


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_delegates(self, paper, mock_data_broker):
        result = await paper.connect()
        assert result is True
        mock_data_broker.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, paper, mock_data_broker):
        await paper.connect()
        await paper.disconnect()
        mock_data_broker.disconnect.assert_called_once()


class TestDataDelegation:
    @pytest.mark.asyncio
    async def test_get_symbol_delegates(self, paper, mock_data_broker):
        result = await paper.get_symbol("EURUSD")
        assert result.name == "EURUSD"
        mock_data_broker.get_symbol.assert_called_once_with("EURUSD")

    @pytest.mark.asyncio
    async def test_get_candles_delegates(self, paper, mock_data_broker):
        result = await paper.get_candles("EURUSD", "H1", 10)
        assert not result.empty
        mock_data_broker.get_candles.assert_called_once()


class TestOpenTrade:
    @pytest.mark.asyncio
    async def test_open_buy(self, paper):
        trade = await paper.open_trade("EURUSD", Side.BUY, 0.1, sl=1.0950, tp=1.1100)
        assert trade.trade_id == 1
        assert trade.side == Side.BUY
        assert trade.volume == 0.1
        assert trade.open_price > 0
        assert trade.status == OrderStatus.OPENED

    @pytest.mark.asyncio
    async def test_open_sell(self, paper):
        trade = await paper.open_trade("EURUSD", Side.SELL, 0.2, sl=1.1100, tp=1.0900)
        assert trade.side == Side.SELL
        assert trade.volume == 0.2

    @pytest.mark.asyncio
    async def test_sequential_trade_ids(self, paper):
        t1 = await paper.open_trade("EURUSD", Side.BUY, 0.1)
        t2 = await paper.open_trade("EURUSD", Side.SELL, 0.1)
        assert t2.trade_id == t1.trade_id + 1

    @pytest.mark.asyncio
    async def test_trade_shows_in_open_trades(self, paper):
        await paper.open_trade("EURUSD", Side.BUY, 0.1)
        trades = await paper.get_open_trades()
        assert len(trades) == 1


class TestCloseTrade:
    @pytest.mark.asyncio
    async def test_close_removes_from_open(self, paper):
        trade = await paper.open_trade("EURUSD", Side.BUY, 0.1)
        result = await paper.close_trade(trade.trade_id)
        assert result is True
        assert await paper.get_open_trades() == []

    @pytest.mark.asyncio
    async def test_close_nonexistent(self, paper):
        result = await paper.close_trade(999)
        assert result is False

    @pytest.mark.asyncio
    async def test_close_updates_balance(self, paper):
        initial = (await paper.get_account_info()).balance
        trade = await paper.open_trade("EURUSD", Side.BUY, 0.1)
        # Set a favorable price
        paper._latest_prices["EURUSD"] = trade.open_price + 0.01
        await paper.close_trade(trade.trade_id)
        after = (await paper.get_account_info()).balance
        assert after > initial


class TestModifyTrade:
    @pytest.mark.asyncio
    async def test_modify_sl_tp(self, paper):
        trade = await paper.open_trade("EURUSD", Side.BUY, 0.1, sl=1.09, tp=1.11)
        result = await paper.modify_trade(trade.trade_id, sl=1.095, tp=1.115)
        assert result is True
        trades = await paper.get_open_trades()
        assert trades[0].sl == 1.095
        assert trades[0].tp == 1.115

    @pytest.mark.asyncio
    async def test_modify_nonexistent(self, paper):
        result = await paper.modify_trade(999, sl=1.0)
        assert result is False


class TestAccountInfo:
    @pytest.mark.asyncio
    async def test_initial_balance(self, paper):
        info = await paper.get_account_info()
        assert info.balance == 10000
        assert info.equity == 10000
        assert info.currency == "USD"
        assert info.free_margin == 10000

    @pytest.mark.asyncio
    async def test_equity_changes_with_position(self, paper):
        trade = await paper.open_trade("EURUSD", Side.BUY, 0.1)
        # Simulate price increase
        paper._latest_prices["EURUSD"] = trade.open_price + 0.01
        info = await paper.get_account_info()
        assert info.equity > info.balance
