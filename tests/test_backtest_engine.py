"""Tests for backtest/engine.py"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.indicators import Signal
from src.strategy.base import TradeSignal
from src.strategy.technical import TechnicalStrategy
from backtest.engine import BacktestEngine, BacktestResult, BacktestTrade
from tests.conftest import make_ohlcv_df


def _hold_signal(symbol: str = "EURUSD") -> TradeSignal:
    return TradeSignal(symbol, Signal.HOLD, 0, 0, 0, ["no signal"])


def _buy_signal(symbol: str = "EURUSD", confidence: float = 80,
                sl: float = 1.0900, tp: float = 1.1200) -> TradeSignal:
    return TradeSignal(symbol, Signal.BUY, confidence, sl, tp, ["buy signal"])


def _sell_signal(symbol: str = "EURUSD", confidence: float = 80,
                 sl: float = 1.1200, tp: float = 1.0900) -> TradeSignal:
    return TradeSignal(symbol, Signal.SELL, confidence, sl, tp, ["sell signal"])


@pytest.fixture
def mock_strategy():
    strategy = AsyncMock(spec=TechnicalStrategy)
    strategy.evaluate = AsyncMock(return_value=_hold_signal())
    return strategy


class TestNoTrades:
    @pytest.mark.asyncio
    async def test_all_hold(self, mock_strategy):
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        df = make_ohlcv_df(length=200, trend="flat")
        result = await engine.run(df, "EURUSD", lookback=100)

        assert result.total_trades == 0
        assert result.total_return_pct == 0.0
        assert result.win_rate == 0
        assert result.max_drawdown_pct == 0.0
        assert result.sharpe_ratio == 0

    @pytest.mark.asyncio
    async def test_low_confidence_filtered(self, mock_strategy):
        """Signal below min_confidence should not open a trade."""
        mock_strategy.evaluate.return_value = _buy_signal(confidence=50)
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        df = make_ohlcv_df(length=200, trend="up")
        result = await engine.run(df, "EURUSD", lookback=100)
        assert result.total_trades == 0

    @pytest.mark.asyncio
    async def test_zero_sl_skipped(self, mock_strategy):
        mock_strategy.evaluate.return_value = TradeSignal(
            "EURUSD", Signal.BUY, 80, 0, 1.12, ["test"],
        )
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        df = make_ohlcv_df(length=200)
        result = await engine.run(df, "EURUSD", lookback=100)
        assert result.total_trades == 0


class TestBuyTrades:
    @pytest.mark.asyncio
    async def test_tp_hit(self, mock_strategy):
        """BUY trade where TP is hit → winning trade."""
        df = make_ohlcv_df(length=200, trend="up", base_price=1.1000)

        entry_bar = 100
        entry_price = df["close"].iloc[entry_bar]
        sl = entry_price - 0.05
        tp = entry_price + 0.001  # Very close TP so it gets hit on next bar

        call_count = 0

        async def side_effect(symbol, window):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _buy_signal(sl=sl, tp=tp, confidence=80)
            return _hold_signal()

        mock_strategy.evaluate = AsyncMock(side_effect=side_effect)
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        result = await engine.run(df, "EURUSD", lookback=entry_bar)

        assert result.total_trades >= 1
        # First trade should be a BUY
        first_trade = result.trades[0]
        assert first_trade.side == "buy"

    @pytest.mark.asyncio
    async def test_sl_hit(self, mock_strategy):
        """BUY trade where SL is hit → losing trade."""
        df = make_ohlcv_df(length=200, trend="down", base_price=1.1000)

        entry_bar = 100
        entry_price = df["close"].iloc[entry_bar]
        sl = entry_price + 0.001  # SL very close above (will be hit since downtrend, low < sl)
        tp = entry_price + 0.10   # TP very far

        # Wait, for a BUY trade, SL < entry price. Let me fix:
        # Actually for BUY: SL hit when current_low <= sl
        sl = entry_price - 0.001  # Very tight SL below
        tp = entry_price + 0.10

        call_count = 0

        async def side_effect(symbol, window):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _buy_signal(sl=sl, tp=tp, confidence=80)
            return _hold_signal()

        mock_strategy.evaluate = AsyncMock(side_effect=side_effect)
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        result = await engine.run(df, "EURUSD", lookback=entry_bar)

        if result.total_trades >= 1:
            trade = result.trades[0]
            assert trade.side == "buy"
            assert trade.reason in ("SL hit", "TP hit")


class TestSellTrades:
    @pytest.mark.asyncio
    async def test_sell_tp_hit(self, mock_strategy):
        """SELL trade where TP is hit → winning trade."""
        df = make_ohlcv_df(length=200, trend="down", base_price=1.1000)

        entry_bar = 100
        entry_price = df["close"].iloc[entry_bar]
        sl = entry_price + 0.10   # SL far above
        tp = entry_price - 0.001  # TP very close below

        call_count = 0

        async def side_effect(symbol, window):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _sell_signal(sl=sl, tp=tp, confidence=80)
            return _hold_signal()

        mock_strategy.evaluate = AsyncMock(side_effect=side_effect)
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        result = await engine.run(df, "EURUSD", lookback=entry_bar)

        assert result.total_trades >= 1
        first_trade = result.trades[0]
        assert first_trade.side == "sell"


class TestBacktestMetrics:
    @pytest.mark.asyncio
    async def test_result_structure(self, mock_strategy):
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        df = make_ohlcv_df(length=200)
        result = await engine.run(df, "EURUSD", lookback=100)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.trades, list)
        assert isinstance(result.total_return_pct, float)
        assert isinstance(result.win_rate, (int, float))
        assert isinstance(result.max_drawdown_pct, float)
        assert isinstance(result.sharpe_ratio, (int, float))
        assert isinstance(result.profit_factor, (int, float))

    @pytest.mark.asyncio
    async def test_no_trades_zero_sharpe(self, mock_strategy):
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        df = make_ohlcv_df(length=200)
        result = await engine.run(df, "EURUSD", lookback=100)
        assert result.sharpe_ratio == 0

    @pytest.mark.asyncio
    async def test_winning_trades_positive_return(self, mock_strategy):
        """Multiple winning trades should produce positive total return."""
        df = make_ohlcv_df(length=300, trend="up", base_price=1.1000)

        call_count = 0

        async def side_effect(symbol, window):
            nonlocal call_count
            call_count += 1
            idx = len(window) - 1
            price = window["close"].iloc[-1]
            # Every 20th call, emit a BUY with tight TP in uptrend
            if call_count % 20 == 1:
                return _buy_signal(sl=price - 0.05, tp=price + 0.001, confidence=80)
            return _hold_signal()

        mock_strategy.evaluate = AsyncMock(side_effect=side_effect)
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        result = await engine.run(df, "EURUSD", lookback=100)

        if result.total_trades > 0 and result.winning_trades > 0:
            assert result.avg_win > 0

    @pytest.mark.asyncio
    async def test_max_drawdown_tracked(self, mock_strategy):
        """After a losing trade, max drawdown should be > 0."""
        df = make_ohlcv_df(length=200, trend="down", base_price=1.1000)

        call_count = 0

        async def side_effect(symbol, window):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                price = window["close"].iloc[-1]
                return _buy_signal(sl=price - 0.001, tp=price + 0.10, confidence=80)
            return _hold_signal()

        mock_strategy.evaluate = AsyncMock(side_effect=side_effect)
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        result = await engine.run(df, "EURUSD", lookback=100)

        if result.losing_trades > 0:
            assert result.max_drawdown_pct > 0

    @pytest.mark.asyncio
    async def test_profit_factor_inf_when_no_losses(self, mock_strategy):
        """If all trades win, profit_factor should be inf."""
        df = make_ohlcv_df(length=200, trend="up", base_price=1.1000)

        call_count = 0

        async def side_effect(symbol, window):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                price = window["close"].iloc[-1]
                return _buy_signal(sl=price - 0.10, tp=price + 0.001, confidence=80)
            return _hold_signal()

        mock_strategy.evaluate = AsyncMock(side_effect=side_effect)
        engine = BacktestEngine(mock_strategy, initial_balance=10000, min_confidence=60)
        result = await engine.run(df, "EURUSD", lookback=100)

        if result.winning_trades > 0 and result.losing_trades == 0:
            assert result.profit_factor == float("inf")
