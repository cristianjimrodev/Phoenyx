"""Tests for src/risk/manager.py"""
from __future__ import annotations

import pytest

from src.broker.base import AccountInfo, Side, Trade, OrderStatus
from src.strategy.base import TradeSignal
from src.analysis.indicators import Signal
from src.risk.manager import RiskManager


def _make_trade(trade_id: int = 1) -> Trade:
    return Trade(
        trade_id=trade_id, symbol="EURUSD", side=Side.BUY,
        volume=0.1, open_price=1.1000, open_time=0,
    )


class TestCanTrade:
    def test_allows_when_under_limits(self, risk_config, mock_account):
        rm = RiskManager(risk_config)
        rm.set_daily_start_balance(mock_account.balance)
        open_trades = [_make_trade(i) for i in range(2)]
        ok, msg = rm.can_trade(mock_account, open_trades)
        assert ok is True
        assert msg == "OK"

    def test_blocks_max_positions(self, risk_config, mock_account):
        rm = RiskManager(risk_config)
        open_trades = [_make_trade(i) for i in range(5)]
        ok, msg = rm.can_trade(mock_account, open_trades)
        assert ok is False
        assert "Max open positions" in msg

    def test_blocks_daily_drawdown(self, risk_config):
        rm = RiskManager(risk_config)
        rm.set_daily_start_balance(10000)
        account = AccountInfo(
            balance=10000, equity=9400, margin=0,
            free_margin=9400, margin_level=0, currency="USD",
        )
        ok, msg = rm.can_trade(account, [])
        assert ok is False
        assert "drawdown" in msg.lower()

    def test_blocks_low_margin(self, risk_config):
        rm = RiskManager(risk_config)
        account = AccountInfo(
            balance=10000, equity=10000, margin=9500,
            free_margin=500, margin_level=0, currency="USD",
        )
        ok, msg = rm.can_trade(account, [])
        assert ok is False
        assert "margin" in msg.lower()

    def test_no_daily_balance_skips_drawdown_check(self, risk_config):
        rm = RiskManager(risk_config)
        # Do NOT call set_daily_start_balance
        account = AccountInfo(
            balance=10000, equity=5000, margin=0,
            free_margin=5000, margin_level=0, currency="USD",
        )
        ok, msg = rm.can_trade(account, [])
        # Drawdown check is skipped because _daily_start_balance is None
        assert ok is True

    def test_custom_max_positions(self):
        rm = RiskManager({"max_open_positions": 2})
        account = AccountInfo(
            balance=10000, equity=10000, margin=0,
            free_margin=10000, margin_level=0, currency="USD",
        )
        open_trades = [_make_trade(1), _make_trade(2)]
        ok, _ = rm.can_trade(account, open_trades)
        assert ok is False

    def test_exactly_at_drawdown_limit(self, risk_config):
        """Drawdown exactly at 5% should block."""
        rm = RiskManager(risk_config)
        rm.set_daily_start_balance(10000)
        account = AccountInfo(
            balance=10000, equity=9500, margin=0,
            free_margin=9500, margin_level=0, currency="USD",
        )
        ok, _ = rm.can_trade(account, [])
        assert ok is False


class TestComputePositionSize:
    def test_basic_calculation(self, risk_config, mock_account):
        rm = RiskManager(risk_config)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=1.0950, suggested_tp=1.1100,
            details=["test"],
        )
        # entry_price=1.1000, sl=1.0950 -> sl_distance=0.005
        # risk_amount = 10000 * 0.02 = 200
        # volume = 200 / (0.005 * 100000) = 0.40
        volume = rm.compute_position_size(
            mock_account, signal, pip_value=10, contract_size=100000,
            entry_price=1.1000,
        )
        assert volume == pytest.approx(0.40, abs=0.01)

    def test_zero_sl_returns_zero(self, risk_config, mock_account):
        rm = RiskManager(risk_config)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=0, suggested_tp=1.1100,
            details=["test"],
        )
        volume = rm.compute_position_size(
            mock_account, signal, pip_value=10, contract_size=100000,
            entry_price=1.1000,
        )
        assert volume == 0.0

    def test_zero_sl_distance_returns_zero(self, risk_config, mock_account):
        """When entry == SL, distance is 0."""
        rm = RiskManager(risk_config)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=1.1000, suggested_tp=1.1100,
            details=["test"],
        )
        volume = rm.compute_position_size(
            mock_account, signal, pip_value=10, contract_size=100000,
            entry_price=1.1000,
        )
        assert volume == 0.0

    def test_minimum_lot(self, risk_config, mock_account):
        """Very large SL distance should still return at least 0.01."""
        rm = RiskManager(risk_config)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=0.5000, suggested_tp=1.5000,
            details=["test"],
        )
        volume = rm.compute_position_size(
            mock_account, signal, pip_value=10, contract_size=100000,
            entry_price=1.1000,
        )
        assert volume >= 0.01

    def test_sell_signal(self, risk_config, mock_account):
        rm = RiskManager(risk_config)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.SELL, confidence=80,
            suggested_sl=1.1100, suggested_tp=1.0900,
            details=["test"],
        )
        volume = rm.compute_position_size(
            mock_account, signal, pip_value=10, contract_size=100000,
            entry_price=1.1000,
        )
        assert volume > 0


class TestAdjustSlTp:
    def test_passthrough(self, risk_config):
        rm = RiskManager(risk_config)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=1.0950, suggested_tp=1.1100,
            details=["test"],
        )
        sl, tp = rm.adjust_sl_tp(signal)
        assert sl == 1.0950
        assert tp == 1.1100

    def test_zero_values_passthrough(self, risk_config):
        rm = RiskManager(risk_config)
        signal = TradeSignal(
            symbol="EURUSD", signal=Signal.BUY, confidence=80,
            suggested_sl=0, suggested_tp=0,
            details=["test"],
        )
        sl, tp = rm.adjust_sl_tp(signal)
        assert sl == 0
        assert tp == 0


class TestSetDailyStartBalance:
    def test_resets_pnl(self, risk_config):
        rm = RiskManager(risk_config)
        rm._daily_pnl = 500.0
        rm.set_daily_start_balance(10000)
        assert rm._daily_pnl == 0.0
        assert rm._daily_start_balance == 10000
