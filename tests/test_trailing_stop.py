"""Tests for trailing stop logic in RiskManager."""
from __future__ import annotations

import pytest

from src.broker.base import Side, Trade, OrderStatus
from src.risk.manager import RiskManager


def _make_buy_trade(sl: float = 1.0950, open_price: float = 1.1000) -> Trade:
    return Trade(
        trade_id=1, symbol="EURUSD", side=Side.BUY,
        volume=0.1, open_price=open_price, open_time=0,
        sl=sl, tp=1.1200, status=OrderStatus.OPENED,
    )


def _make_sell_trade(sl: float = 1.1050, open_price: float = 1.1000) -> Trade:
    return Trade(
        trade_id=2, symbol="EURUSD", side=Side.SELL,
        volume=0.1, open_price=open_price, open_time=0,
        sl=sl, tp=1.0800, status=OrderStatus.OPENED,
    )


class TestTrailingStopEnabled:
    def test_enabled_property(self):
        rm = RiskManager({"trailing_stop": True})
        assert rm.trailing_stop_enabled is True

    def test_disabled_by_default(self):
        rm = RiskManager({})
        assert rm.trailing_stop_enabled is False


class TestComputeTrailingSl:
    def test_disabled_returns_none(self):
        rm = RiskManager({"trailing_stop": False})
        trade = _make_buy_trade()
        result = rm.compute_trailing_sl(trade, current_price=1.1100, atr=0.005)
        assert result is None

    def test_buy_move_sl_up(self):
        rm = RiskManager({"trailing_stop": True, "trailing_stop_distance_atr": 1.5})
        trade = _make_buy_trade(sl=1.0950, open_price=1.1000)
        # Price moved up to 1.1200 → new SL = 1.1200 - 1.5 * 0.005 = 1.1125
        new_sl = rm.compute_trailing_sl(trade, current_price=1.1200, atr=0.005)
        assert new_sl is not None
        assert new_sl == pytest.approx(1.1125, abs=0.0001)

    def test_buy_does_not_move_sl_down(self):
        rm = RiskManager({"trailing_stop": True, "trailing_stop_distance_atr": 1.5})
        trade = _make_buy_trade(sl=1.1100, open_price=1.1000)
        # Price at 1.1050 → new SL would be 1.0975 < current SL 1.1100 → None
        new_sl = rm.compute_trailing_sl(trade, current_price=1.1050, atr=0.005)
        assert new_sl is None

    def test_buy_sl_must_be_above_open(self):
        rm = RiskManager({"trailing_stop": True, "trailing_stop_distance_atr": 1.5})
        trade = _make_buy_trade(sl=1.0950, open_price=1.1000)
        # Price at 1.1010 → new SL = 1.1010 - 0.0075 = 1.0935 < open_price → None
        new_sl = rm.compute_trailing_sl(trade, current_price=1.1010, atr=0.005)
        assert new_sl is None

    def test_sell_move_sl_down(self):
        rm = RiskManager({"trailing_stop": True, "trailing_stop_distance_atr": 1.5})
        trade = _make_sell_trade(sl=1.1050, open_price=1.1000)
        # Price dropped to 1.0800 → new SL = 1.0800 + 0.0075 = 1.0875 < 1.1050
        new_sl = rm.compute_trailing_sl(trade, current_price=1.0800, atr=0.005)
        assert new_sl is not None
        assert new_sl == pytest.approx(1.0875, abs=0.0001)

    def test_sell_does_not_move_sl_up(self):
        rm = RiskManager({"trailing_stop": True, "trailing_stop_distance_atr": 1.5})
        trade = _make_sell_trade(sl=1.0900, open_price=1.1000)
        # Price at 1.0950 → new SL = 1.1025 > current SL → None
        new_sl = rm.compute_trailing_sl(trade, current_price=1.0950, atr=0.005)
        assert new_sl is None

    def test_zero_atr_returns_none(self):
        rm = RiskManager({"trailing_stop": True})
        trade = _make_buy_trade()
        result = rm.compute_trailing_sl(trade, current_price=1.1200, atr=0)
        assert result is None
