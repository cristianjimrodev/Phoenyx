"""Tests for src/data/trade_store.py"""
from __future__ import annotations

import pytest

from src.data.trade_store import TradeStore
from src.orders.manager import OrderRecord


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_trades.db")
    ts = TradeStore(db_path=db_path)
    yield ts
    ts.close()


def _make_record(symbol: str = "EURUSD", status: str = "opened",
                 pnl: float = 0.0, side: str = "buy") -> OrderRecord:
    return OrderRecord(
        timestamp="2024-01-01T10:00:00",
        symbol=symbol,
        side=side,
        volume=0.1,
        entry_price=1.1000,
        sl=1.0950,
        tp=1.1100,
        confidence=80.0,
        reasons=["test reason", "another reason"],
        trade_id=1,
        status=status,
        exit_price=0.0,
        pnl=pnl,
    )


class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, store):
        record = _make_record()
        row_id = store.save_order(record)
        assert row_id > 0

        df = store.load_trades()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "EURUSD"
        assert df.iloc[0]["side"] == "buy"
        assert df.iloc[0]["status"] == "opened"

    def test_reasons_serialized(self, store):
        record = _make_record()
        store.save_order(record)
        df = store.load_trades()
        assert "test reason" in df.iloc[0]["reasons"]

    def test_load_empty_returns_empty(self, store):
        df = store.load_trades()
        assert df.empty

    def test_load_with_symbol_filter(self, store):
        store.save_order(_make_record(symbol="EURUSD"))
        store.save_order(_make_record(symbol="GBPUSD"))

        df_eur = store.load_trades(symbol="EURUSD")
        df_gbp = store.load_trades(symbol="GBPUSD")
        assert len(df_eur) == 1
        assert len(df_gbp) == 1

    def test_load_limit(self, store):
        for i in range(20):
            store.save_order(_make_record())
        df = store.load_trades(limit=5)
        assert len(df) == 5


class TestUpdate:
    def test_update_status(self, store):
        record = _make_record(status="opened")
        record.trade_id = 42
        store.save_order(record)

        store.update_order(trade_id=42, status="closed",
                           exit_price=1.1050, pnl=50.0)
        df = store.load_trades()
        assert df.iloc[0]["status"] == "closed"
        assert df.iloc[0]["exit_price"] == pytest.approx(1.1050)
        assert df.iloc[0]["pnl"] == pytest.approx(50.0)


class TestStats:
    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats == {"total": 0}

    def test_stats_with_trades(self, store):
        # 2 opened, 1 closed winning, 1 closed losing, 1 error
        store.save_order(_make_record(status="opened"))
        store.save_order(_make_record(status="opened"))

        rec_win = _make_record(status="closed", pnl=100.0)
        rec_win.trade_id = 10
        store.save_order(rec_win)

        rec_loss = _make_record(status="closed", pnl=-50.0)
        rec_loss.trade_id = 11
        store.save_order(rec_loss)

        store.save_order(_make_record(status="error"))

        stats = store.get_stats()
        assert stats["total"] == 5
        assert stats["opened"] == 2
        assert stats["closed"] == 2
        assert stats["errors"] == 1
        assert stats["winning"] == 1
        assert stats["losing"] == 1
        assert stats["win_rate"] == 50.0
        assert stats["total_pnl"] == pytest.approx(50.0)

    def test_stats_by_symbol(self, store):
        store.save_order(_make_record(symbol="EURUSD", status="closed", pnl=100))
        store.save_order(_make_record(symbol="GBPUSD", status="closed", pnl=-50))

        stats = store.get_stats(symbol="EURUSD")
        assert stats["total"] == 1
        assert stats["total_pnl"] == pytest.approx(100.0)
