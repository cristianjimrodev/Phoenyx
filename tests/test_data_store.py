"""Tests for src/data/store.py"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.store import DataStore
from tests.conftest import make_ohlcv_df


@pytest.fixture
def store(tmp_path):
    """Create a DataStore with a temporary database."""
    db_path = str(tmp_path / "test_candles.db")
    ds = DataStore(db_path=db_path)
    yield ds
    ds.close()


def _make_candle_df(n: int, start_ts: int = 1000000, period_ms: int = 3600000) -> pd.DataFrame:
    """Create a simple candle DataFrame with millisecond timestamps (matching store expectations)."""
    import numpy as np
    rng = np.random.default_rng(42)
    timestamps = [start_ts + i * period_ms for i in range(n)]
    close = 1.1000 + np.cumsum(rng.normal(0, 0.001, n))
    open_ = np.roll(close, 1)
    open_[0] = 1.1000
    high = np.maximum(open_, close) + 0.0005
    low = np.minimum(open_, close) - 0.0005
    volume = rng.uniform(1000, 5000, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestSaveAndLoad:
    def test_roundtrip(self, store):
        df = _make_candle_df(10)
        store.save_candles("EURUSD", 3600, df)
        loaded = store.load_candles("EURUSD", 3600, limit=500)

        assert len(loaded) == 10
        assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        # Verify values match
        for col in ("open", "high", "low", "close"):
            assert loaded[col].iloc[0] == pytest.approx(df[col].iloc[0], abs=1e-6)

    def test_load_empty_returns_empty(self, store):
        loaded = store.load_candles("GBPUSD", 3600, limit=500)
        assert loaded.empty

    def test_save_empty_df_is_noop(self, store):
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        store.save_candles("EURUSD", 3600, empty)
        loaded = store.load_candles("EURUSD", 3600)
        assert loaded.empty

    def test_sorted_by_timestamp(self, store):
        df = _make_candle_df(20)
        store.save_candles("EURUSD", 3600, df)
        loaded = store.load_candles("EURUSD", 3600)
        timestamps = loaded["timestamp"].tolist()
        assert timestamps == sorted(timestamps)


class TestUpsert:
    def test_upsert_updates_values(self, store):
        df = _make_candle_df(5)
        store.save_candles("EURUSD", 3600, df)

        # Modify close prices and re-save (same timestamps)
        df2 = df.copy()
        df2["close"] = df2["close"] + 0.01
        store.save_candles("EURUSD", 3600, df2)

        loaded = store.load_candles("EURUSD", 3600)
        assert len(loaded) == 5
        assert loaded["close"].iloc[0] == pytest.approx(df2["close"].iloc[0], abs=1e-6)


class TestLimit:
    def test_load_respects_limit(self, store):
        df = _make_candle_df(100)
        store.save_candles("EURUSD", 3600, df)
        loaded = store.load_candles("EURUSD", 3600, limit=20)
        assert len(loaded) == 20

    def test_limit_returns_most_recent(self, store):
        df = _make_candle_df(100)
        store.save_candles("EURUSD", 3600, df)
        loaded = store.load_candles("EURUSD", 3600, limit=10)
        # The last timestamp in loaded should be the last in the full dataset
        all_loaded = store.load_candles("EURUSD", 3600, limit=500)
        assert loaded["timestamp"].iloc[-1] == all_loaded["timestamp"].iloc[-1]


class TestIsolation:
    def test_multiple_symbols_isolated(self, store):
        df1 = _make_candle_df(10, start_ts=1000000)
        df2 = _make_candle_df(5, start_ts=2000000)
        store.save_candles("EURUSD", 3600, df1)
        store.save_candles("GBPUSD", 3600, df2)

        loaded_eur = store.load_candles("EURUSD", 3600)
        loaded_gbp = store.load_candles("GBPUSD", 3600)
        assert len(loaded_eur) == 10
        assert len(loaded_gbp) == 5

    def test_multiple_periods_isolated(self, store):
        df1 = _make_candle_df(10, start_ts=1000000)
        df2 = _make_candle_df(7, start_ts=2000000)
        store.save_candles("EURUSD", 60, df1)
        store.save_candles("EURUSD", 3600, df2)

        loaded_m1 = store.load_candles("EURUSD", 60)
        loaded_h1 = store.load_candles("EURUSD", 3600)
        assert len(loaded_m1) == 10
        assert len(loaded_h1) == 7


class TestDatetimeIndex:
    def test_loaded_has_datetime_index(self, store):
        df = _make_candle_df(5)
        store.save_candles("EURUSD", 3600, df)
        loaded = store.load_candles("EURUSD", 3600)
        assert loaded.index.name == "datetime"
        assert pd.api.types.is_datetime64_any_dtype(loaded.index)
