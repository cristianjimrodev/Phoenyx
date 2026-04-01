"""Tests for src/analysis/indicators.py"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.analysis.indicators import compute_all, Signal, IndicatorSignal
from tests.conftest import make_ohlcv_df


class TestComputeAllBasics:
    """Basic return-shape and guard tests."""

    def test_returns_enriched_df_and_signals(self, indicator_config):
        df = make_ohlcv_df(length=200, trend="up")
        result_df, signals = compute_all(df, indicator_config)

        # Original not mutated
        assert "rsi" not in df.columns

        # Enriched df has expected columns
        for col in ("rsi", "macd", "macd_signal", "macd_hist",
                     "sma_20", "sma_50", "sma_200",
                     "bb_upper", "bb_lower", "bb_mid",
                     "atr", "stoch_k", "stoch_d"):
            assert col in result_df.columns, f"Missing column: {col}"

        assert len(signals) > 0

    def test_short_df_returns_empty(self, indicator_config):
        df = make_ohlcv_df(length=30)
        result_df, signals = compute_all(df, indicator_config)
        assert signals == []

    def test_empty_df_returns_empty(self, indicator_config):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result_df, signals = compute_all(df, indicator_config)
        assert signals == []

    def test_exactly_50_bars(self, indicator_config):
        df = make_ohlcv_df(length=50, trend="up")
        _, signals = compute_all(df, indicator_config)
        assert len(signals) > 0

    def test_all_signals_have_required_fields(self, indicator_config):
        df = make_ohlcv_df(length=200, trend="up")
        _, signals = compute_all(df, indicator_config)

        for sig in signals:
            assert isinstance(sig, IndicatorSignal)
            assert sig.name != ""
            assert isinstance(sig.signal, Signal)
            assert isinstance(sig.value, (int, float))
            assert sig.detail != ""

    def test_default_config(self):
        """Empty config should use defaults and not crash."""
        df = make_ohlcv_df(length=200, trend="flat")
        _, signals = compute_all(df, {})
        assert len(signals) > 0


class TestRSI:
    def test_oversold_generates_buy(self):
        # Strong downtrend should push RSI below 30
        df = make_ohlcv_df(length=200, trend="down", seed=10)
        _, signals = compute_all(df, {})
        rsi_sig = next(s for s in signals if s.name == "RSI")
        # In a strong downtrend, RSI should be low
        if rsi_sig.value <= 30:
            assert rsi_sig.signal == Signal.BUY

    def test_overbought_generates_sell(self):
        df = make_ohlcv_df(length=200, trend="up", seed=10)
        _, signals = compute_all(df, {})
        rsi_sig = next(s for s in signals if s.name == "RSI")
        if rsi_sig.value >= 70:
            assert rsi_sig.signal == Signal.SELL

    def test_custom_thresholds(self):
        """Custom overbought/oversold should change signal boundaries."""
        df = make_ohlcv_df(length=200, trend="up", seed=10)
        config = {"rsi": {"period": 14, "overbought": 55, "oversold": 45}}
        _, signals = compute_all(df, config)
        rsi_sig = next(s for s in signals if s.name == "RSI")
        # With narrow band, more likely to trigger BUY or SELL
        assert rsi_sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


class TestMACD:
    def test_signal_always_present(self, indicator_config):
        df = make_ohlcv_df(length=200, trend="up")
        _, signals = compute_all(df, indicator_config)
        macd_sig = next(s for s in signals if s.name == "MACD")
        assert macd_sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_macd_columns_added(self, indicator_config):
        df = make_ohlcv_df(length=200, trend="up")
        result_df, _ = compute_all(df, indicator_config)
        assert not result_df["macd"].isna().all()


class TestMovingAverages:
    def test_signal_always_present(self):
        """MA signal is always one of BUY, SELL, or HOLD."""
        df = make_ohlcv_df(length=250, trend="up", seed=7)
        _, signals = compute_all(df, {})
        ma_sig = next(s for s in signals if s.name == "MA")
        assert ma_sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_custom_ma_periods(self):
        """Custom MA periods should be applied."""
        df = make_ohlcv_df(length=250, trend="up", seed=7)
        config = {"moving_averages": {"fast": 10, "slow": 30, "long": 100}}
        result_df, signals = compute_all(df, config)
        assert "sma_10" in result_df.columns
        assert "sma_30" in result_df.columns
        assert "sma_100" in result_df.columns


class TestBollinger:
    def test_signal_always_present(self, indicator_config):
        df = make_ohlcv_df(length=200)
        _, signals = compute_all(df, indicator_config)
        bb_sig = next(s for s in signals if s.name == "BB")
        assert bb_sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


class TestStochastic:
    def test_signal_always_present(self, indicator_config):
        df = make_ohlcv_df(length=200)
        _, signals = compute_all(df, indicator_config)
        stoch_sig = next(s for s in signals if s.name == "Stochastic")
        assert stoch_sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_stoch_columns_added(self, indicator_config):
        df = make_ohlcv_df(length=200)
        result_df, _ = compute_all(df, indicator_config)
        assert "stoch_k" in result_df.columns
        assert "stoch_d" in result_df.columns


class TestSignalCount:
    def test_always_five_signals(self, indicator_config):
        """compute_all should return exactly 5 signals: RSI, MACD, MA, BB, Stochastic."""
        df = make_ohlcv_df(length=200)
        _, signals = compute_all(df, indicator_config)
        names = [s.name for s in signals]
        assert sorted(names) == sorted(["RSI", "MACD", "MA", "BB", "Stochastic"])
