"""Tests for src/strategy/technical.py"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.analysis.indicators import Signal, IndicatorSignal
from src.analysis.support_resistance import SRSignal, Level
from src.analysis.patterns import PatternMatch
from src.strategy.technical import TechnicalStrategy
from src.strategy.base import TradeSignal
from tests.conftest import make_ohlcv_df


def _default_strategy_config():
    return {
        "weights": {
            "support_resistance": 0.35,
            "patterns": 0.30,
            "indicators": 0.25,
            "news": 0.10,
        },
        "support_resistance": {},
        "patterns": {},
        "indicators": {},
        "news": {"enabled": False},
    }


class TestEvaluateGuards:
    @pytest.mark.asyncio
    async def test_insufficient_data_returns_hold(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=30)
        result = await strategy.evaluate("EURUSD", df)
        assert result.signal == Signal.HOLD
        assert result.confidence == 0
        assert "Insufficient" in result.details[0]

    @pytest.mark.asyncio
    async def test_empty_df_returns_hold(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = await strategy.evaluate("EURUSD", df)
        assert result.signal == Signal.HOLD


class TestSignalToScore:
    def test_buy(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        assert strategy._signal_to_score(Signal.BUY) == 1.0

    def test_sell(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        assert strategy._signal_to_score(Signal.SELL) == -1.0

    def test_hold(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        assert strategy._signal_to_score(Signal.HOLD) == 0.0


class TestComputeSlTp:
    def test_buy_signal_with_atr(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200, trend="up")
        # Add a fake ATR column
        df["atr"] = 0.005
        price = df["close"].iloc[-1]

        sl, tp = strategy._compute_sl_tp(df, Signal.BUY, [])
        assert sl < price
        assert tp > price
        assert sl == pytest.approx(price - 0.005 * 1.5, abs=0.0001)
        assert tp == pytest.approx(price + 0.005 * 3.0, abs=0.0001)

    def test_sell_signal_with_atr(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200, trend="down")
        df["atr"] = 0.005
        price = df["close"].iloc[-1]

        sl, tp = strategy._compute_sl_tp(df, Signal.SELL, [])
        assert sl > price
        assert tp < price

    def test_hold_returns_zeros(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200)
        df["atr"] = 0.005
        sl, tp = strategy._compute_sl_tp(df, Signal.HOLD, [])
        assert sl == 0.0
        assert tp == 0.0

    def test_no_atr_returns_zeros(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200)
        # No "atr" column
        sl, tp = strategy._compute_sl_tp(df, Signal.BUY, [])
        assert sl == 0.0
        assert tp == 0.0

    def test_buy_sl_adjusted_by_support(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200, trend="up")
        df["atr"] = 0.005
        price = df["close"].iloc[-1]

        # Support level close below price → should pull SL lower
        support = Level(price=price - 0.003, strength=3, is_support=True)
        sl, tp = strategy._compute_sl_tp(df, Signal.BUY, [support])
        default_sl = price - 0.005 * 1.5
        # SL should be adjusted: min(default_sl, support - 0.2*ATR)
        adjusted = support.price - 0.005 * 0.2
        expected_sl = min(default_sl, adjusted)
        assert sl == pytest.approx(expected_sl, abs=0.0001)

    def test_sell_sl_adjusted_by_resistance(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200, trend="down")
        df["atr"] = 0.005
        price = df["close"].iloc[-1]

        resistance = Level(price=price + 0.003, strength=3, is_support=False)
        sl, tp = strategy._compute_sl_tp(df, Signal.SELL, [resistance])
        default_sl = price + 0.005 * 1.5
        adjusted = resistance.price + 0.005 * 0.2
        expected_sl = max(default_sl, adjusted)
        assert sl == pytest.approx(expected_sl, abs=0.0001)


class TestWeightedScoring:
    @pytest.mark.asyncio
    async def test_all_buy_signals(self):
        """When all sub-analyses return BUY with high confidence, final should be BUY."""
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200, trend="up")

        sr_result = SRSignal(Signal.BUY, 90, "Near support", [], [])
        pat_result = [PatternMatch("double_bottom", Signal.BUY, 80, 0, 10, 1.12, "DB")]
        ind_signals = [
            IndicatorSignal("RSI", Signal.BUY, 25, "oversold"),
            IndicatorSignal("MACD", Signal.BUY, 0.001, "crossover"),
            IndicatorSignal("MA", Signal.BUY, 1.10, "above"),
            IndicatorSignal("BB", Signal.BUY, 1.10, "lower band"),
            IndicatorSignal("Stochastic", Signal.BUY, 15, "oversold"),
        ]

        with patch("src.strategy.technical.support_resistance.analyze", return_value=sr_result), \
             patch("src.strategy.technical.patterns.analyze", return_value=pat_result), \
             patch("src.strategy.technical.indicators.compute_all", return_value=(df, ind_signals)):
            result = await strategy.evaluate("EURUSD", df)
            assert result.signal == Signal.BUY
            assert result.confidence > 50

    @pytest.mark.asyncio
    async def test_all_sell_signals(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200, trend="down")

        sr_result = SRSignal(Signal.SELL, 90, "Near resistance", [], [])
        pat_result = [PatternMatch("double_top", Signal.SELL, 80, 0, 10, 1.08, "DT")]
        ind_signals = [
            IndicatorSignal("RSI", Signal.SELL, 75, "overbought"),
            IndicatorSignal("MACD", Signal.SELL, -0.001, "crossover"),
            IndicatorSignal("MA", Signal.SELL, 1.10, "below"),
            IndicatorSignal("BB", Signal.SELL, 1.10, "upper band"),
            IndicatorSignal("Stochastic", Signal.SELL, 85, "overbought"),
        ]

        with patch("src.strategy.technical.support_resistance.analyze", return_value=sr_result), \
             patch("src.strategy.technical.patterns.analyze", return_value=pat_result), \
             patch("src.strategy.technical.indicators.compute_all", return_value=(df, ind_signals)):
            result = await strategy.evaluate("EURUSD", df)
            assert result.signal == Signal.SELL
            assert result.confidence > 50

    @pytest.mark.asyncio
    async def test_mixed_signals(self):
        """S/R buy + patterns sell + indicators mixed → result depends on weights."""
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200)

        sr_result = SRSignal(Signal.BUY, 80, "support", [], [])
        pat_result = [PatternMatch("double_top", Signal.SELL, 70, 0, 10, 1.08, "DT")]
        ind_signals = [
            IndicatorSignal("RSI", Signal.SELL, 75, "overbought"),
            IndicatorSignal("MACD", Signal.BUY, 0.001, "crossover"),
            IndicatorSignal("MA", Signal.HOLD, 1.10, "mixed"),
            IndicatorSignal("BB", Signal.HOLD, 1.10, "within"),
            IndicatorSignal("Stochastic", Signal.SELL, 85, "overbought"),
        ]

        with patch("src.strategy.technical.support_resistance.analyze", return_value=sr_result), \
             patch("src.strategy.technical.patterns.analyze", return_value=pat_result), \
             patch("src.strategy.technical.indicators.compute_all", return_value=(df, ind_signals)):
            result = await strategy.evaluate("EURUSD", df)
            # Just verify it produces a valid signal
            assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
            assert 0 <= result.confidence <= 100


class TestConfidenceNormalization:
    @pytest.mark.asyncio
    async def test_news_disabled_normalization(self):
        """With news disabled, active weight = 0.90, max_possible = 90."""
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200)

        sr_result = SRSignal(Signal.BUY, 100, "test", [], [])
        pat_result = [PatternMatch("db", Signal.BUY, 100, 0, 10, 1.12, "test")]
        ind_signals = [
            IndicatorSignal("RSI", Signal.BUY, 25, ""),
            IndicatorSignal("MACD", Signal.BUY, 0.001, ""),
            IndicatorSignal("MA", Signal.BUY, 1.10, ""),
            IndicatorSignal("BB", Signal.BUY, 1.10, ""),
            IndicatorSignal("Stochastic", Signal.BUY, 15, ""),
        ]

        with patch("src.strategy.technical.support_resistance.analyze", return_value=sr_result), \
             patch("src.strategy.technical.patterns.analyze", return_value=pat_result), \
             patch("src.strategy.technical.indicators.compute_all", return_value=(df, ind_signals)):
            result = await strategy.evaluate("EURUSD", df)
            # All BUY at 100% → confidence should be 100 (fully normalized)
            assert result.confidence == pytest.approx(100, abs=1)


class TestDetails:
    @pytest.mark.asyncio
    async def test_contains_all_sections(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200, trend="up")

        sr_result = SRSignal(Signal.HOLD, 0, "No interaction", [], [])
        with patch("src.strategy.technical.support_resistance.analyze", return_value=sr_result), \
             patch("src.strategy.technical.patterns.analyze", return_value=[]), \
             patch("src.strategy.technical.indicators.compute_all", return_value=(df, [])):
            result = await strategy.evaluate("EURUSD", df)
            detail_text = " ".join(result.details)
            assert "[S/R]" in detail_text
            assert "[Pattern]" in detail_text
            assert "[News]" in detail_text

    @pytest.mark.asyncio
    async def test_news_disabled_label(self):
        strategy = TechnicalStrategy(_default_strategy_config())
        df = make_ohlcv_df(length=200)

        sr_result = SRSignal(Signal.HOLD, 0, "test", [], [])
        with patch("src.strategy.technical.support_resistance.analyze", return_value=sr_result), \
             patch("src.strategy.technical.patterns.analyze", return_value=[]), \
             patch("src.strategy.technical.indicators.compute_all", return_value=(df, [])):
            result = await strategy.evaluate("EURUSD", df)
            assert any("Disabled" in d for d in result.details)
