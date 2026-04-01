"""Tests for multi-timeframe analysis feature."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pandas as pd
import pytest

from src.analysis.indicators import Signal, IndicatorSignal
from src.analysis.support_resistance import SRSignal, Level
from src.analysis.patterns import PatternMatch
from src.strategy.technical import TechnicalStrategy
from src.strategy.base import Strategy, TradeSignal
from src.data.feed import DataFeed
from tests.conftest import make_ohlcv_df


def _mtf_strategy_config():
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
        "multi_timeframe": {
            "enabled": True,
            "weights": {"H1": 0.50, "H4": 0.30, "D1": 0.20},
            "min_htf_agreement": 1,
        },
    }


def _make_buy_signal(symbol="EURUSD", confidence=80):
    return TradeSignal(symbol, Signal.BUY, confidence, 1.0900, 1.1200, ["BUY signal"])


def _make_sell_signal(symbol="EURUSD", confidence=70):
    return TradeSignal(symbol, Signal.SELL, confidence, 1.1200, 1.0900, ["SELL signal"])


def _make_hold_signal(symbol="EURUSD", confidence=0):
    return TradeSignal(symbol, Signal.HOLD, confidence, 0, 0, ["HOLD signal"])


class TestEvaluateMtfAllAgreeBuy:
    """All timeframes agree BUY -> should return BUY with boosted confidence."""

    @pytest.mark.asyncio
    async def test_all_agree_buy(self):
        strategy = TechnicalStrategy(_mtf_strategy_config())

        buy_h1 = _make_buy_signal(confidence=80)
        buy_h4 = _make_buy_signal(confidence=70)
        buy_d1 = _make_buy_signal(confidence=60)

        with patch.object(strategy, "evaluate", new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = [buy_h1, buy_h4, buy_d1]

            dataframes = {
                "H1": make_ohlcv_df(length=200),
                "H4": make_ohlcv_df(length=200),
                "D1": make_ohlcv_df(length=200),
            }
            result = await strategy.evaluate_mtf("EURUSD", dataframes)

        assert result.signal == Signal.BUY
        assert result.confidence > 0
        # Weighted confidence: (80*0.5 + 70*0.3 + 60*0.2) / (0.5+0.3+0.2) = 73
        assert result.confidence == pytest.approx(73.0, abs=1)
        assert any("[MTF]" in d for d in result.details)


class TestEvaluateMtfHtfDisagree:
    """Higher timeframes disagree with H1 -> should be filtered to HOLD."""

    @pytest.mark.asyncio
    async def test_htf_disagree_filtered_to_hold(self):
        strategy = TechnicalStrategy(_mtf_strategy_config())

        buy_h1 = _make_buy_signal(confidence=80)
        sell_h4 = _make_sell_signal(confidence=70)
        sell_d1 = _make_sell_signal(confidence=60)

        with patch.object(strategy, "evaluate", new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = [buy_h1, sell_h4, sell_d1]

            dataframes = {
                "H1": make_ohlcv_df(length=200),
                "H4": make_ohlcv_df(length=200),
                "D1": make_ohlcv_df(length=200),
            }
            result = await strategy.evaluate_mtf("EURUSD", dataframes)

        # Primary is BUY but 0 HTFs agree (both SELL), min_htf_agreement=1 -> filtered to HOLD
        assert result.signal == Signal.HOLD
        assert result.confidence == 0
        assert any("Filtered" in d for d in result.details)


class TestEvaluateMtfPrimaryHold:
    """Primary timeframe is HOLD -> should stay HOLD regardless of HTFs."""

    @pytest.mark.asyncio
    async def test_primary_hold_stays_hold(self):
        strategy = TechnicalStrategy(_mtf_strategy_config())

        hold_h1 = _make_hold_signal()
        buy_h4 = _make_buy_signal(confidence=90)
        buy_d1 = _make_buy_signal(confidence=85)

        with patch.object(strategy, "evaluate", new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = [hold_h1, buy_h4, buy_d1]

            dataframes = {
                "H1": make_ohlcv_df(length=200),
                "H4": make_ohlcv_df(length=200),
                "D1": make_ohlcv_df(length=200),
            }
            result = await strategy.evaluate_mtf("EURUSD", dataframes)

        assert result.signal == Signal.HOLD
        assert result.confidence == 0


class TestEvaluateMtfEmptyDataframes:
    """Empty dataframes dict -> returns HOLD."""

    @pytest.mark.asyncio
    async def test_empty_dataframes(self):
        strategy = TechnicalStrategy(_mtf_strategy_config())
        result = await strategy.evaluate_mtf("EURUSD", {})
        assert result.signal == Signal.HOLD
        assert "No timeframe data" in result.details[0]


class TestEvaluateMtfPartialAgreement:
    """One HTF agrees, one is HOLD -> passes min_htf_agreement=1."""

    @pytest.mark.asyncio
    async def test_partial_agreement_passes(self):
        strategy = TechnicalStrategy(_mtf_strategy_config())

        buy_h1 = _make_buy_signal(confidence=80)
        buy_h4 = _make_buy_signal(confidence=70)
        hold_d1 = _make_hold_signal(confidence=30)

        with patch.object(strategy, "evaluate", new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = [buy_h1, buy_h4, hold_d1]

            dataframes = {
                "H1": make_ohlcv_df(length=200),
                "H4": make_ohlcv_df(length=200),
                "D1": make_ohlcv_df(length=200),
            }
            result = await strategy.evaluate_mtf("EURUSD", dataframes)

        # 1 HTF agrees (H4), min_htf_agreement=1 -> passes
        assert result.signal == Signal.BUY
        # Weighted: (80*0.5 + 70*0.3 + 30*0.2*0.3) / 1.0 = 40 + 21 + 1.8 = 62.8
        assert result.confidence > 0


class TestGetMultiTimeframeCandles:
    """DataFeed.get_multi_timeframe_candles returns correct dict."""

    @pytest.mark.asyncio
    async def test_returns_correct_dict(self):
        mock_broker = AsyncMock()
        mock_store = MagicMock()

        feed = DataFeed(mock_broker, mock_store)

        df_h1 = make_ohlcv_df(length=100, seed=1)
        df_h4 = make_ohlcv_df(length=100, seed=2)
        df_d1 = make_ohlcv_df(length=100, seed=3)

        with patch.object(feed, "get_candles", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [df_h1, df_h4, df_d1]

            result = await feed.get_multi_timeframe_candles("EURUSD", ["H1", "H4", "D1"], 100)

        assert list(result.keys()) == ["H1", "H4", "D1"]
        assert len(result["H1"]) == 100
        assert len(result["H4"]) == 100
        assert len(result["D1"]) == 100
        assert mock_get.call_count == 3

    @pytest.mark.asyncio
    async def test_single_timeframe(self):
        mock_broker = AsyncMock()
        mock_store = MagicMock()
        feed = DataFeed(mock_broker, mock_store)

        df_h1 = make_ohlcv_df(length=50, seed=1)

        with patch.object(feed, "get_candles", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = df_h1
            result = await feed.get_multi_timeframe_candles("EURUSD", ["H1"], 50)

        assert list(result.keys()) == ["H1"]
        assert len(result["H1"]) == 50


class TestDefaultStrategyEvaluateMtf:
    """Strategy base class evaluate_mtf uses first dataframe."""

    @pytest.mark.asyncio
    async def test_default_uses_first_df(self):
        class ConcreteStrategy(Strategy):
            async def evaluate(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
                # Return a signal based on the length of the dataframe as a marker
                return TradeSignal(symbol, Signal.BUY, float(len(df)), 0, 0, ["from evaluate"])

        strategy = ConcreteStrategy()
        df_h1 = make_ohlcv_df(length=100)
        df_h4 = make_ohlcv_df(length=200)

        result = await strategy.evaluate_mtf("EURUSD", {"H1": df_h1, "H4": df_h4})

        # Should use the first dataframe (H1 with length 100)
        assert result.signal == Signal.BUY
        assert result.confidence == 100.0  # len(df_h1) = 100

    @pytest.mark.asyncio
    async def test_default_empty_dataframes_returns_hold(self):
        class ConcreteStrategy(Strategy):
            async def evaluate(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
                return TradeSignal(symbol, Signal.BUY, 50, 0, 0, ["test"])

        strategy = ConcreteStrategy()
        result = await strategy.evaluate_mtf("EURUSD", {})

        assert result.signal == Signal.HOLD
        assert "No data" in result.details[0]


class TestEvaluateMtfWeightedConfidence:
    """Verify the weighted confidence calculation in detail."""

    @pytest.mark.asyncio
    async def test_weighted_confidence_all_sell(self):
        strategy = TechnicalStrategy(_mtf_strategy_config())

        sell_h1 = _make_sell_signal(confidence=90)
        sell_h4 = _make_sell_signal(confidence=60)
        sell_d1 = _make_sell_signal(confidence=40)

        with patch.object(strategy, "evaluate", new_callable=AsyncMock) as mock_eval:
            mock_eval.side_effect = [sell_h1, sell_h4, sell_d1]

            dataframes = {
                "H1": make_ohlcv_df(length=200),
                "H4": make_ohlcv_df(length=200),
                "D1": make_ohlcv_df(length=200),
            }
            result = await strategy.evaluate_mtf("EURUSD", dataframes)

        assert result.signal == Signal.SELL
        # Weighted: (90*0.5 + 60*0.3 + 40*0.2) / 1.0 = 45 + 18 + 8 = 71
        assert result.confidence == pytest.approx(71.0, abs=1)
