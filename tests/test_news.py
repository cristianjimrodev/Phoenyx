"""Tests for src/analysis/news.py"""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest

from src.analysis.news import (
    _score_headline,
    _is_relevant,
    _aggregate_scores,
    analyze,
    NewsSignal,
)
from src.analysis.indicators import Signal


class TestScoreHeadline:
    def test_bullish_headline(self):
        score = _score_headline("Stock market surges on strong earnings")
        assert score > 0

    def test_bearish_headline(self):
        score = _score_headline("Markets crash amid recession fears")
        assert score < 0

    def test_neutral_headline(self):
        score = _score_headline("Weather forecast for tomorrow")
        assert score == 0.0

    def test_mixed_headline(self):
        # Has both bullish and bearish words
        score = _score_headline("Markets rally despite fears of recession")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_score_capped(self):
        score = _score_headline("surge rally soar boom gains strong recovery")
        assert score <= 1.0
        score2 = _score_headline("crash plunge crash crisis fear losses")
        assert score2 >= -1.0


class TestIsRelevant:
    def test_eurusd_relevant(self):
        assert _is_relevant("ECB raises interest rates", "EURUSD") is True
        assert _is_relevant("Euro strengthens against dollar", "EURUSD") is True

    def test_eurusd_usd_keywords(self):
        assert _is_relevant("Fed announces rate decision", "EURUSD") is True

    def test_stock_relevant(self):
        assert _is_relevant("Apple reports record iPhone sales", "AAPL") is True

    def test_irrelevant(self):
        assert _is_relevant("Local weather update", "EURUSD") is False

    def test_case_insensitive(self):
        assert _is_relevant("EURO zone data", "EURUSD") is True


class TestAggregateScores:
    def test_empty_returns_hold(self):
        signal, conf = _aggregate_scores([])
        assert signal == Signal.HOLD
        assert conf == 0.0

    def test_strong_bullish(self):
        signal, conf = _aggregate_scores([0.8, 0.7, 0.6])
        assert signal == Signal.BUY
        assert conf > 50

    def test_strong_bearish(self):
        signal, conf = _aggregate_scores([-0.8, -0.7, -0.6])
        assert signal == Signal.SELL
        assert conf > 50

    def test_below_threshold_is_hold(self):
        signal, conf = _aggregate_scores([0.1, -0.1, 0.05], threshold=0.3)
        assert signal == Signal.HOLD

    def test_confidence_range(self):
        _, conf = _aggregate_scores([0.5, 0.6, 0.7])
        assert 0 <= conf <= 100


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_disabled_returns_hold(self):
        result = await analyze("EURUSD", {"enabled": False})
        assert result.signal == Signal.HOLD
        assert "disabled" in result.detail.lower()

    @pytest.mark.asyncio
    async def test_returns_news_signal(self):
        """With mocked RSS, should return a proper NewsSignal."""
        mock_headlines = [
            "Euro surges as ECB signals rate pause",
            "EUR/USD rallies on strong eurozone data",
            "Dollar weakens after Fed comments",
        ]

        with patch("src.analysis.news._fetch_rss", new_callable=AsyncMock,
                    return_value=mock_headlines):
            result = await analyze("EURUSD", {
                "enabled": True,
                "sources": ["investing_com"],
                "sentiment_threshold": 0.2,
                "update_interval": 0,  # disable cache
            })

        assert isinstance(result, NewsSignal)
        assert isinstance(result.signal, Signal)
        assert 0 <= result.confidence <= 100
        assert isinstance(result.headlines, list)

    @pytest.mark.asyncio
    async def test_no_relevant_headlines(self):
        mock_headlines = ["Weather in Tokyo is sunny today"]

        with patch("src.analysis.news._fetch_rss", new_callable=AsyncMock,
                    return_value=mock_headlines):
            result = await analyze("EURUSD", {
                "enabled": True,
                "sources": ["investing_com"],
                "update_interval": 0,
            })

        assert result.signal == Signal.HOLD
        assert "No relevant news" in result.detail

    @pytest.mark.asyncio
    async def test_empty_rss_returns_hold(self):
        with patch("src.analysis.news._fetch_rss", new_callable=AsyncMock,
                    return_value=[]):
            result = await analyze("EURUSD", {
                "enabled": True,
                "sources": ["investing_com"],
                "update_interval": 0,
            })

        assert result.signal == Signal.HOLD
