"""News sentiment analysis using RSS feeds and keyword-based scoring."""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from xml.etree import ElementTree

import aiohttp
from loguru import logger

from src.analysis.indicators import Signal


@dataclass
class NewsSignal:
    signal: Signal
    confidence: float
    headlines: list[str]
    detail: str


# Keyword dictionaries with weights (-1.0 to +1.0)
_BULLISH_KEYWORDS: dict[str, float] = {
    "surge": 0.8, "surges": 0.8, "rally": 0.7, "rallies": 0.7,
    "soar": 0.8, "soars": 0.8, "jump": 0.6, "jumps": 0.6,
    "gain": 0.5, "gains": 0.5, "rise": 0.4, "rises": 0.4,
    "bullish": 0.7, "upgrade": 0.6, "upgrades": 0.6,
    "beat": 0.5, "beats": 0.5, "exceed": 0.5, "exceeds": 0.5,
    "strong": 0.4, "positive": 0.4, "optimism": 0.5, "optimistic": 0.5,
    "growth": 0.4, "recover": 0.5, "recovery": 0.5,
    "breakout": 0.6, "outperform": 0.5, "buy": 0.3,
    "boom": 0.7, "record high": 0.7, "all-time high": 0.8,
    "upbeat": 0.4, "strength": 0.4,
}

_BEARISH_KEYWORDS: dict[str, float] = {
    "crash": -0.9, "crashes": -0.9, "plunge": -0.8, "plunges": -0.8,
    "drop": -0.5, "drops": -0.5, "fall": -0.4, "falls": -0.4,
    "decline": -0.5, "declines": -0.5, "sink": -0.6, "sinks": -0.6,
    "bearish": -0.7, "downgrade": -0.6, "downgrades": -0.6,
    "miss": -0.5, "misses": -0.5, "weak": -0.4, "weakness": -0.4,
    "recession": -0.7, "crisis": -0.8, "fear": -0.5, "fears": -0.5,
    "negative": -0.4, "pessimism": -0.5, "pessimistic": -0.5,
    "sell-off": -0.7, "selloff": -0.7, "underperform": -0.5,
    "loss": -0.4, "losses": -0.4, "risk": -0.3,
    "warning": -0.5, "warn": -0.5, "default": -0.7,
    "inflation": -0.3, "rate hike": -0.4, "tariff": -0.4, "tariffs": -0.4,
    "sanctions": -0.5, "shutdown": -0.5, "layoff": -0.4, "layoffs": -0.4,
}

# RSS feed sources for financial news
_RSS_FEEDS: dict[str, str] = {
    "investing_com": "https://www.investing.com/rss/news.rss",
    "reuters_markets": "https://news.google.com/rss/search?q=financial+markets&hl=en",
    "forex_factory": "https://news.google.com/rss/search?q=forex+currency&hl=en",
    "stocks": "https://news.google.com/rss/search?q=stock+market&hl=en",
}

# Symbol → search terms mapping for relevance filtering
_SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "EURUSD": ["eur", "euro", "ecb", "eurozone", "eur/usd", "eurusd"],
    "GBPUSD": ["gbp", "pound", "sterling", "boe", "bank of england", "gbp/usd"],
    "USDJPY": ["jpy", "yen", "boj", "japan", "usd/jpy"],
    "USDCHF": ["chf", "swiss", "franc", "snb", "usd/chf"],
    "AUDUSD": ["aud", "aussie", "rba", "australia", "aud/usd"],
    "NZDUSD": ["nzd", "kiwi", "rbnz", "new zealand", "nzd/usd"],
    "USDCAD": ["cad", "loonie", "boc", "canada", "usd/cad"],
    "AAPL": ["apple", "aapl", "iphone", "tim cook"],
    "MSFT": ["microsoft", "msft", "azure", "satya nadella"],
    "GOOGL": ["google", "alphabet", "googl"],
    "AMZN": ["amazon", "amzn", "aws"],
    "TSLA": ["tesla", "tsla", "elon musk"],
    "GOLD": ["gold", "xau", "precious metal", "bullion"],
    "BTC": ["bitcoin", "btc", "crypto"],
    "ETH": ["ethereum", "eth", "crypto"],
}

# Always-relevant keywords (apply to USD-paired forex and general market)
_GLOBAL_KEYWORDS = [
    "usd", "dollar", "fed", "federal reserve", "fomc", "interest rate",
    "inflation", "nonfarm", "payroll", "gdp", "cpi", "ppi",
]


class _HeadlineCache:
    """Simple time-based cache for fetched headlines."""

    def __init__(self):
        self._cache: dict[str, tuple[float, list[str]]] = {}  # feed_key → (timestamp, headlines)

    def get(self, key: str, max_age: int = 300) -> list[str] | None:
        if key in self._cache:
            ts, headlines = self._cache[key]
            if time.time() - ts < max_age:
                return headlines
        return None

    def set(self, key: str, headlines: list[str]) -> None:
        self._cache[key] = (time.time(), headlines)


_cache = _HeadlineCache()


async def _fetch_rss(url: str, timeout: int = 10) -> list[str]:
    """Fetch and parse RSS feed, return list of headline strings."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    logger.debug(f"RSS feed returned {resp.status}: {url}")
                    return []
                text = await resp.text()

        root = ElementTree.fromstring(text)

        headlines = []
        # Standard RSS 2.0 items
        for item in root.iter("item"):
            title = item.findtext("title", "")
            if title:
                headlines.append(title.strip())
        # Atom entries
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            title = entry.findtext("{http://www.w3.org/2005/Atom}title", "")
            if title:
                headlines.append(title.strip())

        return headlines
    except Exception as e:
        logger.debug(f"RSS fetch failed for {url}: {e}")
        return []


def _is_relevant(headline: str, symbol: str) -> bool:
    """Check if a headline is relevant to the given symbol."""
    lower = headline.lower()

    # Check symbol-specific keywords
    sym_keywords = _SYMBOL_KEYWORDS.get(symbol.upper(), [symbol.lower()])
    for kw in sym_keywords:
        if kw in lower:
            return True

    # For forex pairs containing USD, also match global keywords
    if "USD" in symbol.upper():
        for kw in _GLOBAL_KEYWORDS:
            if kw in lower:
                return True

    return False


def _score_headline(headline: str) -> float:
    """Score a single headline. Returns value between -1.0 and 1.0."""
    lower = headline.lower()
    score = 0.0
    matches = 0

    for keyword, weight in _BULLISH_KEYWORDS.items():
        if keyword in lower:
            score += weight
            matches += 1

    for keyword, weight in _BEARISH_KEYWORDS.items():
        if keyword in lower:
            score += weight  # weight is already negative
            matches += 1

    if matches == 0:
        return 0.0

    # Average the matched weights
    return max(-1.0, min(1.0, score / matches))


def _aggregate_scores(scores: list[float], threshold: float = 0.3) -> tuple[Signal, float]:
    """Aggregate headline scores into a signal and confidence.

    Args:
        scores: List of individual headline scores (-1 to 1).
        threshold: Minimum average score to generate a non-HOLD signal.

    Returns:
        (Signal, confidence 0-100)
    """
    if not scores:
        return Signal.HOLD, 0.0

    avg = sum(scores) / len(scores)

    if abs(avg) < threshold:
        return Signal.HOLD, abs(avg) / threshold * 30  # low confidence HOLD

    if avg > 0:
        signal = Signal.BUY
    else:
        signal = Signal.SELL

    # Scale confidence: threshold→30, 1.0→100
    raw_conf = 30 + (abs(avg) - threshold) / (1.0 - threshold) * 70
    confidence = min(100.0, max(0.0, raw_conf))

    return signal, confidence


async def analyze(symbol: str, config: dict) -> NewsSignal:
    """Fetch news headlines and compute sentiment signal for a symbol.

    Config keys:
        enabled: bool - must be True to run
        sources: list[str] - feed keys to use (default: all)
        sentiment_threshold: float - min score to trigger signal (default: 0.3)
        update_interval: int - cache TTL in seconds (default: 300)
    """
    if not config.get("enabled", False):
        return NewsSignal(Signal.HOLD, 0, [], "News analysis disabled")

    sources = config.get("sources", list(_RSS_FEEDS.keys()))
    threshold = config.get("sentiment_threshold", 0.3)
    cache_ttl = config.get("update_interval", 300)

    # Fetch headlines from configured sources
    all_headlines: list[str] = []

    # Also add a symbol-specific Google News search
    symbol_search_url = (
        f"https://news.google.com/rss/search?q={symbol}+trading&hl=en"
    )
    feeds_to_fetch: dict[str, str] = {}
    for src in sources:
        if src in _RSS_FEEDS:
            feeds_to_fetch[src] = _RSS_FEEDS[src]
    feeds_to_fetch[f"symbol_{symbol}"] = symbol_search_url

    tasks = []
    keys = []
    for key, url in feeds_to_fetch.items():
        cached = _cache.get(key, max_age=cache_ttl)
        if cached is not None:
            all_headlines.extend(cached)
        else:
            tasks.append(_fetch_rss(url))
            keys.append(key)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for key, result in zip(keys, results):
            if isinstance(result, list):
                _cache.set(key, result)
                all_headlines.extend(result)

    # Filter to relevant headlines
    relevant = [h for h in all_headlines if _is_relevant(h, symbol)]

    if not relevant:
        return NewsSignal(
            Signal.HOLD, 0, [],
            f"No relevant news found for {symbol} ({len(all_headlines)} total headlines)",
        )

    # Score each headline
    scores = [_score_headline(h) for h in relevant]

    signal, confidence = _aggregate_scores(scores, threshold)

    # Build detail string
    scored_headlines = sorted(zip(scores, relevant), key=lambda x: abs(x[0]), reverse=True)
    top_headlines = [h for _, h in scored_headlines[:5]]

    avg_score = sum(scores) / len(scores)
    detail = (
        f"{len(relevant)} relevant headlines, avg_score={avg_score:.2f}, "
        f"signal={signal.value}, conf={confidence:.0f}"
    )

    logger.info(f"[News] {symbol}: {detail}")

    return NewsSignal(
        signal=signal,
        confidence=confidence,
        headlines=top_headlines,
        detail=detail,
    )
