"""News sentiment analysis module (Phase 4 - placeholder)."""

from dataclasses import dataclass

from src.analysis.indicators import Signal


@dataclass
class NewsSignal:
    signal: Signal
    confidence: float
    headlines: list[str]
    detail: str


async def analyze(symbol: str, config: dict) -> NewsSignal:
    """Placeholder for news sentiment analysis. To be implemented in Phase 4."""
    return NewsSignal(
        signal=Signal.HOLD,
        confidence=0,
        headlines=[],
        detail="News analysis not yet enabled",
    )
