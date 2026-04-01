"""Portfolio-level risk management."""
from __future__ import annotations

from loguru import logger

from src.broker.base import AccountInfo, Side, Trade
from src.risk.correlation import CorrelationMatrix


class PortfolioRiskManager:
    """Manages portfolio-level exposure limits based on correlation."""

    def __init__(self, config: dict, correlation: CorrelationMatrix):
        self._correlation = correlation
        self._max_correlated_exposure = config.get("max_correlated_exposure", 0.04)
        self._correlation_threshold = config.get("correlation_threshold", 0.70)
        self._max_sector_positions = config.get("max_sector_positions", 3)

    def check_portfolio_risk(self, symbol: str, side: Side,
                             proposed_volume: float,
                             open_trades: list[Trade],
                             account: AccountInfo) -> tuple[bool, str, float]:
        """Check if a new trade respects portfolio risk limits.

        Returns: (allowed, reason, adjusted_volume)
        """
        # Check max positions in correlated group
        correlated = self._correlation.get_correlated_symbols(
            symbol, self._correlation_threshold
        )
        correlated_symbols = {sym for sym, _ in correlated}

        correlated_trades = [
            t for t in open_trades
            if t.symbol in correlated_symbols or t.symbol == symbol
        ]

        if len(correlated_trades) >= self._max_sector_positions:
            return (
                False,
                f"Max correlated positions ({self._max_sector_positions}) reached "
                f"for {symbol} group: {[t.symbol for t in correlated_trades]}",
                0.0,
            )

        # Check combined exposure
        max_exposure = account.balance * self._max_correlated_exposure
        current_exposure = sum(
            t.volume * abs(t.open_price) for t in correlated_trades
        )
        proposed_exposure = proposed_volume * account.equity / 100  # simplified

        if current_exposure + proposed_exposure > max_exposure:
            # Reduce volume to fit within limits
            remaining = max(0, max_exposure - current_exposure)
            if remaining <= 0:
                return (
                    False,
                    f"Correlated exposure limit ({self._max_correlated_exposure:.0%}) "
                    f"exceeded for {symbol} group",
                    0.0,
                )
            adjusted = remaining / (account.equity / 100) if account.equity > 0 else 0
            adjusted = round(max(0.01, adjusted), 2)
            logger.info(
                f"[Portfolio] Volume reduced {proposed_volume:.2f} → {adjusted:.2f} "
                f"for {symbol} due to correlated exposure"
            )
            return True, "Volume adjusted for correlation", adjusted

        return True, "OK", proposed_volume
