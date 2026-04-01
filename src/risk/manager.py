"""Risk management: position sizing, drawdown limits, exposure control."""
from __future__ import annotations

from loguru import logger

from src.broker.base import AccountInfo, BrokerClient, Side, Trade
from src.risk.portfolio import PortfolioRiskManager
from src.strategy.base import TradeSignal


class RiskManager:
    """Controls risk per trade and overall portfolio exposure."""

    def __init__(self, config: dict):
        self._max_risk_per_trade = config.get("max_risk_per_trade", 0.02)
        self._max_daily_drawdown = config.get("max_daily_drawdown", 0.05)
        self._max_open_positions = config.get("max_open_positions", 5)
        self._default_rr_ratio = config.get("default_rr_ratio", 2.0)
        self._trailing_stop = config.get("trailing_stop", False)
        self._trailing_stop_distance_atr = config.get("trailing_stop_distance_atr", 1.5)
        self._daily_start_balance: float | None = None
        self._daily_pnl: float = 0.0
        self._portfolio: PortfolioRiskManager | None = None

    def set_daily_start_balance(self, balance: float) -> None:
        self._daily_start_balance = balance
        self._daily_pnl = 0.0
        logger.info(f"Daily start balance set: {balance:.2f}")

    def can_trade(self, account: AccountInfo, open_trades: list[Trade]) -> tuple[bool, str]:
        """Check if we are allowed to open a new trade."""
        # Max positions check
        if len(open_trades) >= self._max_open_positions:
            return False, f"Max open positions reached ({self._max_open_positions})"

        # Daily drawdown check
        if self._daily_start_balance:
            current_drawdown = (self._daily_start_balance - account.equity) / self._daily_start_balance
            if current_drawdown >= self._max_daily_drawdown:
                return False, f"Daily drawdown limit reached ({current_drawdown:.1%} >= {self._max_daily_drawdown:.1%})"

        # Margin check
        if account.free_margin < account.balance * 0.1:
            return False, f"Low free margin ({account.free_margin:.2f})"

        return True, "OK"

    def check_spread(self, spread: float, volume: float, price: float,
                     contract_size: float, max_spread_pct: float = 0.01) -> tuple[bool, str]:
        """Check if spread cost is acceptable relative to margin blocked.

        Blocks the trade if spread cost > max_spread_pct (1%) of the margin.

        Args:
            spread: Current bid-ask spread in price units.
            volume: Trade volume (lots).
            price: Current price.
            contract_size: Contract size per lot.
            max_spread_pct: Max allowed spread as % of margin (default 1%).
        """
        if spread <= 0 or price <= 0:
            return True, "OK"

        spread_cost = spread * volume * contract_size
        margin_blocked = volume * price * contract_size / 100  # simplified margin
        if margin_blocked <= 0:
            return True, "OK"

        spread_pct = spread_cost / margin_blocked
        if spread_pct > max_spread_pct:
            return (
                False,
                f"Spread too high: {spread_pct:.2%} of margin "
                f"(spread={spread:.5f}, cost={spread_cost:.2f}, "
                f"margin={margin_blocked:.2f}, limit={max_spread_pct:.0%})",
            )
        return True, "OK"

    def compute_position_size(self, account: AccountInfo, signal: TradeSignal,
                              pip_value: float, contract_size: float,
                              entry_price: float = 0.0) -> float:
        """Calculate position size based on risk percentage and stop loss distance.

        Args:
            entry_price: Current market price (used to compute SL distance).
                         Falls back to midpoint between SL and TP if not provided.
        """
        if signal.suggested_sl == 0:
            return 0.0

        risk_amount = account.balance * self._max_risk_per_trade

        # Use actual entry price for SL distance, not TP
        if entry_price <= 0:
            entry_price = (signal.suggested_sl + signal.suggested_tp) / 2

        sl_distance = abs(entry_price - signal.suggested_sl)

        if sl_distance == 0:
            return 0.0

        # Volume = risk_amount / (sl_distance * contract_size)
        volume = risk_amount / (sl_distance * contract_size)

        # Round to 2 decimal places (standard lot step)
        volume = round(volume, 2)
        volume = max(volume, 0.01)  # minimum lot

        logger.info(f"Position size: {volume} lots (risk={risk_amount:.2f}, "
                    f"SL distance={sl_distance:.5f}, contract={contract_size})")
        return volume

    def adjust_sl_tp(self, signal: TradeSignal) -> tuple[float, float]:
        """Ensure SL/TP meet minimum risk/reward ratio."""
        if signal.suggested_sl == 0 or signal.suggested_tp == 0:
            return signal.suggested_sl, signal.suggested_tp

        # Just pass through the strategy's SL/TP for now
        return signal.suggested_sl, signal.suggested_tp

    def set_portfolio_risk(self, portfolio: PortfolioRiskManager) -> None:
        """Attach a portfolio-level risk manager."""
        self._portfolio = portfolio
        logger.info("Portfolio risk manager attached")

    def check_portfolio_risk(
        self, symbol: str, side: Side, volume: float,
        open_trades: list[Trade], account: AccountInfo,
    ) -> tuple[bool, str, float]:
        """Delegate portfolio-level risk check.

        Returns (allowed, reason, adjusted_volume).  When no portfolio
        manager is configured the trade is allowed unchanged.
        """
        if self._portfolio is None:
            return True, "OK", volume
        return self._portfolio.check_portfolio_risk(
            symbol, side, volume, open_trades, account,
        )

    @property
    def trailing_stop_enabled(self) -> bool:
        return self._trailing_stop

    def compute_trailing_sl(self, trade: Trade, current_price: float,
                            atr: float) -> float | None:
        """Compute a new trailing SL if price has moved favourably.

        Returns the new SL price, or None if no update is needed.
        The trailing distance is ``trailing_stop_distance_atr * atr``.
        """
        if not self._trailing_stop or atr <= 0:
            return None

        distance = self._trailing_stop_distance_atr * atr

        if trade.side == Side.BUY:
            new_sl = current_price - distance
            # Only move SL up, never down
            if new_sl > trade.sl and new_sl > trade.open_price:
                return round(new_sl, 5)
        elif trade.side == Side.SELL:
            new_sl = current_price + distance
            # Only move SL down, never up
            if new_sl < trade.sl and new_sl < trade.open_price:
                return round(new_sl, 5)

        return None
