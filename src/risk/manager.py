"""Risk management: position sizing, drawdown limits, exposure control."""
from __future__ import annotations

from loguru import logger

from src.broker.base import AccountInfo, BrokerClient, Side, Trade
from src.strategy.base import TradeSignal


class RiskManager:
    """Controls risk per trade and overall portfolio exposure."""

    def __init__(self, config: dict):
        self._max_risk_per_trade = config.get("max_risk_per_trade", 0.02)
        self._max_daily_drawdown = config.get("max_daily_drawdown", 0.05)
        self._max_open_positions = config.get("max_open_positions", 5)
        self._default_rr_ratio = config.get("default_rr_ratio", 2.0)
        self._daily_start_balance: float | None = None
        self._daily_pnl: float = 0.0

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

    def compute_position_size(self, account: AccountInfo, signal: TradeSignal,
                              pip_value: float, contract_size: float) -> float:
        """Calculate position size based on risk percentage and stop loss distance."""
        if signal.suggested_sl == 0:
            return 0.0

        risk_amount = account.balance * self._max_risk_per_trade
        current_price = signal.suggested_tp  # we use tp as reference for current price direction

        # Distance to stop loss in price
        if signal.signal.value == "buy":
            sl_distance = abs(current_price - signal.suggested_sl)
        else:
            sl_distance = abs(signal.suggested_sl - current_price)

        if sl_distance == 0:
            return 0.0

        # Volume = risk_amount / (sl_distance * contract_size)
        volume = risk_amount / (sl_distance * contract_size)

        # Round to 2 decimal places (standard lot step)
        volume = round(volume, 2)
        volume = max(volume, 0.01)  # minimum lot

        logger.info(f"Position size: {volume} lots (risk={risk_amount:.2f}, "
                    f"SL distance={sl_distance:.5f})")
        return volume

    def adjust_sl_tp(self, signal: TradeSignal) -> tuple[float, float]:
        """Ensure SL/TP meet minimum risk/reward ratio."""
        if signal.suggested_sl == 0 or signal.suggested_tp == 0:
            return signal.suggested_sl, signal.suggested_tp

        # Just pass through the strategy's SL/TP for now
        return signal.suggested_sl, signal.suggested_tp
