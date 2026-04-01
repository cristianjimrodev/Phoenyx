"""Paper trading broker — simulates order execution using live price data.

Wraps a real broker connection for market data but executes trades
in a simulated account, so no real capital is at risk.
State is persisted to a JSON file between runs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

import pandas as pd

from src.broker.base import (
    AccountInfo, BrokerClient, PriceCallback, Side, Symbol,
    TickPrice, Trade, OrderStatus,
)


class PaperBroker(BrokerClient):
    """Simulated broker that uses a real broker for data but keeps a virtual account.

    Args:
        data_broker: A real BrokerClient used only for market data
                     (get_candles, subscribe_prices, get_symbol, etc.).
        initial_balance: Starting virtual balance.
        currency: Account currency.
    """

    def __init__(self, data_broker: BrokerClient,
                 initial_balance: float = 10000.0,
                 currency: str = "USD",
                 state_file: str = "data/paper_state.json"):
        self._data_broker = data_broker
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._equity = initial_balance
        self._currency = currency
        self._trades: dict[int, Trade] = {}
        self._next_trade_id = 1
        self._connected = False
        self._latest_prices: dict[str, float] = {}
        self._price_callbacks: dict[str, PriceCallback] = {}
        self._state_file = Path(state_file)

        # Load persisted state if exists
        self._load_state()

    async def connect(self) -> bool:
        """Connect the underlying data broker."""
        result = await self._data_broker.connect()
        if result:
            self._connected = True
            logger.info(
                f"Paper broker connected (virtual balance: "
                f"{self._balance:.2f} {self._currency})"
            )
        return result

    async def disconnect(self) -> None:
        await self._data_broker.disconnect()
        self._connected = False
        logger.info("Paper broker disconnected")

    async def get_symbols(self) -> list[Symbol]:
        return await self._data_broker.get_symbols()

    async def get_symbol(self, symbol: str) -> Symbol:
        return await self._data_broker.get_symbol(symbol)

    async def get_candles(self, symbol: str, period: int | str,
                          count: int) -> pd.DataFrame:
        return await self._data_broker.get_candles(symbol, period, count)

    async def subscribe_prices(self, symbol: str,
                               callback: PriceCallback) -> None:
        self._price_callbacks[symbol] = callback

        async def _on_tick(tick: TickPrice) -> None:
            # Track latest price for PnL calculation
            mid = (tick.ask + tick.bid) / 2 if tick.ask and tick.bid else tick.bid or tick.ask
            self._latest_prices[symbol] = mid
            self._update_equity()

            # Forward to external callback
            await callback(tick)

        await self._data_broker.subscribe_prices(symbol, _on_tick)

    async def unsubscribe_prices(self, symbol: str) -> None:
        await self._data_broker.unsubscribe_prices(symbol)
        self._price_callbacks.pop(symbol, None)

    async def open_trade(self, symbol: str, side: Side, volume: float,
                         sl: float = 0, tp: float = 0) -> Trade:
        """Simulate opening a trade at the current market price."""
        # Get current price from latest tick or fetch candles
        price = self._latest_prices.get(symbol)
        if price is None:
            df = await self._data_broker.get_candles(symbol, "M1", 1)
            if not df.empty:
                price = df["close"].iloc[-1]
            else:
                raise ValueError(f"No price data available for {symbol}")

        # Get real contract_size from broker
        try:
            sym_info = await self._data_broker.get_symbol(symbol)
            contract_size = sym_info.contract_size
        except Exception:
            contract_size = 100000

        trade_id = self._next_trade_id
        self._next_trade_id += 1

        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            volume=volume,
            open_price=price,
            open_time=int(time.time() * 1000),
            sl=sl,
            tp=tp,
            profit=0.0,
            status=OrderStatus.OPENED,
            contract_size=contract_size,
        )
        self._trades[trade_id] = trade

        logger.info(
            f"[PAPER] Opened {side.value} {volume} {symbol} @ {price:.5f} "
            f"(SL={sl:.5f}, TP={tp:.5f}) trade_id={trade_id} "
            f"contract_size={contract_size}"
        )
        return trade

    async def close_trade(self, trade_id: int) -> bool:
        """Simulate closing a trade at the current market price."""
        trade = self._trades.get(trade_id)
        if not trade:
            logger.warning(f"[PAPER] Trade {trade_id} not found")
            return False

        close_price = self._latest_prices.get(trade.symbol, trade.open_price)

        if trade.side == Side.BUY:
            pnl = (close_price - trade.open_price) * trade.volume * trade.contract_size
        else:
            pnl = (trade.open_price - close_price) * trade.volume * trade.contract_size

        self._balance += pnl
        trade.close_price = close_price
        trade.close_time = int(time.time() * 1000)
        trade.profit = pnl
        trade.status = OrderStatus.CLOSED

        del self._trades[trade_id]
        self._update_equity()

        logger.info(
            f"[PAPER] Closed trade {trade_id} @ {close_price:.5f} "
            f"PnL={pnl:+.2f} | Balance={self._balance:.2f}"
        )
        return True

    async def modify_trade(self, trade_id: int, sl: float = 0,
                           tp: float = 0) -> bool:
        """Modify SL/TP of a paper trade."""
        trade = self._trades.get(trade_id)
        if not trade:
            return False

        old_sl, old_tp = trade.sl, trade.tp
        if sl:
            trade.sl = sl
        if tp:
            trade.tp = tp

        logger.info(
            f"[PAPER] Modified trade {trade_id}: "
            f"SL {old_sl:.5f}→{trade.sl:.5f}, TP {old_tp:.5f}→{trade.tp:.5f}"
        )
        return True

    async def get_open_trades(self) -> list[Trade]:
        """Return all open paper trades with updated unrealized PnL."""
        self._update_equity()
        return list(self._trades.values())

    async def get_account_info(self) -> AccountInfo:
        self._update_equity()
        margin = sum(
            t.volume * t.open_price * t.contract_size / 100  # 1:100 leverage
            for t in self._trades.values()
        )
        return AccountInfo(
            balance=self._balance,
            equity=self._equity,
            margin=margin,
            free_margin=self._equity - margin,
            margin_level=(self._equity / margin * 100) if margin > 0 else 0,
            currency=self._currency,
        )

    def _update_equity(self) -> None:
        """Recalculate equity based on open positions and latest prices."""
        unrealized = 0.0
        for trade in self._trades.values():
            current = self._latest_prices.get(trade.symbol, trade.open_price)
            if trade.side == Side.BUY:
                trade_pnl = (current - trade.open_price) * trade.volume * trade.contract_size
            else:
                trade_pnl = (trade.open_price - current) * trade.volume * trade.contract_size
            trade.profit = trade_pnl
            unrealized += trade_pnl

        self._equity = self._balance + unrealized

    def save_state(self) -> None:
        """Persist balance and open trades to JSON."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "balance": self._balance,
            "equity": self._equity,
            "currency": self._currency,
            "next_trade_id": self._next_trade_id,
            "trades": {
                str(tid): {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side.value,
                    "volume": t.volume,
                    "open_price": t.open_price,
                    "open_time": t.open_time,
                    "sl": t.sl,
                    "tp": t.tp,
                    "profit": t.profit,
                    "contract_size": t.contract_size,
                }
                for tid, t in self._trades.items()
            },
        }
        self._state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info(f"Paper state saved: balance={self._balance:.2f}, {len(self._trades)} open trades")

    def _load_state(self) -> None:
        """Load persisted state if file exists."""
        if not self._state_file.exists():
            return
        try:
            state = json.loads(self._state_file.read_text(encoding="utf-8"))
            self._balance = state["balance"]
            self._equity = state.get("equity", self._balance)
            self._next_trade_id = state.get("next_trade_id", 1)

            for tid_str, t_data in state.get("trades", {}).items():
                trade = Trade(
                    trade_id=t_data["trade_id"],
                    symbol=t_data["symbol"],
                    side=Side.BUY if t_data["side"] == "buy" else Side.SELL,
                    volume=t_data["volume"],
                    open_price=t_data["open_price"],
                    open_time=t_data["open_time"],
                    sl=t_data.get("sl", 0),
                    tp=t_data.get("tp", 0),
                    profit=t_data.get("profit", 0),
                    status=OrderStatus.OPENED,
                    contract_size=t_data.get("contract_size", 100000),
                )
                self._trades[trade.trade_id] = trade

            logger.info(
                f"Paper state loaded: balance={self._balance:.2f}, "
                f"{len(self._trades)} open trades"
            )
        except Exception as e:
            logger.warning(f"Could not load paper state: {e}")
