"""XTB broker client implementing the BrokerClient interface."""
from __future__ import annotations

import asyncio
import time

import pandas as pd
from loguru import logger

from src.broker.base import (
    AccountInfo, BrokerClient, Candle, PriceCallback, Side, Symbol,
    TickPrice, Trade, OrderStatus,
)
from src.broker.xtb.models import CMD_BUY, CMD_SELL, PERIODS, TYPE_CLOSE, TYPE_MODIFY, TYPE_OPEN
from src.broker.xtb.websocket import XTBConnection


class XTBClient(BrokerClient):
    """XTB xStation 5 API client."""

    def __init__(self, user_id: str, password: str, mode: str = "demo",
                 urls: dict | None = None):
        self._user_id = user_id
        self._password = password
        self._mode = mode

        default_urls = {
            "demo": "wss://ws.xtb.com/demo",
            "demo_stream": "wss://ws.xtb.com/demoStream",
            "real": "wss://ws.xtb.com/real",
            "real_stream": "wss://ws.xtb.com/realStream",
        }
        urls = urls or default_urls

        cmd_url = urls[mode]
        stream_url = urls[f"{mode}_stream"]
        self._conn = XTBConnection(cmd_url, stream_url)
        self._stream_session_id: str | None = None
        self._price_callbacks: dict[str, PriceCallback] = {}
        self._stream_listener_task: asyncio.Task | None = None

    async def connect(self) -> bool:
        try:
            self._stream_session_id = await self._conn.login(
                self._user_id, self._password
            )
            self._stream_listener_task = asyncio.create_task(self._stream_listener())
            logger.info(f"XTB client connected in {self._mode} mode")
            return True
        except Exception as e:
            logger.error(f"XTB connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        if self._stream_listener_task:
            self._stream_listener_task.cancel()
            try:
                await self._stream_listener_task
            except asyncio.CancelledError:
                pass
        await self._conn.logout()

    async def get_symbols(self) -> list[Symbol]:
        resp = await self._conn.command.send("getAllSymbols")
        symbols = []
        for s in resp.get("returnData", []):
            symbols.append(self._parse_symbol(s))
        return symbols

    async def get_symbol(self, symbol: str) -> Symbol:
        resp = await self._conn.command.send("getSymbol", {"symbol": symbol})
        return self._parse_symbol(resp["returnData"])

    async def get_candles(self, symbol: str, period: int, count: int) -> pd.DataFrame:
        # period can be int (minutes) or string key like "H1"
        period_code = PERIODS.get(str(period), period) if isinstance(period, str) else period

        # Calculate start time based on count and period
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (count * period_code * 60 * 1000)

        resp = await self._conn.command.send("getChartRangeRequest", {
            "info": {
                "end": now_ms,
                "period": period_code,
                "start": start_ms,
                "symbol": symbol,
                "ticks": 0,
            }
        })

        data = resp.get("returnData", {})
        digits = data.get("digits", 5)
        candles = data.get("rateInfos", [])

        rows = []
        for c in candles:
            o = c["open"] / (10 ** digits)
            rows.append({
                "timestamp": c["ctm"],
                "open": o,
                "high": o + c["high"] / (10 ** digits),
                "low": o + c["low"] / (10 ** digits),
                "close": o + c["close"] / (10 ** digits),
                "volume": c["vol"],
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("datetime")
        return df

    async def subscribe_prices(self, symbol: str, callback: PriceCallback) -> None:
        self._price_callbacks[symbol] = callback
        await self._conn.stream.send_stream("getTickPrices", {
            "command": "getTickPrices",
            "streamSessionId": self._stream_session_id,
            "symbol": symbol,
            "minArrivalTime": 500,
            "maxLevel": 0,
        })
        logger.info(f"Subscribed to price stream: {symbol}")

    async def unsubscribe_prices(self, symbol: str) -> None:
        self._price_callbacks.pop(symbol, None)
        await self._conn.stream.send_stream("stopTickPrices", {
            "command": "stopTickPrices",
            "symbol": symbol,
        })
        logger.info(f"Unsubscribed from price stream: {symbol}")

    async def open_trade(self, symbol: str, side: Side, volume: float,
                         sl: float = 0, tp: float = 0) -> Trade:
        cmd = CMD_BUY if side == Side.BUY else CMD_SELL

        # Get current price for the order
        sym_info = await self.get_symbol(symbol)

        resp = await self._conn.command.send("tradeTransaction", {
            "tradeTransInfo": {
                "cmd": cmd,
                "customComment": "auto_trading_system",
                "expiration": 0,
                "offset": 0,
                "order": 0,
                "price": 1.0,  # Market order: price is ignored by server
                "sl": sl,
                "symbol": symbol,
                "tp": tp,
                "type": TYPE_OPEN,
                "volume": volume,
            }
        })

        order_id = resp["returnData"]["order"]
        logger.info(f"Trade opened: {side.value} {volume} {symbol} (order={order_id})")

        # Check trade status
        await asyncio.sleep(0.5)
        status_resp = await self._conn.command.send("tradeTransactionStatus", {
            "order": order_id
        })
        status_data = status_resp["returnData"]

        return Trade(
            trade_id=order_id,
            symbol=symbol,
            side=side,
            volume=volume,
            open_price=status_data.get("price", 0),
            open_time=int(time.time() * 1000),
            sl=sl,
            tp=tp,
            status=OrderStatus.OPENED,
        )

    async def close_trade(self, trade_id: int) -> bool:
        # Get trade info first to know symbol, volume, side
        trades = await self.get_open_trades()
        trade = next((t for t in trades if t.trade_id == trade_id), None)
        if not trade:
            logger.warning(f"Trade {trade_id} not found in open trades")
            return False

        # Close command: opposite cmd
        close_cmd = CMD_SELL if trade.side == Side.BUY else CMD_BUY

        resp = await self._conn.command.send("tradeTransaction", {
            "tradeTransInfo": {
                "cmd": close_cmd,
                "customComment": "auto_close",
                "expiration": 0,
                "offset": 0,
                "order": trade_id,
                "price": 1.0,
                "sl": 0,
                "symbol": trade.symbol,
                "tp": 0,
                "type": TYPE_CLOSE,
                "volume": trade.volume,
            }
        })

        logger.info(f"Trade {trade_id} closed")
        return True

    async def modify_trade(self, trade_id: int, sl: float = 0, tp: float = 0) -> bool:
        trades = await self.get_open_trades()
        trade = next((t for t in trades if t.trade_id == trade_id), None)
        if not trade:
            return False

        cmd = CMD_BUY if trade.side == Side.BUY else CMD_SELL

        await self._conn.command.send("tradeTransaction", {
            "tradeTransInfo": {
                "cmd": cmd,
                "customComment": "auto_modify",
                "expiration": 0,
                "offset": 0,
                "order": trade_id,
                "price": trade.open_price,
                "sl": sl,
                "symbol": trade.symbol,
                "tp": tp,
                "type": TYPE_MODIFY,
                "volume": trade.volume,
            }
        })

        logger.info(f"Trade {trade_id} modified: SL={sl} TP={tp}")
        return True

    async def get_open_trades(self) -> list[Trade]:
        resp = await self._conn.command.send("getTrades", {"openedOnly": True})
        trades = []
        for t in resp.get("returnData", []):
            side = Side.BUY if t["cmd"] == CMD_BUY else Side.SELL
            trades.append(Trade(
                trade_id=t["order"],
                symbol=t["symbol"],
                side=side,
                volume=t["volume"],
                open_price=t["open_price"],
                open_time=t["open_time"],
                sl=t.get("sl", 0),
                tp=t.get("tp", 0),
                profit=t.get("profit", 0),
                status=OrderStatus.OPENED,
            ))
        return trades

    async def get_account_info(self) -> AccountInfo:
        resp = await self._conn.command.send("getMarginLevel")
        d = resp["returnData"]
        return AccountInfo(
            balance=d["balance"],
            equity=d["equity"],
            margin=d["margin"],
            free_margin=d["margin_free"],
            margin_level=d.get("margin_level", 0),
            currency=d["currency"],
        )

    # --- Internal ---

    def _parse_symbol(self, data: dict) -> Symbol:
        return Symbol(
            name=data["symbol"],
            description=data.get("description", ""),
            category=data.get("categoryName", ""),
            currency=data.get("currency", ""),
            lot_min=data.get("lotMin", 0.01),
            lot_max=data.get("lotMax", 100),
            lot_step=data.get("lotStep", 0.01),
            pip_size=data.get("pipsPrecision", 4),
            contract_size=data.get("contractSize", 1),
            leverage=data.get("leverage", 1),
            swap_long=data.get("swapLong", 0),
            swap_short=data.get("swapShort", 0),
        )

    async def _stream_listener(self) -> None:
        """Listen for streaming messages and dispatch to callbacks."""
        while True:
            try:
                msg = await self._conn.stream.receive_stream()
                command = msg.get("command")

                if command == "tickPrices":
                    data = msg.get("data", {})
                    symbol = data.get("symbol", "")
                    callback = self._price_callbacks.get(symbol)
                    if callback:
                        tick = TickPrice(
                            symbol=symbol,
                            ask=data.get("ask", 0),
                            bid=data.get("bid", 0),
                            spread=data.get("spreadRaw", 0),
                            timestamp=data.get("timestamp", 0),
                        )
                        await callback(tick)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream listener error: {e}")
                await asyncio.sleep(1)
