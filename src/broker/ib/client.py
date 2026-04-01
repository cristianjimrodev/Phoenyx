"""Interactive Brokers client implementing the BrokerClient interface."""
from __future__ import annotations

import asyncio
import time

import pandas as pd
from loguru import logger
from ib_insync import (
    IB, Contract, ContFuture, Forex, Stock, Future, CFD, Crypto,
    MarketOrder, LimitOrder, Trade as IBTrade,
    util,
)

from src.broker.base import (
    AccountInfo, BrokerClient, Candle, PriceCallback, Side, Symbol,
    TickPrice, Trade, OrderStatus,
)


# Timeframe mapping to IB bar sizes
BAR_SIZES = {
    "M1": "1 min",
    "M5": "5 mins",
    "M15": "15 mins",
    "M30": "30 mins",
    "H1": "1 hour",
    "H4": "4 hours",
    "D1": "1 day",
    "W1": "1 week",
    "MN1": "1 month",
}

# Duration strings for IB based on bar count and period
DURATIONS = {
    "M1": "1 D",
    "M5": "2 D",
    "M15": "5 D",
    "M30": "10 D",
    "H1": "1 M",
    "H4": "3 M",
    "D1": "1 Y",
    "W1": "2 Y",
    "MN1": "5 Y",
}


def _parse_symbol(symbol_str: str) -> Contract:
    """Parse a symbol string into an IB Contract.

    Supported formats:
      - Forex: 'EURUSD', 'GBPUSD' (6-char currency pairs)
      - Stocks: 'AAPL', 'MSFT' (assumed US stock)
      - Futures: 'ES', 'NQ' (assumed GLOBEX)
      - Crypto: 'BTC', 'ETH' (traded on PAXOS)
      - Indices/CFDs: 'US500', 'DE40'
    """
    # Forex pairs (6 characters, all alpha)
    forex_pairs = {
        # EUR pairs
        "EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD",
        "EURCAD", "EURNZD", "EURSEK", "EURNOK", "EURPLN",
        # GBP pairs
        "GBPUSD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD",
        "GBPNZD", "GBPSEK", "GBPNOK", "GBPPLN", "GBPSGD",
        # USD pairs
        "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        "USDSEK", "USDNOK", "USDPLN", "USDSGD", "USDMXN",
    }
    if symbol_str.upper() in forex_pairs:
        pair = symbol_str.upper()
        return Forex(pair[:3] + pair[3:])

    # Crypto
    crypto_map = {
        "BITCOIN": "BTC", "BTC": "BTC",
        "ETHEREUM": "ETH", "ETH": "ETH",
    }
    if symbol_str.upper() in crypto_map:
        return Crypto(crypto_map[symbol_str.upper()], "USD", "PAXOS")

    # Index CFDs
    cfd_map = {
        "US500": "SPX", "US30": "INDU", "US100": "NDX",
        "DE40": "DAX", "UK100": "Z",
    }
    if symbol_str.upper() in cfd_map:
        return CFD(cfd_map[symbol_str.upper()])

    # Futures
    futures_exchange = {
        "ES": "CME", "NQ": "CME", "YM": "CBOT", "RTY": "CME",
        "CL": "NYMEX", "GC": "COMEX", "SI": "COMEX",
        "ZB": "CBOT", "ZN": "CBOT",
        # Commodities / Agriculture
        "ZW": "CBOT", "ZC": "CBOT", "ZS": "CBOT", "ZL": "CBOT",  # wheat, corn, soybeans, soy oil
        "KC": "NYBOT", "CT": "NYBOT", "SB": "NYBOT", "CC": "NYBOT",  # coffee, cotton, sugar, cocoa
        "HG": "COMEX", "NG": "NYMEX",  # copper, natural gas
        # Metals
        "PA": "NYMEX", "PL": "NYMEX",  # palladium, platinum
        "ALI": "COMEX",  # aluminum
        # Note: zinc, nickel not directly on CME — use LME via SMART
    }
    if symbol_str.upper() in futures_exchange:
        # Use continuous futures for historical data
        return ContFuture(symbol_str.upper(), exchange=futures_exchange[symbol_str.upper()])

    # Gold
    if symbol_str.upper() == "GOLD":
        return Forex("XAUUSD")

    # Default: treat as US stock
    return Stock(symbol_str.upper(), "SMART", "USD")


class IBClient(BrokerClient):
    """Interactive Brokers API client using ib_insync."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1):
        """
        Args:
            host: TWS/Gateway host (default localhost)
            port: TWS paper=7497, TWS live=7496, Gateway paper=4002, Gateway live=4001
            client_id: Unique client ID for this connection
        """
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib = IB()
        self._price_callbacks: dict[str, PriceCallback] = {}
        self._ticker_map: dict[str, object] = {}
        self._contracts_cache: dict[str, Contract] = {}

    async def connect(self) -> bool:
        try:
            await self._ib.connectAsync(
                self._host, self._port, clientId=self._client_id
            )
            logger.info(f"Connected to IB at {self._host}:{self._port} "
                        f"(clientId={self._client_id})")

            # Log account info
            accounts = self._ib.managedAccounts()
            logger.info(f"Managed accounts: {accounts}")
            return True
        except Exception as e:
            logger.error(f"IB connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        self._ib.disconnect()
        logger.info("Disconnected from IB")

    async def get_symbols(self) -> list[Symbol]:
        # IB doesn't have a "get all symbols" - return empty
        # Symbols are defined in config
        logger.warning("IB does not support getAllSymbols. Use config symbols.")
        return []

    async def get_symbol(self, symbol: str) -> Symbol:
        contract = await self._resolve_contract(symbol)
        details_list = await self._ib.reqContractDetailsAsync(contract)

        if not details_list:
            raise ValueError(f"No contract details found for {symbol}")

        details = details_list[0]
        c = details.contract

        return Symbol(
            name=symbol,
            description=details.longName or symbol,
            category=c.secType,
            currency=c.currency,
            lot_min=float(details.minSize) if hasattr(details, 'minSize') else 1.0,
            lot_max=100000.0,
            lot_step=float(details.sizeIncrement) if hasattr(details, 'sizeIncrement') else 1.0,
            pip_size=float(details.priceMagnifier) if details.priceMagnifier else 1.0,
            contract_size=float(details.contract.multiplier) if details.contract.multiplier else 1.0,
            leverage=1.0,
        )

    async def get_candles(self, symbol: str, period: int | str, count: int,
                          duration: str | None = None) -> pd.DataFrame:
        contract = await self._resolve_contract(symbol)

        # period can be a string like "H1" or an int (minutes)
        if isinstance(period, str):
            timeframe = period
        else:
            # Map int minutes to timeframe key
            minute_map = {1: "M1", 5: "M5", 15: "M15", 30: "M30",
                          60: "H1", 240: "H4", 1440: "D1", 10080: "W1", 43200: "MN1"}
            timeframe = minute_map.get(period, "H1")

        bar_size = BAR_SIZES.get(timeframe, "1 hour")
        if duration is None:
            duration = DURATIONS.get(timeframe, "1 M")

        bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="MIDPOINT" if contract.secType == "CASH" else "TRADES",
            useRTH=False,
            formatDate=1,
        )

        if not bars:
            logger.warning(f"No historical data for {symbol}")
            return pd.DataFrame()

        rows = []
        for bar in bars:
            rows.append({
                "timestamp": int(bar.date.timestamp() * 1000) if hasattr(bar.date, 'timestamp') else 0,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("datetime")

        # Trim to requested count
        if len(df) > count:
            df = df.tail(count)

        return df

    async def subscribe_prices(self, symbol: str, callback: PriceCallback) -> None:
        contract = await self._resolve_contract(symbol)
        self._price_callbacks[symbol] = callback

        ticker = self._ib.reqMktData(contract, "", False, False)
        self._ticker_map[symbol] = ticker

        # Set up update callback
        ticker.updateEvent += lambda t: asyncio.ensure_future(
            self._on_tick_update(symbol, t)
        )

        logger.info(f"Subscribed to price stream: {symbol}")

    async def unsubscribe_prices(self, symbol: str) -> None:
        ticker = self._ticker_map.pop(symbol, None)
        if ticker:
            self._ib.cancelMktData(ticker.contract)
        self._price_callbacks.pop(symbol, None)
        logger.info(f"Unsubscribed from price stream: {symbol}")

    async def open_trade(self, symbol: str, side: Side, volume: float,
                         sl: float = 0, tp: float = 0) -> Trade:
        contract = await self._resolve_contract(symbol)

        action = "BUY" if side == Side.BUY else "SELL"
        order = MarketOrder(action, volume)

        ib_trade = self._ib.placeOrder(contract, order)

        # Wait for fill
        while not ib_trade.isDone():
            await asyncio.sleep(0.1)
            self._ib.sleep(0)

        fill_price = ib_trade.orderStatus.avgFillPrice or 0
        order_id = ib_trade.order.orderId

        logger.info(f"Trade opened: {action} {volume} {symbol} @ {fill_price} "
                     f"(orderId={order_id})")

        # Place SL/TP bracket orders if specified
        if sl > 0:
            sl_action = "SELL" if side == Side.BUY else "BUY"
            sl_order = LimitOrder(sl_action, volume, sl)
            sl_order.orderType = "STP"
            sl_order.auxPrice = sl
            self._ib.placeOrder(contract, sl_order)

        if tp > 0:
            tp_action = "SELL" if side == Side.BUY else "BUY"
            tp_order = LimitOrder(tp_action, volume, tp)
            self._ib.placeOrder(contract, tp_order)

        return Trade(
            trade_id=order_id,
            symbol=symbol,
            side=side,
            volume=volume,
            open_price=fill_price,
            open_time=int(time.time() * 1000),
            sl=sl,
            tp=tp,
            status=OrderStatus.OPENED,
        )

    async def close_trade(self, trade_id: int) -> bool:
        # Find the open trade and close it with a market order
        open_trades = self._ib.openTrades()
        target = None
        for t in open_trades:
            if t.order.orderId == trade_id:
                target = t
                break

        if not target:
            logger.warning(f"Trade {trade_id} not found")
            return False

        close_action = "SELL" if target.order.action == "BUY" else "BUY"
        close_order = MarketOrder(close_action, target.order.totalQuantity)
        self._ib.placeOrder(target.contract, close_order)

        logger.info(f"Trade {trade_id} close order placed")
        return True

    async def modify_trade(self, trade_id: int, sl: float = 0, tp: float = 0) -> bool:
        # IB doesn't modify trades directly - would need to cancel/replace bracket orders
        logger.warning("Modify trade not fully implemented for IB - cancel and replace manually")
        return False

    async def get_open_trades(self) -> list[Trade]:
        positions = self._ib.positions()
        trades = []
        for pos in positions:
            if pos.position == 0:
                continue
            side = Side.BUY if pos.position > 0 else Side.SELL
            trades.append(Trade(
                trade_id=0,
                symbol=pos.contract.symbol,
                side=side,
                volume=abs(pos.position),
                open_price=pos.avgCost,
                open_time=0,
                profit=pos.unrealizedPNL if hasattr(pos, 'unrealizedPNL') else 0,
                status=OrderStatus.OPENED,
            ))
        return trades

    async def get_account_info(self) -> AccountInfo:
        # Use account values which are updated automatically
        account_id = self._ib.managedAccounts()[0] if self._ib.managedAccounts() else ""

        summary = await self._ib.accountSummaryAsync()

        values = {}
        for v in summary:
            if v.tag in ("TotalCashValue", "NetLiquidation",
                         "BuyingPower", "GrossPositionValue"):
                try:
                    values[v.tag] = float(v.value)
                except (ValueError, TypeError):
                    pass

        return AccountInfo(
            balance=values.get("TotalCashValue", 0),
            equity=values.get("NetLiquidation", 0),
            margin=values.get("GrossPositionValue", 0),
            free_margin=values.get("BuyingPower", 0),
            margin_level=0,
            currency="USD",
        )

    # --- Internal ---

    async def _resolve_contract(self, symbol: str) -> Contract:
        """Resolve and qualify a contract by symbol name."""
        if symbol in self._contracts_cache:
            return self._contracts_cache[symbol]

        contract = _parse_symbol(symbol)

        qualified = await self._ib.qualifyContractsAsync(contract)
        if qualified:
            contract = qualified[0]

        self._contracts_cache[symbol] = contract
        return contract

    async def _on_tick_update(self, symbol: str, ticker) -> None:
        callback = self._price_callbacks.get(symbol)
        if not callback:
            return

        tick = TickPrice(
            symbol=symbol,
            ask=ticker.ask if ticker.ask == ticker.ask else 0,  # NaN check
            bid=ticker.bid if ticker.bid == ticker.bid else 0,
            spread=(ticker.ask - ticker.bid) if (ticker.ask == ticker.ask and ticker.bid == ticker.bid) else 0,
            timestamp=int(time.time() * 1000),
        )
        await callback(tick)
