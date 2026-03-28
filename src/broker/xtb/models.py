"""XTB xAPI constants and data mappings."""

# xAPI timeframe period codes
PERIODS = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}

# xAPI trade command types
CMD_BUY = 0
CMD_SELL = 1
CMD_BUY_LIMIT = 2
CMD_SELL_LIMIT = 3
CMD_BUY_STOP = 4
CMD_SELL_STOP = 5

# xAPI trade transaction types
TYPE_OPEN = 0
TYPE_PENDING = 1
TYPE_CLOSE = 2
TYPE_MODIFY = 3
TYPE_DELETE = 4
