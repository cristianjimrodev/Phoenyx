#!/usr/bin/env bash
set -euo pipefail

# Back up existing (empty) VM db, install uploaded one.
if [ -f ~/Phoenyx/data/trades.db ]; then
    mv ~/Phoenyx/data/trades.db ~/Phoenyx/data/trades.db.bak.$(date +%s)
fi
cp /tmp/trades.db ~/Phoenyx/data/trades.db

# Close the two trades that were opened locally but got stopped out on the
# first VM daily run (their rows exist in trades.db but update_order had no
# effect because the rows were inserted here only now).
sqlite3 ~/Phoenyx/data/trades.db <<'SQL'
UPDATE trades SET
    status = 'closed',
    exit_price = sl,
    pnl = CASE side
        WHEN 'buy'  THEN (sl - entry_price) * volume * contract_size
        WHEN 'sell' THEN (entry_price - sl) * volume * contract_size
        ELSE 0
    END
WHERE id IN (6, 7) AND status = 'opened';

SELECT id, symbol, side, status, ROUND(exit_price, 5) AS exit,
       ROUND(pnl, 2) AS pnl
FROM trades ORDER BY id;
SQL

sudo systemctl restart phoenyx-dashboard
echo "=== dashboard restarted ==="
