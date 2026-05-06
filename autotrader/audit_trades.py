import sqlite3
db = sqlite3.connect("data/trades.db")
db.row_factory = sqlite3.Row
rows = db.execute("SELECT strategy, direction, entry_price, exit_price, pnl, status, created_at, closed_at, symbol FROM trades WHERE status = 'closed' ORDER BY closed_at DESC LIMIT 200").fetchall()
stats = {}
for r in rows:
    s = r["strategy"]
    if s not in stats:
        stats[s] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0, "recent": []}
    pnl = r["pnl"] or 0
    stats[s]["total_pnl"] += pnl
    stats[s]["count"] += 1
    if pnl > 0:
        stats[s]["wins"] += 1
    else:
        stats[s]["losses"] += 1
    if len(stats[s]["recent"]) < 5:
        stats[s]["recent"].append(f'  {r["closed_at"]} {r["symbol"]} {r["direction"]} pnl={pnl:.2f}')

for s, d in sorted(stats.items(), key=lambda x: x[1]["total_pnl"]):
    total = d["count"]
    wr = d["wins"] / total * 100 if total else 0
    print(f'{s}: {total} trades, {d["wins"]}W/{d["losses"]}L, WR={wr:.0f}%, PnL=${d["total_pnl"]:.2f}')
    for line in d["recent"]:
        print(line)

total_pnl = sum(r["pnl"] or 0 for r in rows)
print(f'\nTotal: {len(rows)} trades, PnL=${total_pnl:.2f}')

# Also show open trades
open_rows = db.execute("SELECT strategy, direction, entry_price, symbol, created_at, unrealised_pnl FROM trades WHERE status != 'closed' ORDER BY created_at DESC").fetchall()
if open_rows:
    print(f'\nOpen positions: {len(open_rows)}')
    for r in open_rows:
        print(f'  {r["strategy"]} {r["symbol"]} {r["direction"]} entry={r["entry_price"]} since={r["created_at"]}')
