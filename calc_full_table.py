"""Full results table with leverage by asset type."""

results = {
    # symbol: (return%, trades, tipo, max_leverage)
    # v5: cooldown + ATR filter + breakeven + partial + trailing 1.5ATR + exclusions + loss streak
    # Excluded: GBPAUD, GBPSEK, NZDUSD, PA, PL, VIXY (0 trades, protected capital)
    "EURUSD": (8.75, 13, "Forex Major", 30),
    "EURGBP": (4.27, 4, "Forex Major", 30),
    "EURJPY": (5.75, 9, "Forex Major", 30),
    "EURCHF": (0.00, 0, "Forex Major", 30),
    "EURAUD": (-0.21, 18, "Forex Minor", 20),
    "EURCAD": (8.03, 5, "Forex Minor", 20),
    "EURNZD": (5.50, 15, "Forex Minor", 20),
    "EURSEK": (9.98, 10, "Forex Minor", 20),
    "EURNOK": (-0.08, 18, "Forex Minor", 20),
    "EURPLN": (4.00, 1, "Forex Minor", 20),
    "GBPUSD": (-0.36, 9, "Forex Major", 30),
    "GBPJPY": (17.54, 28, "Forex Major", 30),
    "GBPCHF": (0.00, 0, "Forex Major", 30),
    "GBPCAD": (-0.47, 12, "Forex Minor", 20),
    "GBPNZD": (-0.47, 12, "Forex Minor", 20),
    "GBPSEK": (3.43, 11, "Forex Minor", 20),
    "GBPNOK": (11.97, 6, "Forex Minor", 20),
    "GBPPLN": (5.75, 9, "Forex Minor", 20),
    "GBPSGD": (3.59, 4, "Forex Minor", 20),
    "USDJPY": (13.69, 35, "Forex Major", 30),
    "USDCHF": (4.00, 1, "Forex Major", 30),
    "USDCAD": (-0.83, 34, "Forex Minor", 20),
    "AUDUSD": (-0.21, 18, "Forex Major", 30),
    "USDSEK": (-0.59, 15, "Forex Minor", 20),
    "USDNOK": (-4.07, 5, "Forex Minor", 20),
    "USDPLN": (11.69, 13, "Forex Minor", 20),
    "USDSGD": (1.30, 17, "Forex Minor", 20),
    "USDMXN": (5.62, 12, "Forex Minor", 20),
    "QQQ": (3.47, 6, "Indice", 20),
    "SPY": (3.08, 2, "Indice", 20),
    "DIA": (0.00, 0, "Indice", 20),
    "IWM": (1.40, 1, "Indice", 20),
    "DAX": (-2.00, 1, "Indice", 20),
    "EWP": (5.75, 9, "Indice", 20),
    "EWU": (1.68, 8, "Indice", 20),
    "EWQ": (-5.88, 3, "Indice", 20),
    "EWI": (7.78, 11, "Indice", 20),
    "FEZ": (3.88, 4, "Indice", 20),
    "EWW": (0.00, 0, "Indice", 20),
    "EWZ": (1.92, 2, "Indice", 20),
    "EWJ": (4.00, 1, "Indice", 20),
    "VNM": (9.98, 10, "Indice", 20),
    "EWH": (12.53, 8, "Indice", 20),
    "NG": (10.00, 22, "Commodity", 10),
    "ZS": (-7.95, 19, "Commodity", 10),
    "KC": (2.22, 1, "Commodity", 10),
    "CL": (-2.90, 16, "Commodity", 10),
    "ZW": (-2.00, 1, "Commodity", 10),
    "SB": (-2.12, 4, "Commodity", 10),
    "CT": (-2.12, 4, "Commodity", 10),
    "CC": (4.00, 1, "Commodity", 10),
    "ZL": (-3.96, 2, "Commodity", 10),
    "ZC": (5.87, 6, "Commodity", 10),
    "GC": (0.66, 5, "Metal", 10),
    "SI": (-3.96, 2, "Metal", 10),
    "HG": (0.00, 0, "Metal", 10),
    "ALI": (1.68, 8, "Metal", 10),
    "ZN": (3.31, 25, "Bono", 10),
}

# Weighted allocation by category performance
category_weights = {
    "Forex Major": 1.5,   # best performers, more capital
    "Forex Minor": 0.8,
    "Indice": 1.3,
    "Commodity": 1.0,
    "Metal": 0.5,         # weakest, less capital
    "Bono": 1.0,
}

# Calculate weighted allocation
total_weight = sum(
    category_weights.get(tipo, 1.0)
    for _, (_, _, tipo, _) in results.items()
)
inv = 1000.0
n = len(results)

print()
print("=" * 100)
print(f"  RESULTADOS COMPLETOS - 63 ACTIVOS - 10 MESES - DATOS REALES IB")
print("=" * 100)
print(f"  {'Symbol':<10} {'Tipo':<14} {'Return':>8} {'Trades':>7} {'Lever':>6} {'Ret*Lever':>10} {'Invest':>9} {'Final':>9} {'PnL':>9}")
print("  " + "-" * 93)

total_final_lev = 0
pos = 0
neg = 0
total_trades = 0

sorted_items = sorted(results.items(), key=lambda x: x[1][0] * x[1][3], reverse=True)

for sym, (ret, trades, tipo, lever) in sorted_items:
    w = category_weights.get(tipo, 1.0)
    pa = inv * w / total_weight
    ret_lev = ret * lever
    final_lev = pa * (1 + ret_lev / 100)
    pnl = final_lev - pa
    total_final_lev += final_lev
    total_trades += trades
    if ret > 0:
        pos += 1
    elif ret < 0:
        neg += 1
    print(f"  {sym:<10} {tipo:<14} {ret:>+7.2f}% {trades:>6}  1:{lever:<4} {ret_lev:>+9.1f}%  {pa:>8.2f}  {final_lev:>8.2f}  {pnl:>+8.2f}")

profit_lev = total_final_lev - inv
pct_lev = profit_lev / inv * 100

print("  " + "-" * 93)
print(f"  {'TOTAL':<10} {n} activos                           {pct_lev:>+9.1f}%  {inv:>8.2f}  {total_final_lev:>8.2f}  {profit_lev:>+8.2f}")
print("=" * 100)

print()
print(f"  Inversion:            1.000,00 EUR")
print(f"  Valor final:          {total_final_lev:,.2f} EUR")
print(f"  Beneficio neto:       {profit_lev:+,.2f} EUR")
print(f"  Rentabilidad (10m):   {pct_lev:+.2f}%")
print(f"  Total operaciones:    {total_trades}")
print(f"  Activos positivos:    {pos}/{n} ({pos/n*100:.0f}%)")
print(f"  Activos negativos:    {neg}/{n}")
print()

m_lev = pct_lev / 10
anual = ((1 + m_lev / 100) ** 12 - 1) * 100
print(f"  Rent. mensual:        {m_lev:+.2f}%")
print(f"  Rent. anualizada:     {anual:+.2f}%")
print()

# Por categoria con leverage
cats = {
    "Forex Major": [], "Forex Minor": [],
    "Indice": [], "Commodity": [], "Metal": [], "Bono": [],
}
for sym, (ret, trades, tipo, lever) in results.items():
    cats[tipo].append((sym, ret, lever))

print("  POR CATEGORIA (con leverage + peso):")
print(f"  {'Categoria':<14} {'N':>3} {'Peso':>5} {'Invest':>9} {'Final':>9} {'PnL':>9} {'Ret':>8}")
print("  " + "-" * 62)
for cat, items in cats.items():
    w = category_weights.get(cat, 1.0)
    pa_cat = inv * w / total_weight
    c_inv = pa_cat * len(items)
    c_fin = sum(pa_cat * (1 + r * l / 100) for _, r, l in items)
    c_pnl = c_fin - c_inv
    c_ret = (c_fin / c_inv - 1) * 100 if c_inv > 0 else 0
    print(f"  {cat:<14} {len(items):>3}  x{w:.1f}  {c_inv:>9.2f} {c_fin:>9.2f} {c_pnl:>+9.2f} {c_ret:>+7.2f}%")

print()
print("  PROYECCION CON INTERES COMPUESTO:")
for a in [1, 2, 3, 5]:
    v = 1000 * (1 + m_lev / 100) ** (a * 12)
    print(f"    {a} anno{'s' if a > 1 else ' '}: {v:>12,.2f} EUR  ({(v / 1000 - 1) * 100:>+.1f}%)")
print("=" * 100)
