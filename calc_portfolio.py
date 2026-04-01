"""Calculate portfolio returns from backtest results."""

results = {
    # Forex EUR (10)
    "EURUSD": 3.01, "EURGBP": 4.27, "EURJPY": 5.75, "EURCHF": -6.22,
    "EURAUD": -3.45, "EURCAD": 8.03, "EURNZD": 5.50, "EURSEK": 9.98,
    "EURNOK": 19.12, "EURPLN": 4.00,
    # Forex GBP (10)
    "GBPUSD": -0.36, "GBPJPY": 38.66, "GBPCHF": -7.76, "GBPAUD": -9.61,
    "GBPCAD": -0.47, "GBPNZD": -0.47, "GBPSEK": -9.72, "GBPNOK": 11.97,
    "GBPPLN": 5.75, "GBPSGD": 3.59,
    # Forex USD (10)
    "USDJPY": -15.13, "USDCHF": 1.44, "USDCAD": -13.60, "AUDUSD": 25.62,
    "NZDUSD": -8.19, "USDSEK": -0.59, "USDNOK": -4.07, "USDPLN": 11.69,
    "USDSGD": 1.30, "USDMXN": 5.62,
    # Indices (16)
    "QQQ": -4.07, "SPY": 10.24, "DIA": 0.00, "IWM": 8.03,
    "DAX": -2.00, "EWP": 5.75, "EWU": 1.68, "EWQ": -5.88,
    "EWI": 7.78, "FEZ": 3.88, "VIXY": -5.99,
    "EWW": 0.00, "EWZ": 1.92, "EWJ": 4.00, "VNM": 9.98, "EWH": 44.76,
    # Commodities (10)
    "NG": 29.04, "ZS": 46.45, "KC": -3.96, "CL": 14.57,
    "ZW": -2.00, "SB": -2.12, "CT": -2.12, "CC": 4.00,
    "ZL": -3.96, "ZC": 5.87,
    # Metals (6)
    "GC": 21.21, "SI": -4.07, "PA": -6.10, "PL": -6.21,
    "HG": 1.68, "ALI": 1.68,
    # Bonds (1)
    "ZN": 15.47,
}

n = len(results)
inversion = 1000.0
por_activo = inversion / n

total_final = 0
positivos = 0
negativos = 0

for ret in results.values():
    total_final += por_activo * (1 + ret / 100)
    if ret > 0:
        positivos += 1
    elif ret < 0:
        negativos += 1

profit = total_final - inversion
pct = (total_final / inversion - 1) * 100

print()
print("=" * 64)
print(f"  PORTFOLIO: 1.000 EUR repartidos en {n} activos")
print("=" * 64)
print(f"  Inversion por activo: {por_activo:.2f} EUR")
print(f"  Inversion total:      1.000,00 EUR")
print(f"  Valor final:          {total_final:,.2f} EUR")
print(f"  Beneficio neto:       {profit:+,.2f} EUR")
print(f"  Rentabilidad total:   {pct:+.2f}%")
print(f"  Activos positivos:    {positivos}/{n} ({positivos/n*100:.0f}%)")
print(f"  Activos negativos:    {negativos}/{n}")
print(f"  Activos neutros:      {n - positivos - negativos}/{n}")
print("-" * 64)

items = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("  TOP 5:")
for sym, ret in items[:5]:
    f = por_activo * (1 + ret / 100)
    print(f"    {sym:<10} {ret:>+7.2f}%   {por_activo:.2f} -> {f:.2f} EUR  ({f - por_activo:+.2f})")
print("  BOTTOM 5:")
for sym, ret in items[-5:]:
    f = por_activo * (1 + ret / 100)
    print(f"    {sym:<10} {ret:>+7.2f}%   {por_activo:.2f} -> {f:.2f} EUR  ({f - por_activo:+.2f})")

print("-" * 64)
cats = {
    "Forex EUR": ["EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD",
                   "EURCAD", "EURNZD", "EURSEK", "EURNOK", "EURPLN"],
    "Forex GBP": ["GBPUSD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD",
                   "GBPNZD", "GBPSEK", "GBPNOK", "GBPPLN", "GBPSGD"],
    "Forex USD": ["USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
                   "USDSEK", "USDNOK", "USDPLN", "USDSGD", "USDMXN"],
    "Indices":   ["QQQ", "SPY", "DIA", "IWM", "DAX", "EWP", "EWU", "EWQ",
                   "EWI", "FEZ", "VIXY", "EWW", "EWZ", "EWJ", "VNM", "EWH"],
    "Commodities": ["NG", "ZS", "KC", "CL", "ZW", "SB", "CT", "CC", "ZL", "ZC"],
    "Metales":   ["GC", "SI", "PA", "PL", "HG", "ALI"],
    "Bonos":     ["ZN"],
}
print("  DESGLOSE POR CATEGORIA:")
for cat, syms in cats.items():
    inv = por_activo * len(syms)
    fin = sum(por_activo * (1 + results[s] / 100) for s in syms)
    ret = (fin / inv - 1) * 100
    ben = fin - inv
    print(f"    {cat:<15} {len(syms):>2} act.  {inv:>7.2f} -> {fin:>7.2f} EUR  {ret:>+6.2f}%  ({ben:>+6.2f})")

print("=" * 64)
mensual = pct / 10
anual = ((1 + pct / 100) ** (12 / 10) - 1) * 100
print(f"  Rentabilidad mensual:    {mensual:+.2f}%")
print(f"  Rentabilidad anualizada: {anual:+.2f}%")

balance = inversion
for mes in range(10):
    balance *= (1 + mensual / 100)
print(f"  Con interes compuesto:   {balance:,.2f} EUR ({(balance/inversion-1)*100:+.2f}%)")

print()
print("  PROYECCION CON INTERES COMPUESTO:")
for anos in [1, 2, 3, 5, 10]:
    meses = anos * 12
    valor = inversion * (1 + mensual / 100) ** meses
    print(f"    {anos:>2} anno{'s' if anos > 1 else ''}:  {valor:>10,.2f} EUR  ({(valor/inversion-1)*100:>+8.2f}%)")
print("=" * 64)
