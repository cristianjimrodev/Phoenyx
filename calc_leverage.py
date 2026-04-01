"""Calculate portfolio with leverage."""

leverage = 20
base_return = 4.37  # % en 10 meses sin apalancamiento
mensual_base = base_return / 10
max_dd_portfolio = 7.0  # DD medio estimado del portfolio

lev_return = base_return * leverage
lev_mensual = mensual_base * leverage

print()
print("=" * 64)
print(f"  PORTFOLIO 1.000 EUR con apalancamiento 1:{leverage}")
print("=" * 64)
print(f"  Sin apalancamiento:    +{base_return:.2f}% (10 meses)")
print(f"  Con 1:{leverage}:               +{lev_return:.2f}% (10 meses)")
print(f"  Rent. mensual:         +{lev_mensual:.2f}%")
print()
valor_10m = 1000 * (1 + lev_return / 100)
print(f"  1.000 EUR -> {valor_10m:,.2f} EUR en 10 meses")
print(f"  Beneficio:   +{valor_10m - 1000:,.2f} EUR")
print()

print(f"  RIESGO:")
print(f"  Max DD portfolio sin lever:  ~{max_dd_portfolio:.1f}%")
print(f"  Max DD portfolio con 1:{leverage}:  ~{max_dd_portfolio * leverage:.1f}%")
print(f"  (diversificacion en 63 activos reduce el DD real)")
print()

print(f"  PROYECCION CON INTERES COMPUESTO (1:{leverage}):")
for anos in [1, 2, 3, 5, 10]:
    meses = anos * 12
    valor = 1000 * (1 + lev_mensual / 100) ** meses
    print(f"    {anos:>2} anno{'s' if anos > 1 else ' '}:  {valor:>12,.2f} EUR  ({(valor / 1000 - 1) * 100:>+9.2f}%)")
print()

print("  COMPARATIVA POR NIVEL DE APALANCAMIENTO:")
print(f"  {'Lever':<8} {'10 meses':>10} {'1 anno':>12} {'3 annos':>12} {'5 annos':>12} {'MaxDD':>8}")
print(f"  {'-' * 56}")
for lev in [1, 5, 10, 20, 30]:
    ret = base_return * lev
    m = mensual_base * lev
    v10m = 1000 * (1 + ret / 100)
    v1 = 1000 * (1 + m / 100) ** 12
    v3 = 1000 * (1 + m / 100) ** 36
    v5 = 1000 * (1 + m / 100) ** 60
    dd = max_dd_portfolio * lev
    print(f"  1:{lev:<5} {v10m:>10,.2f} {v1:>12,.2f} {v3:>12,.2f} {v5:>12,.2f} {dd:>7.0f}%")
print("=" * 64)
