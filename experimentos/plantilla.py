"""Plantilla de experimento. Cópiala (``experimentos/mi_idea.py``) y ejecútala:

    python -m experimentos.mi_idea

Antes de ejecutar, **escribe aquí la hipótesis y la regla de decisión**: qué esperas, a qué
horizonte, qué resultado te haría usarla y cuál descartarla. Escribirlo después de ver el
resultado es exactamente el sobreajuste que el laboratorio intenta evitar (§9.7).

Hipótesis: ...
Horizonte: ...
La uso si: la diferencia tiene el signo esperado, q < 0,10 en ``summary()`` y el mismo signo
en las dos mitades. Después, UNA vez, en el holdout.
"""

from __future__ import annotations

import pandas as pd

from lab import Lab, evaluate, summary, test
from lab.signals import build

lab = Lab()

# --- 1. Una regla del catálogo con otros parámetros --------------------------------------
# `python run_lab.py reglas` lista todas. Cambiar un parámetro es una prueba NUEVA.
for result in test(lab, "knife", max_return=-0.30)[0]:
    print(result)

# --- 2. Una regla temporal propia: por ejemplo, "el dólar sube más de un 2 % en 3 meses" ---
# ⚠️ Las observaciones son fechas separadas por el horizonte (independientes): una señal rara
# da muy pocas. El dólar amplio solo tiene historia publicada desde 2019, y con +5 % la
# rejilla no cae dentro de ningún episodio (n = 0). Mira siempre la n antes que la media.
dollar = lab.fred("DTWEXBGS")                       # tal como se publicó cada día (§9.4)
change = dollar / dollar.shift(63) - 1
signal = (change > 0.02).astype(float).where(change.notna())
for result in evaluate.timing(lab, signal, rule="mi_dolar_fuerte", horizons=(30, 90),
                              expected="lower", params={"rise": 0.02, "days": 63}):
    print(result)

# --- 3. Una regla transversal propia: por ejemplo, "rentabilidad de 6 meses, la mejor" ----
tr = lab.total_return()


def six_months(day: pd.Timestamp, members: list[str]) -> pd.Series:
    """Solo lo conocido el día ``day``: precios hasta ese cierre, nunca después."""
    past = tr[members].loc[:day].ffill(limit=5)
    if len(past) < 130:
        return pd.Series(dtype=float)
    return (past.iloc[-1] / past.iloc[-127] - 1).dropna()


results, per_date = evaluate.cross_section(lab, six_months, rule="mi_momentum_6m",
                                           mode="top", horizons=(90,),
                                           params={"months": 6})
for result in results:
    print(result)

# --- 4. Eventos propios: cualquier tabla [ticker, date] ----------------------------------
# Por ejemplo, las compras de directivos de más de 1 M USD (fecha de presentación).
purchases = lab.insider_purchases()
print(purchases.columns.tolist())          # mira qué columnas hay antes de filtrar

# --- 5. Una estrategia simulada con una regla (NO valida nada) ---------------------------
scores, _ = build(lab, "low_accruals")
backtest = evaluate.strategy(lab, lambda d, m: scores(d, m).nlargest(30).index,
                             rebalance_days=91, cost_bps=10)
print(backtest)

# --- 6. Todo lo que has probado, corregido por el número de pruebas ----------------------
print(summary().head(20).to_string())
