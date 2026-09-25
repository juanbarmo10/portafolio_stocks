"""📋 Desplegar capital — level 4: is it time to execute the plan? (CLAUDE.md section 2).

The last step of the checklist, and the only page that talks about putting money in. It
answers three questions in order, and the order is the point: whether today is a day to
execute at all (levels 1-3 as gates), how large the tranche is and how many orders it
deserves, and what it costs to bring the pesos to IBKR.

Private: it shows the account's cash and deposits. Not in ``PUBLIC_PAGES``.

Nothing here places an order or moves money (section 12): the page prepares a decision
that the user executes in IBKR by hand.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import MISSING, money, pct
from core.config import load_settings
from ingest.macro_calendar import current_calendar
from transform import funding as fx
from transform import portfolio as port
from transform import regime as rg

settings = load_settings()
local = settings.raw
level3 = local.get("panel", {}).get("level3", {})
earnings_window = int(level3.get("earnings_window_days", 5))
funding_cfg = dict(local.get("funding") or {})
contribution = dict(local.get("contributions") or {})

st.title("📋 Desplegar capital")
st.caption("Nivel 4 del checklist. La página prepara la decisión; la orden la pones tú en "
           "IBKR. El panel es de solo lectura y nunca opera (§12).")

if not app_data.database_ready():
    st.info("Todavía no hay base de datos. Ejecuta `python run_ingest.py`.")
    st.stop()

today = pd.Timestamp.today().normalize()

# --- 1. Gates ---------------------------------------------------------------------------
st.subheader("1 · ¿Es día de ejecutar?")
gates: list[tuple[str, str, str]] = []

view = app_data.regime_view(app_data.db_mtime())
if view is None:
    gates.append(("⚪", "Semáforo (niveles 1 y 2)", "sin datos todavía"))
else:
    r = view["reading"]
    icon = {rg.RISK_OFF: "🔴", rg.RISK_ON: "🟢"}.get(r.verdict, "🟡")
    gates.append((icon, "Semáforo (niveles 1 y 2)",
                  f"{rg.verdict_label(r.verdict)} al {r.date}"))

events = app_data.events()
calendar = current_calendar(events) if not events.empty else pd.DataFrame()
soon = []
for row in calendar.to_dict("records"):
    when = pd.Timestamp(row["ts"])
    if today.tz_localize("UTC") <= when <= (today + pd.Timedelta(hours=48)).tz_localize("UTC"):
        soon.append(f"{row['label']} ({when.tz_convert('America/New_York'):%Y-%m-%d %H:%M} NY)")
gates.append(("🟡" if soon else "🟢", "Dato macro de alto impacto en 48 h",
              "; ".join(soon) + " — §2 propone posponer el aporte" if soon else "ninguno"))

companies = app_data.companies()
by_ticker = dict(zip(companies["ticker"], companies["cik"])) if not companies.empty else {}
positions = port.latest_positions(app_data.account_observations())
watched = {str(c["cik"]).zfill(10): str(c["ticker"]) for c in settings.researched_companies
           if c.get("cik")}
for ticker in positions["ticker"] if not positions.empty else []:
    if by_ticker.get(ticker):
        watched.setdefault(by_ticker[ticker], ticker)
near = []
if not events.empty:
    earnings = events[(events["category"] == "earnings") & events["cik"].isin(watched)]
    for row in earnings.to_dict("records"):
        day = pd.Timestamp(str(row["ts"])[:10])
        if today <= day <= today + pd.Timedelta(days=earnings_window):
            kind = "estimada" if int(row.get("is_estimated") or 0) else "confirmada"
            near.append(f"{watched[row['cik']]} el {day:%Y-%m-%d} ({kind})")
gates.append(("🟡" if near else "🟢", f"Resultados en {earnings_window} días",
              "; ".join(near) + " — no abrir posición en ellas sin decisión explícita"
              if near else "ninguno en cartera ni en estudio"))

st.markdown("| | Puerta | Estado |\n|---|---|---|\n"
            + "\n".join(f"| {i} | **{g}** | {s} |" for i, g, s in gates))
if view is not None and view["reading"].verdict == rg.RISK_OFF:
    st.error("Regla dura de §2: con el semáforo en rojo no se compra, aunque la empresa sea "
             "perfecta. El dinero espera en efectivo; en la validación, esperar costó poco.")

# --- 2. The tranche ---------------------------------------------------------------------
st.subheader("2 · El tramo")
nav = app_data.account_observations()
cash = port.nav_series(nav, port.NAV_CASH) if not nav.empty else pd.DataFrame()
cash_now = float(cash["value"].iloc[-1]) if not cash.empty else None
fee = fx.order_fee(app_data.trades())
default_amount = float(contribution.get("monthly_amount_usd") or 100.0)

c1, c2, c3 = st.columns(3)
amount = c1.number_input("Tramo (USD)", min_value=1.0, value=default_amount, step=10.0,
                         help="Definido ANTES de abrir IBKR (§2, nivel 4).")
orders = c2.number_input("Órdenes", min_value=1, max_value=5, value=1, step=1)
c3.metric("Efectivo en IBKR, último extracto", money(cash_now, public=False))
share = None if fee is None else orders * fee / amount
st.markdown(
    f"Comisión mediana por ejecución en tu cuenta: **{money(fee, public=False)}**. "
    f"{orders} orden(es) sobre {money(amount, public=False)} = **{pct(share, decimals=2)}** "
    "del tramo. La comisión mínima es por orden, así que con tramos pequeños **una orden "
    "por aporte** — a la posición más por debajo de su peso objetivo — y rotar entre "
    "aportes, en vez de repartir cada tramo entre varias.")
if cash_now and cash_now >= amount:
    st.info(f"Ya hay {money(cash_now, public=False)} en efectivo en IBKR. Desplegar eso no "
            "cuesta ninguna transferencia: va antes que cualquier depósito nuevo.")

targets = dict(local.get("portfolio", {}).get("target_weights") or {})
if targets and not positions.empty:
    valued = port.valuation(positions, port.latest_prices(app_data.prices_for(
        sorted(set(positions["ticker"]) | set(targets)))))
    drift = port.target_drift(valued, targets)
    if not drift.empty:
        under = drift.dropna(subset=["drift"]).iloc[0]
        st.markdown(f"Más por debajo de su objetivo: **{under['ticker']}** — pesa "
                    f"{pct(under['weight'])} de las acciones frente a un objetivo de "
                    f"{pct(under['target_weight'])}.")
else:
    st.caption("Sin cartera objetivo escrita (`portfolio.target_weights`, entrada de usuario "
               "nº4), el panel no puede decir a qué posición va el tramo: un drift contra un "
               "objetivo inexistente parecería equilibrio.")
st.caption("Antes de comprar: la regla de salida escrita en la `exit_ladder` de la ficha "
           "(§2, nivel 4). Si la empresa no tiene ficha, la posición existiría antes que la "
           "razón escrita para tenerla.")

# --- 3. The deposit ---------------------------------------------------------------------
st.subheader("3 · El depósito: de pesos a IBKR")
st.markdown(
    "Dos costes viajan con cada aporte y piden hábitos opuestos:\n"
    "- **Proporcional** — el diferencial del cambio frente a la TRM. Es el mismo porcentaje "
    "con 100 que con 1.000 dólares; agrupar no lo reduce. Solo lo baja un mejor cambio.\n"
    "- **Fijo** — una tarifa por transferencia, la comisión mínima por orden. Pesa más cuanto "
    "más pequeño el aporte; **agrupar es la única palanca**.")

cash_tx = app_data.cash_transactions()
deposits = cash_tx[cash_tx["kind"] == "deposit"] if not cash_tx.empty else pd.DataFrame()
trm = app_data.observations("banrep", app_data.db_mtime())
trm = trm[trm["series_id"] == "TRM:COP_USD"] if not trm.empty else trm
costs = fx.deposit_costs(deposits, funding_cfg.get("deposits") or [], trm)

if costs.empty:
    st.caption("Todavía no hay depósitos en el extracto de IBKR.")
else:
    st.dataframe(pd.DataFrame({
        "Abono en IBKR": costs["date"],
        "USD recibidos": costs["usd_received"],
        "COP pagados": costs["cop_paid"],
        "TRM": costs["trm"],
        "TRM del": costs["trm_date"],
        "Coste (COP)": costs["cost_cop"],
        "Coste": costs["cost_pct"],
    }), hide_index=True, width="stretch", column_config={
        "USD recibidos": st.column_config.NumberColumn(format="localized"),
        "COP pagados": st.column_config.NumberColumn(format="localized"),
        "TRM": st.column_config.NumberColumn(format="localized"),
        "Coste (COP)": st.column_config.NumberColumn(format="localized"),
        "Coste": st.column_config.NumberColumn(format="percent"),
    })
    st.caption("Coste de punta a punta = COP pagados − USD abonados × TRM. Recoge el "
               "diferencial del cambio y **todas** las tarifas del camino (intermediario, "
               "bancos corresponsales, IBKR), sin suponer ninguna. IBKR no cargó nada por "
               "recibir estos depósitos: no hay ninguna tarifa en el extracto.")
    missing = costs[costs["cop_paid"].isna()]
    if not missing.empty:
        snippet = "funding:\n  deposits:\n" + "".join(
            f"    - {{date: {d}, cop_paid: null, bought_on: null}}\n" for d in missing["date"])
        st.markdown("Para medir el coste, apunta en `config/settings.local.yaml` los pesos que "
                    "pagaste por cada depósito (lo único que el extracto de IBKR no sabe). "
                    "`bought_on` es el día que compraste los dólares, si no fue el del abono:")
        st.code(snippet, language="yaml")

split = fx.split_costs(costs)
fixed_usd = funding_cfg.get("fixed_fee_usd")
proportional = funding_cfg.get("proportional_cost")
source = "config"
if split is not None and (fixed_usd is None or proportional is None):
    rate_now, _ = fx.trm_on(trm, today)
    fixed_usd = split.fixed_cop / rate_now if rate_now else None
    proportional = split.proportional
    source = f"ajuste lineal sobre {split.deposits} depósitos"
if fixed_usd is not None or proportional is not None:
    st.markdown(f"Coste fijo por transferencia **{money(fixed_usd, public=False)}** · "
                f"proporcional **{pct(proportional, decimals=2)}** — {source}.")

monthly = float(contribution.get("monthly_amount_usd") or amount)
table = fx.batching(monthly, every_months=[1, 2, 3, 6],
                    fixed_usd=None if fixed_usd is None else float(fixed_usd),
                    proportional=None if proportional is None else float(proportional),
                    fee_per_order=fee, orders=1)
st.markdown(f"**Aportando {money(monthly, public=False)} al mes, transferir cada…**")
st.dataframe(pd.DataFrame({
    "Cada (meses)": table["every_months"],
    "Transferencia (USD)": table["amount_usd"],
    "Fijo": table["fixed"],
    "Cambio": table["proportional"],
    "Comisión (1 orden)": table["commission"],
    "Total": table["total"],
    "Fijos al año (USD)": table["fixed_and_commission_usd_year"],
}), hide_index=True, width="stretch", column_config={
    "Transferencia (USD)": st.column_config.NumberColumn(format="localized"),
    "Fijo": st.column_config.NumberColumn(format="percent"),
    "Cambio": st.column_config.NumberColumn(format="percent"),
    "Comisión (1 orden)": st.column_config.NumberColumn(format="percent"),
    "Total": st.column_config.NumberColumn(format="percent"),
    "Fijos al año (USD)": st.column_config.NumberColumn(format="localized"),
})
if fixed_usd is None:
    st.caption(f"Columnas vacías: {MISSING} = coste todavía no medido. Se llenan al apuntar los "
               "pesos pagados de al menos tres depósitos de distinto tamaño, o al escribir "
               "`funding.fixed_fee_usd` y `funding.proportional_cost` si conoces la tarifa.")
st.caption(
    "Lo que la tabla no pone en número: agrupar deja el dinero esperando en pesos en vez de "
    "invertido. Con un horizonte de años eso cuesta poco — la validación del freno (fase 4) "
    "midió que aplazar aportes apenas cambió el resultado —, pero ponerle cifra exigiría un "
    "rendimiento esperado, que es un supuesto tuyo y no una medición. Regla práctica: si el "
    "fijo por transferencia pesa más que un mes de rentabilidad esperada del dinero que "
    "espera, agrupa.")
