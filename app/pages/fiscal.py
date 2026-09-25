"""🧾 Fiscal — the local-currency view of the account (CLAUDE.md sections 9.9, 11). LOCAL ONLY.

Never deployed: it is not in ``PUBLIC_PAGES``, ``app/main.py`` drops it in public mode, and
the page itself stops if it is ever reached there. Its configuration reveals the owner's
jurisdiction, and its figures are the account's.

**ESTIMADO everywhere.** Nothing on this page is a settled tax. It asserts no rate, no
threshold and no treatment; what depends on one says "no calculable" until the owner writes
the parameter in ``settings.local.yaml`` after asking an accountant (RESEARCH.md section 3).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import MISSING, money, pct
from core.config import load_settings
from fiscal import estimates as est
from fiscal import lots as fl
from ingest.local_fx import fx_config
from transform import portfolio

settings = load_settings()
if settings.public_mode:
    st.stop()   # defence in depth: main.py never registers this page in public mode

fiscal_cfg = settings.raw.get("fiscal") or {}
fx_cfg = fx_config(settings)
local = str(fx_cfg.get("currency") or "moneda local")
months = fiscal_cfg.get("holding_period_months")

st.title("🧾 Fiscal")
st.warning(
    "**ESTIMADO — nada de esta página es un impuesto liquidado.** Convierte tu cuenta a "
    f"{local} con la tasa oficial del día de cada operación; si tu contador indica otra fecha "
    "de conversión, estas cifras cambian (pregunta nº 4 de RESEARCH.md §3). No se calcula "
    "ningún impuesto, ni se supone ninguna tasa ni ningún umbral."
)


def local_money(value: float | None) -> str:
    return money(value, public=False, currency=local)


def usd(value: float | None) -> str:
    return money(value, public=False)


# --- Data -------------------------------------------------------------------------------

if not fx_cfg.get("url"):
    st.info(
        "Falta la tasa de cambio oficial de tu moneda. Configura `fiscal.local_fx` en "
        "`config/settings.local.yaml` (la plantilla `.example` explica cada campo) y ejecuta "
        "`python run_ingest.py --only local_fx`."
    )
    st.stop()

fx_rows = app_data.observations(str(fx_cfg.get("source", "local_fx")), app_data.db_mtime())
fx = fl.fx_series(fx_rows, str(fx_cfg.get("series_id", "FX:LOCAL_PER_USD")))
account = app_data.account_observations()
trades = app_data.trades()
if fx.empty or account.empty:
    st.info("Sin tasa o sin cuenta en la base: `python run_ingest.py --only local_fx ibkr`.")
    st.stop()

fx_now = float(fx.iloc[-1])
fx_date = fx.index[-1].date().isoformat()
positions = portfolio.latest_positions(account)
prices = app_data.prices_for(list(positions["ticker"]))
latest = portfolio.latest_prices(prices)
price_map = dict(zip(latest["ticker"], latest["price"]))
open_lots, disposals, unmatched = portfolio.fifo_lots(trades)
# Only lots still open in the latest statement: a lot of a position sold outside the data
# window is not in the account any more.
open_lots = open_lots[open_lots["ticker"].isin(positions["ticker"])]
lots = fl.valued_lots(fl.local_lots(open_lots, fx), price_map, fx_now)

# --- 1. The portfolio in local currency -------------------------------------------------

st.subheader(f"1. Tu cartera en {local}")
st.caption(f"Tasa de hoy: {local_money(fx_now)} por USD (vigente desde el {fx_date}).")
split = est.attribution(lots)
if split is None:
    st.info("Algún lote no tiene tasa de su fecha o precio actual; no se muestra un total "
            "parcial como si fuera el total (§12).")
else:
    cols = st.columns(4)
    cols[0].metric("Costo (congelado)", local_money(split.cost_local))
    cols[1].metric("Valor hoy", local_money(split.value_local))
    cols[2].metric("Resultado", local_money(split.result_local),
                   help=f"En dólares: {usd(split.result_usd)}.")
    cols[3].metric("Tasa media de compra", local_money(split.fx_cost_average))
    cols = st.columns(2)
    cols[0].metric("Parte por los activos", local_money(split.asset_part_local),
                   help="Lo que se movieron las acciones, a tu tasa de compra.")
    cols[1].metric("Parte por la divisa", local_money(split.fx_part_local),
                   help="Lo que movió la tasa sobre lo que tienes hoy.")
    st.caption(
        "Tener todo en dólares y vivir en otra moneda es una **posición larga en dólares del "
        "tamaño de la cartera**, que nadie decidió (§9.9). La segunda fila la separa: una "
        "pérdida en dólares puede ser una ganancia en tu moneda, y al revés."
    )

# --- 2. Open lots -----------------------------------------------------------------------

st.subheader("2. Lotes abiertos")
lots = est.holding_threshold(lots, pd.Timestamp.today(), months)
table = pd.DataFrame({
    "Compra": lots["ts"].str[:10],
    "Valor": lots["ticker"],
    "Cantidad": lots["quantity"].round(4),
    "Costo USD": lots["cost_usd"].map(usd),
    "Tasa de compra": lots["fx_cost"].map(local_money),
    f"Costo {local}": lots["cost_local"].map(local_money),
    f"Valor {local}": lots["value_local"].map(local_money),
    "Parte activo": lots["asset_part_local"].map(local_money),
    "Parte divisa": lots["fx_part_local"].map(local_money),
    "Cumple el periodo": [d if d else "no calculable" for d in lots["threshold_date"]],
})
st.dataframe(table, hide_index=True, width="stretch")
if months is None:
    st.caption("**Periodo de tenencia: no calculable.** Cuántos meses de tenencia cambian el "
               "tratamiento de una ganancia es pregunta para tu contador; escríbelo en "
               "`fiscal.holding_period_months` y esta columna mostrará la fecha de cada lote.")

# --- 3. Sales ---------------------------------------------------------------------------

st.subheader("3. Ventas realizadas")
sold = est.classify_disposals(fl.local_disposals(disposals, fx), months)
if sold.empty:
    st.caption("Sin ventas en los datos.")
else:
    st.dataframe(pd.DataFrame({
        "Venta": sold["ts"].str[:10],
        "Compra": sold["acquired_ts"].str[:10],
        "Valor": sold["ticker"],
        "Días": sold["holding_days"],
        "Resultado USD": sold["result_usd"].map(usd),
        f"Resultado {local}": sold["result_local"].map(local_money),
        "Parte activo": sold["asset_part_local"].map(local_money),
        "Parte divisa": sold["fx_part_local"].map(local_money),
        "Pasó el periodo": ["no calculable" if v is None else ("sí" if v else "no")
                            for v in sold["held_past_threshold"]],
    }), hide_index=True, width="stretch")
    total_local = pd.to_numeric(sold["result_local"], errors="coerce")
    st.caption(
        f"Suma: {usd(float(sold['result_usd'].sum()))} · "
        f"{local_money(float(total_local.sum())) if total_local.notna().all() else MISSING}. "
        "Mismo emparejamiento FIFO que la página de Cartera, verificado contra el de IBKR."
    )
if unmatched:
    st.warning("Ventas sin compra en los datos, fuera de esta tabla: " + "; ".join(unmatched))

# --- 4. Dividends and withholding -------------------------------------------------------

st.subheader("4. Dividendos y retención")
div = est.dividends_local(app_data.cash_transactions(), fx)
if div.empty:
    st.caption("Sin dividendos en los datos.")
else:
    st.dataframe(pd.DataFrame({
        "Fecha": div["date"], "Valor": div["ticker"],
        "Bruto USD": div["gross_usd"].map(usd), "Retención USD": div["withholding_usd"].map(usd),
        f"Neto {local}": div["net_local"].map(local_money),
    }), hide_index=True, width="stretch")
    gross, withheld = float(div["gross_usd"].sum()), float(div["withholding_usd"].sum())
    st.caption(
        f"Retención **observada**: {pct(-withheld / gross if gross else None)} del bruto — el "
        "dato que reporta IBKR, nunca una tasa supuesta (§11). Cómo se declara y si da "
        "derecho a descuento es la pregunta nº 1 para tu contador."
    )

# --- 5. US situs ------------------------------------------------------------------------

st.subheader("5. Activos con *situs* en EE. UU.")
valued = portfolio.valuation(positions, latest)
situs = est.us_situs_value(valued, app_data.securities())
cols = st.columns(2)
cols[0].metric("Valor en emisores de EE. UU.", usd(situs["us_usd"]))
cols[1].metric(f"En {local}", local_money(situs["us_usd"] * fx_now))
if situs["unknown"]:
    st.caption(f"Sin país de emisor conocido, fuera de la suma: {', '.join(situs['unknown'])}.")
st.caption(
    "**Punto a consultar con un asesor, no un cálculo.** Para no residentes puede existir "
    "un impuesto sucesorio estadounidense sobre activos con *situs* en EE. UU.; si aplica, "
    "desde qué monto y cómo se evita no se responde aquí (pregunta nº 2 de RESEARCH.md §3)."
)
