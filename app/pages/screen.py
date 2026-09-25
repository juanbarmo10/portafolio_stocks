"""🔎 Cribado — which companies deserve a thesis (CLAUDE.md §15.1, question 3 of section 2).

Once a quarter, the current S&P 500 plus the researched companies, on audited annual
figures from the SEC (``ingest/screen``, ``transform/screen``). The output is a short list
of companies to **study** on the company page — the screen never says "buy", and nothing on
it is fresher than the last annual report (sections 2, 12).

Private: it marks what is held and what is being studied, which is account information.
It is not in ``PUBLIC_PAGES``, so a public deployment does not register it (RESEARCH.md §2.8).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import MISSING
from core.config import load_settings
from ingest.screen import SOURCE
from transform import portfolio as port
from transform import screen as sc

settings = load_settings()
defaults = dict(settings.raw.get("screen", {}).get("defaults", {}))

st.title("🔎 Cribado")
st.caption(
    "Una vez al trimestre: ¿qué empresas merecen una ficha? El resultado es una lista de "
    "candidatas **para estudiar** en la página de Empresa, no una lista de compra. Cifras "
    "anuales auditadas de la SEC; nada aquí es más reciente que el último informe anual."
)


if not app_data.database_ready():
    st.info("Todavía no hay base de datos. Ejecuta `python run_ingest.py`.")
    st.stop()

table = app_data.screen_view(app_data.db_mtime())
ingested = app_data.source_last_ingest(SOURCE, app_data.db_mtime())
if table.empty:
    st.info("El cribado todavía no se ha descargado. Ejecuta "
            "`python run_ingest.py --only screen --force`; después corre solo cada "
            f"{settings.raw.get('screen', {}).get('run_every_days', 90)} días.")
    st.stop()

year = int(table["fiscal_year"].iloc[0])
st.markdown(f"**Ejercicio CY{year}** · {len(table)} empresas · descargado el {ingested} · "
            f"precios al {table['price_date'].dropna().max() or MISSING}")

# --- Filters ------------------------------------------------------------------------------
st.subheader("Filtros")
st.caption("Puntos de partida de lectura, escritos en `settings.yaml` (`screen.defaults`). "
           "No se optimizaron contra nada; muévelos con criterio, no hasta que salga algo.")
c1, c2, c3, c4 = st.columns(4)


def limit(column: Any, label: str, key: str, *, scale: float, step: float,
          help_text: str) -> float | None:
    on = column.checkbox(label, value=defaults.get(key) is not None, key=f"on_{key}")
    value = column.number_input(label, value=float(defaults.get(key) or 0.0) * scale,
                                step=step, key=key, label_visibility="collapsed",
                                disabled=not on, help=help_text)
    return value / scale if on else None

limits = {
    "max_dilution": limit(c1, "Dilución máx. (%)", "max_dilution", scale=100, step=0.5,
                          help_text="Recuento diluido medio frente al año anterior."),
    "max_sbc_over_revenue": limit(c2, "SBC / ingresos máx. (%)", "max_sbc_over_revenue",
                                  scale=100, step=1.0,
                                  help_text="Remuneración en acciones sobre ingresos."),
    "min_fcf_margin": limit(c3, "Margen FCF mín. (%)", "min_fcf_margin", scale=100,
                            step=1.0, help_text="Flujo de caja libre sobre ingresos."),
    "min_market_cap": limit(c4, "Capitalización mín. (mil M USD)", "min_market_cap",
                            scale=1e-9, step=1.0, help_text="Precio × recuento diluido."),
}
keep_unknown = st.checkbox(
    "Conservar las empresas que un filtro no puede juzgar (dato faltante)", value=False,
    help="Desconocido no es aprobar ni suspender. Por defecto quedan fuera, pero se cuentan.")
result = sc.apply_filters(table, limits, keep_unknown=keep_unknown)

unknown = {k: v for k, v in result.unknown.items() if v}
labels = {"dilution": "dilución", "sbc_over_revenue": "SBC", "fcf_margin": "margen FCF",
          "market_cap": "capitalización"}
line = f"**{len(result.table)}** pasan · {result.failed} no pasan"
if unknown and not keep_unknown:
    line += (" · fuera por falta de dato: "
             + ", ".join(f"{v} ({labels[k]})" for k, v in unknown.items()))
st.markdown(line)

# --- The table ----------------------------------------------------------------------------
held = set()
account = app_data.account_observations()
if not account.empty:
    positions = port.latest_positions(account)
    held = set(positions["ticker"]) if not positions.empty else set()
tracked = {str(c["ticker"]) for c in settings.tracked_companies}
watch = {str(c["ticker"]) for c in settings.watchlist_companies}


def status(ticker: str) -> str:
    marks = []
    if ticker in held:
        marks.append("en cartera")
    if ticker in tracked:
        marks.append("con tesis")
    elif ticker in watch:
        marks.append("en estudio")
    return " · ".join(marks)


shown = result.table.sort_values("fcf_after_sbc_yield", ascending=False, na_position="last")
display = pd.DataFrame({
    "Ticker": shown["ticker"],
    "Empresa": shown["name"],
    "Estado": shown["ticker"].map(status),
    "Ingresos (mil M)": shown["revenue"] / 1e9,
    "Crec. ingresos": shown["revenue_growth"],
    "Margen operativo": shown["operating_margin"],
    "Margen FCF": shown["fcf_margin"],
    "FCF / beneficio": shown["cash_conversion"],
    "SBC / ingresos": shown["sbc_over_revenue"],
    "Dilución": shown["dilution"],
    "Capitalización (mil M)": shown["market_cap"] / 1e9,
    "Rent. FCF": shown["fcf_yield"],
    "Rent. FCF − SBC": shown["fcf_after_sbc_yield"],
    "VE / EBIT": shown["ev_ebit"],
    "VE / ventas": shown["ev_sales"],
    "Sin deuda LP": shown["debt_missing"],
})
percent = st.column_config.NumberColumn(format="percent")
multiple = st.column_config.NumberColumn(format="%.1f×")
st.dataframe(
    display, hide_index=True, width="stretch", height=560,
    column_config={
        "Ingresos (mil M)": st.column_config.NumberColumn(format="%.1f"),
        "Capitalización (mil M)": st.column_config.NumberColumn(format="%.1f"),
        "Crec. ingresos": percent, "Margen operativo": percent, "Margen FCF": percent,
        "SBC / ingresos": percent, "Dilución": percent, "Rent. FCF": percent,
        "Rent. FCF − SBC": st.column_config.NumberColumn(
            format="percent", help="Flujo de caja libre menos SBC, sobre la capitalización. "
                                   "El SBC es un coste real: lo pagan los accionistas en "
                                   "dilución en vez de la empresa en efectivo."),
        "FCF / beneficio": multiple, "VE / EBIT": multiple, "VE / ventas": multiple,
        "Sin deuda LP": st.column_config.CheckboxColumn(
            help="La empresa no presentó ningún concepto de deuda a largo plazo: el valor "
                 "de empresa la cuenta como cero, igual que la página de Empresa."),
    },
)
st.caption("Ordenada por rentabilidad FCF después del SBC; pulsa una cabecera para "
           "reordenar. Una casilla vacía es un dato que no existe o un cociente sobre una "
           "base negativa (un múltiplo sobre pérdidas se leería como barato): nunca una "
           "estimación.")

# --- From the screen to the study -----------------------------------------------------------
st.subheader("Pasar una candidata a estudio")
options = [t for t in shown["ticker"] if t not in tracked | watch]
if options:
    pick = st.selectbox("Candidata", options, index=None, placeholder="Elige un ticker")
    if pick:
        cik = str(shown.loc[shown["ticker"] == pick, "cik"].iloc[0])
        st.markdown("Añádela a `universe.watchlist` en `config/settings.local.yaml` — "
                    "ticker y CIK, **nada más** (§5.1), y el CIK **entre comillas** (§9.13):")
        st.code(f'- ticker: {pick}\n  cik: "{cik}"', language="yaml")
        st.caption("La siguiente ingesta baja sus fundamentales completos, presentaciones y "
                   "calendario, y aparece en la página de Empresa. La ficha de tesis se "
                   "escribe después de mirar los números, no antes.")

with st.expander("Límites del cribado"):
    st.markdown(f"""
- **Periodos de calendario.** La API `frames` de la SEC asigna cada ejercicio al año natural
  con el que más se solapa: un ejercicio de julio a junio aparece como CY{year} o CY{year - 1}.
  Sirve para ordenar, no para cuadrar al dólar — para eso está la página de Empresa.
- **Sin fecha de presentación.** `frames` no la trae: estas cifras sirven para el cribado de
  hoy y **nunca** para un backtest (§9.4).
- **Universo:** el S&P 500 **actual** más las empresas estudiadas y las posiciones. Es una
  lista de hoy, con el sesgo de supervivencia que eso implica si se usara hacia atrás (§9.5)
  — aquí no se usa hacia atrás.
- **Varias clases de acción** (Visa, Hershey, Alphabet…): cuando la empresa presenta el
  recuento por clase y no el total, `frames` no lo trae y la capitalización queda vacía.
  Cuando sí presenta el total, la capitalización usa el precio de una sola clase.
- **Bancos y aseguradoras** no presentan margen operativo ni capex como una industrial: sus
  casillas vacías son de definición, no de datos, y su "flujo de caja libre" no es el de
  una industrial. Lo mismo las que tienen **brazo financiero** (Ford, GM): los préstamos a
  clientes inflan su flujo operativo. Encabezan la ordenación por rentabilidad FCF y no
  deberían: este cribado no sirve para ellas.
- **VE aproximado**: capitalización + deuda a largo plazo − efectivo; sin deuda corriente,
  inversiones financieras ni arrendamientos.
""")
