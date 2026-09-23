"""🔬 Cartera — level 4: what you actually hold (CLAUDE.md section 2, phase 1 points 4-5).

The real portfolio is read from the IBKR Activity Flex Query and nothing else; it is never
typed in by hand (section 5.1). Every figure here is **recomputed** from the stored trades
and prices rather than copied from the broker's own summary, and the reconciliation at the
top is where the two versions collide — that is the acceptance criterion of phase 1, and
the only place a bug in the lot matching would show itself.

In public mode this page renders **no absolute amount**: weights, percentages and an
equity curve rebased to 100, which carry the method without carrying the balance
(``app/format.py``, RESEARCH.md section 2.8).
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import PUBLIC_NOTE, money, pct, rebase_100
from core.config import load_settings
from transform import corporate_actions as reorg
from transform import portfolio

# Categorical slot 1 of the reference palette, as on the macro page.
LINE_COLOR = "#2a78d6"

settings = load_settings()
public = settings.public_mode
tolerance = float(
    settings.raw.get("panel", {}).get("portfolio", {}).get("nav_tolerance", 0.005)
)

st.title("🔬 Cartera")

account = app_data.account_observations()

# --- Nothing to show yet -------------------------------------------------------------

if account.empty:
    if public:
        st.caption(PUBLIC_NOTE)
        st.info("La cartera todavía no está publicada en esta vista.")
        st.stop()

    has_credentials = settings.secret("IBKR_FLEX_TOKEN") and settings.secret(
        "IBKR_FLEX_QUERY_ID"
    )
    if has_credentials:
        st.warning(
            "Credenciales configuradas, pero la base no tiene datos de la cuenta. "
            "Ejecuta `python run_ingest.py --only ibkr`."
        )
    else:
        st.warning("La cuenta de IBKR no está conectada todavía.")
        st.markdown(
            """
Esta página lee la cuenta **en solo lectura**, vía el *Activity Flex Query* del Flex Web
Service. Ese token es estructuralmente incapaz de colocar órdenes o mover fondos: es una
limitación del sistema de IBKR, no una configuración que haya que recordar mantener.

**Para desbloquearla:**

1. Crea un **Activity Flex Query** en formato XML. Las secciones mínimas están en §4.1,
   pero hacen falta más para reconciliar: `Net Asset Value (NAV) Summary in Base`,
   `Change in NAV`, `Corporate Actions`, `Financial Instrument Information` y la casilla
   *Include Currency Rates?* de Configuración general.
2. En la sección *Trades*, **no** actives *Symbol Summary*, *Orders* ni *Asset Class*
   (la de arriba): rompen el parser de `ibflex`.
3. Usa formato de fecha `yyyyMMdd` — el europeo también rompe el parser.
4. Genera el token del **Flex Web Service**, anota su caducidad, y escribe
   `IBKR_FLEX_TOKEN` e `IBKR_FLEX_QUERY_ID` en `config/.env`.
"""
        )
    st.stop()

# --- Data ----------------------------------------------------------------------------

positions = portfolio.latest_positions(account)
prices_frame = app_data.prices_for(list(positions["ticker"]))
observations = pd.concat([account, prices_frame], ignore_index=True)

valued = portfolio.valuation(positions, portfolio.latest_prices(prices_frame))
trades = app_data.trades()
cash = app_data.cash_transactions()
income = portfolio.income_summary(cash, trades)
open_lots, disposals, unmatched = portfolio.fifo_lots(trades)
costs = portfolio.average_cost(open_lots)
reconciliation = portfolio.reconcile_nav(observations, tolerance=tolerance)
nav = portfolio.nav_series(observations)
nav_total = float(nav["value"].iloc[-1]) if not nav.empty else None

if public:
    st.caption(PUBLIC_NOTE)
else:
    st.caption(
        f"Cartera y NAV según el último *statement* de IBKR: "
        f"**{reconciliation.as_of[:10] if reconciliation.as_of else 'sin fecha'}**. "
        "Resolución diaria, panel *pull* (§2)."
    )

# --- 1. Reconciliation ---------------------------------------------------------------

st.subheader("Reconciliación contra el NAV de IBKR")

if reconciliation.agrees is None:
    # Unknown is not agreement, and must not render as a tick (section 12).
    st.info(reconciliation.detail if not public else
            "No hay datos suficientes para reconciliar en esta vista.")
elif reconciliation.agrees:
    st.success(
        (reconciliation.detail if not public else
         f"La valoración propia y la de IBKR coinciden dentro del "
         f"{pct(tolerance)}: diferencia {pct(reconciliation.relative, decimals=3)}.")
        + " Dos cálculos independientes del mismo número."
    )
else:
    st.error(
        (reconciliation.detail if not public else
         f"Discrepancia de {pct(reconciliation.relative, decimals=3)}, por encima del "
         f"{pct(tolerance)} tolerado.")
        + " Se muestra la diferencia; **no se ajusta nada automáticamente** (§9.8): "
        "IBKR marca con su propio feed y el panel guarda cierres de yfinance, así que "
        "la discrepancia es información sobre los datos."
    )

missing_prices = portfolio.unpriced(valued)
if missing_prices:
    st.warning(
        "Sin precio almacenado para " + ", ".join(missing_prices) + ". Esas posiciones "
        "salen vacías y **los pesos de abajo son sobre el resto** — no se estima un "
        "precio (§12). Ejecuta `python run_ingest.py --only prices`."
    )
if unmatched:
    st.warning(
        "Ventas sin compra en el histórico descargado: " + "; ".join(unmatched) + ". "
        "No se les asigna coste cero, así que su PnL realizado no se puede afirmar."
    )

# Section 9.2: an unrecognized or contradicted corporate action puts the quantity and the
# cost base of that position in doubt, so it is raised here — next to the other reasons a
# number below might not mean what it says — and never resolved automatically.
for review in reorg.review_positions(app_data.corporate_actions(), positions):
    st.error(
        f"**{review.ticker} — revisar.** "
        + " ".join(finding.detail for finding in review.reviews)
    )

notes = reorg.data_quality_notes(app_data.corporate_actions())
if notes:
    with st.expander(f"Notas de calidad de datos ({len(notes)})"):
        st.caption(
            "Sobre valores que **no** tienes en cartera: referencias de mercado, sobre "
            "todo. No bloquean nada; están aparte para que una bandera permanente no "
            "entrene a ignorar las que sí importan."
        )
        for note in notes:
            st.markdown(f"- **{note.ticker}** ({note.ex_date}): {note.detail}")

# --- 2. Positions --------------------------------------------------------------------

st.subheader("Posiciones")

table = valued.merge(costs[["ticker", "avg_cost"]], on="ticker", how="left")
if public:
    display = pd.DataFrame({
        "Posición": table["ticker"],
        "Peso": table["weight"],
        "Precio": table["price"],
        "Coste medio": table["avg_cost"],
        "Retorno no realizado": table["unrealized_return"],
        "Fecha precio": table["price_ts"].str[:10],
    })
    column_config = {
        "Peso": st.column_config.NumberColumn(format="percent"),
        "Retorno no realizado": st.column_config.NumberColumn(format="percent"),
        "Precio": st.column_config.NumberColumn(format="%.2f"),
        "Coste medio": st.column_config.NumberColumn(format="%.2f"),
    }
else:
    display = pd.DataFrame({
        "Posición": table["ticker"],
        "Cantidad": table["quantity"],
        "Coste medio": table["avg_cost"],
        "Precio": table["price"],
        "Valor": table["market_value"],
        "PnL no realizado": table["unrealized_pnl"],
        "Retorno": table["unrealized_return"],
        "Peso": table["weight"],
        "Fecha precio": table["price_ts"].str[:10],
    })
    column_config = {
        "Cantidad": st.column_config.NumberColumn(format="%.4f"),
        "Coste medio": st.column_config.NumberColumn(format="%.2f"),
        "Precio": st.column_config.NumberColumn(format="%.2f"),
        "Valor": st.column_config.NumberColumn(format="%.2f"),
        "PnL no realizado": st.column_config.NumberColumn(format="%+.2f"),
        "Retorno": st.column_config.NumberColumn(format="percent"),
        "Peso": st.column_config.NumberColumn(format="percent"),
    }

st.dataframe(display, hide_index=True, width="stretch", column_config=column_config)

# In public mode the quantity is left out on purpose: prices are public, so a quantity is
# a position size in disguise.

price_dates = sorted({d for d in table["price_ts"].dropna().str[:10]})
if price_dates and reconciliation.as_of and price_dates[-1] != reconciliation.as_of[:10]:
    # The two totals on this page are taken on two different days on purpose, and a reader
    # who spots the gap deserves the reason rather than a suspicion.
    st.caption(
        f"Valorado con el último cierre almacenado (**{price_dates[-1]}**). La "
        f"reconciliación de arriba compara ambos lados en la fecha del *statement* "
        f"(**{reconciliation.as_of[:10]}**), que es lo único comparable con IBKR: usar "
        "el precio de hoy contra un NAV de ayer fabricaría una discrepancia."
    )

# --- 3. Drift against the written target ---------------------------------------------

st.subheader("Drift contra el objetivo")

target_weights = settings.raw.get("portfolio", {}).get("target_weights", {}) or {}
drift = portfolio.target_drift(valued, target_weights)

if drift.empty:
    st.info(
        "No hay pesos objetivo escritos en `config/settings.local.yaml`. Sin objetivo el "
        "drift no significa nada, así que no se muestra un cero que parecería equilibrio."
    )
else:
    st.dataframe(
        pd.DataFrame({
            "Posición": drift["ticker"],
            "Peso actual": drift["weight"],
            "Objetivo": drift["target_weight"],
            "Drift": drift["drift"],
        }),
        hide_index=True, width="stretch",
        column_config={
            "Peso actual": st.column_config.NumberColumn(format="percent"),
            "Objetivo": st.column_config.NumberColumn(format="percent"),
            "Drift": st.column_config.NumberColumn(
                format="percent", help="Actual − objetivo. Negativo = falta comprar.",
            ),
        },
    )

# --- 4. Concentration by thesis (section 5.3) ----------------------------------------

st.subheader("Concentración por categoría de tesis")

metadata = portfolio.thesis_metadata(settings.tracked_companies)
grouped = portfolio.concentration(valued, metadata)

st.caption(
    "La diversificación real es por **modo de fallo**, no por número de tickers: ocho "
    "empresas de cinco sectores GICS pueden ser una sola apuesta a tipos bajos (§5.3)."
)
if grouped.empty:
    st.info("Sin posiciones que agrupar.")
else:
    columns = {"Categoría": grouped["group"], "Peso": grouped["weight"],
               "Posiciones": grouped["positions"], "Tickers": grouped["tickers"]}
    if not public:
        columns["Valor"] = grouped["market_value"]
    st.dataframe(
        pd.DataFrame(columns), hide_index=True, width="stretch",
        column_config={
            "Peso": st.column_config.NumberColumn(format="percent"),
            "Valor": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    if "sin clasificar" in set(grouped["group"]):
        st.info(
            "Hay posiciones **sin ficha de tesis** (§5.2). Son justo aquellas cuyo modo "
            "de fallo nadie ha escrito: añádelas a `universe.tracked` en "
            "`config/settings.local.yaml`."
        )

# --- 5. Equity curve -----------------------------------------------------------------

st.subheader("Patrimonio de la cuenta")

if nav.empty:
    st.info("El *statement* no trae la serie de NAV.")
else:
    curve = pd.DataFrame({"date": pd.to_datetime(nav["ts"]).dt.date})
    if public:
        # Rebasing is what makes this publishable: the curve is identical whatever the
        # account is worth, so it carries no size information (RESEARCH.md section 2.8).
        curve["value"] = rebase_100(nav["value"]).to_numpy()
        y_title, value_format = "Base 100", ".1f"
    else:
        curve["value"] = nav["value"].to_numpy()
        y_title, value_format = "USD", ".2f"

    st.altair_chart(
        alt.Chart(curve)
        .mark_line(strokeWidth=2, color=LINE_COLOR)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(grid=False)),
            y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False),
                    axis=alt.Axis(grid=True, gridOpacity=0.25)),
            tooltip=[alt.Tooltip("date:T", title="Fecha"),
                     alt.Tooltip("value:Q", title=y_title, format=value_format)],
        )
        .interactive(bind_y=False)
        .properties(height=240),
        width="stretch",
    )
    st.caption(
        "Incluye aportes: una curva que sube porque entró dinero no es rentabilidad. "
        "El desglose aporte-vs-rendimiento llega con la atribución de la fase 5."
    )

# --- 6. Income, costs and realized PnL ------------------------------------------------

st.subheader("Rendimiento realizado, ingresos y costes")

drag = portfolio.cost_drag(income, nav_total)
realized = portfolio.realized_pnl(trades)

if public:
    left, right = st.columns(2)
    left.metric("Comisiones + gastos sobre patrimonio", pct(drag, decimals=2) if drag else "—")
    right.metric(
        "Retención observada sobre dividendos",
        pct(income.effective_withholding_rate, decimals=1),
    )
else:
    columns = st.columns(4)
    columns[0].metric("PnL realizado (FIFO)", money(
        float(realized["realized_pnl"].sum()) if not realized.empty else 0.0, public=False))
    columns[1].metric("Dividendos netos", money(income.dividends_net, public=False),
                      help=f"Brutos {money(income.dividends_gross, public=False)}, "
                           f"retención {money(income.withholding, public=False)}.")
    columns[2].metric("Comisiones", money(income.commissions, public=False))
    columns[3].metric("Comisiones + gastos / NAV", pct(drag, decimals=2) if drag else "—")

    if not realized.empty:
        st.dataframe(
            pd.DataFrame({
                "Posición": realized["ticker"],
                "PnL realizado": realized["realized_pnl"],
                "Cantidad vendida": realized["quantity_sold"],
                "Ventas": realized["disposals"],
            }),
            hide_index=True, width="stretch",
            column_config={
                "PnL realizado": st.column_config.NumberColumn(format="%+.2f"),
                "Cantidad vendida": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    st.caption(
        f"Retención **observada** sobre dividendos: "
        f"{pct(income.effective_withholding_rate)}. Es el dato que reporta IBKR, no una "
        "tasa supuesta; qué significa fiscalmente es pregunta para el contador (§11). "
        "El PnL realizado sale del motor FIFO propio, no del resumen del bróker — "
        "coinciden, y esa coincidencia es la prueba."
    )
