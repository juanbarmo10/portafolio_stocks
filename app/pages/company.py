"""🏢 Empresa — level 3, one company at a time (CLAUDE.md sections 2, 5.2, 8 phase 2).

Everything known about a single company in one place: its written thesis and what would
kill it, the audited fundamentals behind it, how much of the company's success actually
reaches the shareholder, its filings and its earnings calendar.

Two disciplines run through the whole page:

**Point-in-time.** The date control at the top is not decoration. Every figure is computed
from filings public *on that date*, so "what did this look like before the last 10-K?" is a
question the page answers honestly (section 9.4). Fundamentals carry a filing lag of weeks,
and the page says how many.

**Public mode hides the position, not the company.** Microsoft's revenue is in a 10-K
anyone can download; hiding it would be theatre. What disappears publicly is the block
about *your* holding — quantity, cost and value (RESEARCH.md section 2.8).

Sections are deliberately independent so more can be added without touching the rest.
"""

from __future__ import annotations

import datetime as dt
import json

import altair as alt
import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import MISSING, PUBLIC_NOTE, money, pct, reported_amount
from core.config import load_settings
from transform import fundamentals as fun
from transform import portfolio as port
from transform import thesis as th
from transform import value_accrual as va

SERIES_COLORS = ["#2a78d6", "#eb6834"]

settings = load_settings()
public = settings.public_mode
cards = settings.tracked_companies
hurdle = settings.raw.get("panel", {}).get("level3", {}).get("hurdle_rate")

st.title("🏢 Empresa")

# --- No universe yet -----------------------------------------------------------------

if not cards:
    if public:
        st.caption(PUBLIC_NOTE)
    st.warning("No hay ninguna empresa con ficha de tesis escrita.")
    st.markdown(
        """
Esta página analiza una empresa a la vez, y **no se abre sin ficha** (§5.2). No es una
limitación técnica: sin criterio de invalidación escrito no hay forma de saber cuándo
vender, y la posición se sostiene sola por inercia.

Cada ficha vive en `config/settings.local.yaml` y necesita, como mínimo:

| Campo | Qué es |
|---|---|
| `ticker` / `cik` | El CIK es la clave real; el ticker es mutable (§9.3) |
| `thesis` | Una frase: por qué esta empresa gana |
| `value_accrual` | Cómo llega ese resultado **a ti** como accionista |
| `key_metric` | El concepto que hay que vigilar |
| `invalidation` | Qué observación mata la tesis. **Con umbral.** Obligatorio |
| `review_date` | Cuándo toca releerla |

Opcionalmente, `invalidation_rule: { metric, operator, threshold }` para la parte que el
panel puede evaluar solo. Los CIK de tus posiciones ya están resueltos y comentados en el
fichero.

Después: `python run_ingest.py --only sec sec_filings`.
"""
    )
    st.stop()

# --- Controls -------------------------------------------------------------------------

by_ticker = {str(card.get("ticker", "?")): card for card in cards}
left, right = st.columns([1, 1])
with left:
    ticker = st.selectbox("Empresa", sorted(by_ticker), index=0)
with right:
    as_of = st.date_input(
        "Ver la empresa al día",
        value=dt.date.today(),
        max_value=dt.date.today(),
        help="Solo se usa lo que ya estaba presentado a la SEC en esa fecha (§9.4).",
    )

card = by_ticker[ticker]
cik = str(card.get("cik", "")).zfill(10) if card.get("cik") else ""
as_of_iso = as_of.isoformat()

if not cik:
    # Defence in depth: core.config.validate_theses already refuses to load a card with no
    # CIK, naming every missing field, so this branch should be unreachable through the
    # normal path. It stays because a card could arrive from somewhere else one day, and
    # because a page that renders SEC data for an empty key would be worse than an error.
    st.error(
        f"La ficha de **{ticker}** no tiene `cik`. Es la clave real de todo lo que viene "
        "de la SEC; el ticker no sirve porque se reasigna (§9.3)."
    )
    st.stop()

observations = app_data.sec_observations()
filings = app_data.filings()
events = app_data.events()

status = th.status(card, observations, filings, events, as_of_iso,
                   hurdle_rate=hurdle, earnings_window_days=5)
snapshot = fun.snapshot(observations, cik, as_of_iso)
accrual = va.assess(observations, cik, as_of_iso, hurdle_rate=hurdle)

if public:
    st.caption(PUBLIC_NOTE)

st.subheader(f"{ticker} · {card.get('sector') or 'sector sin definir'}")
header = [f"CIK `{cik}`"]
if card.get("thesis_category"):
    header.append(f"categoría de tesis: **{card['thesis_category']}**")
if status.earnings_date:
    kind = "estimada" if status.earnings_is_estimated else "confirmada"
    header.append(f"próximos resultados: **{status.earnings_date}** ({kind})")
if snapshot.stale_days is not None:
    header.append(f"último dato presentado hace {snapshot.stale_days} días")
st.caption(" · ".join(header))

if observations.empty or snapshot.revenue_ttm is None:
    st.info(
        "Sin fundamentales calculables para esta empresa a esa fecha. Ejecuta "
        "`python run_ingest.py --only sec sec_filings`, o revisa si hacen falta cuatro "
        "trimestres contiguos (§9.12). No se estima nada (§12)."
    )

# --- 1. The thesis --------------------------------------------------------------------

st.divider()
st.subheader("1 · La tesis")

for flag in status.flags:
    st.warning(flag)

thesis_column, invalidation_column = st.columns([1, 1])
with thesis_column:
    st.markdown(f"**Tesis.** {card.get('thesis') or '_sin escribir_'}")
    st.markdown(f"**Captura de valor.** {card.get('value_accrual') or '_sin escribir_'}")
    st.markdown(f"**Métrica clave.** `{card.get('key_metric') or '—'}`")
with invalidation_column:
    st.markdown(f"**Criterio de invalidación.** {card.get('invalidation') or '_sin escribir_'}")
    st.caption(
        "Se muestra tal cual lo escribiste. El panel **no lo interpreta ni decide que se "
        "ha cumplido** — para eso lo escribiste tú."
    )
    if status.rule:
        st.markdown(f"**Regla automática.** {status.rule_detail}")
    review = status.review_date or "sin fecha"
    st.markdown(
        f"**Revisión.** {review}"
        + (" — ⚠️ vencida" if status.review_overdue else "")
    )

# --- 2. Fundamentals ------------------------------------------------------------------

st.divider()
st.subheader("2 · Fundamentales auditados")
st.caption(
    "Todo desde XBRL de la SEC, con la fecha de presentación de cada hecho. El cuarto "
    "trimestre se deriva del 10-K porque nadie presenta un 10-Q de Q4 (§9.12)."
)

row = st.columns(4)
row[0].metric("Ingresos (TTM)", reported_amount(snapshot.revenue_ttm))
row[1].metric("Beneficio neto (TTM)", reported_amount(snapshot.net_income_ttm))
row[2].metric("Flujo de caja libre (TTM)", reported_amount(snapshot.fcf_ttm))
row[3].metric(
    "Crecimiento de ingresos",
    pct(snapshot.revenue_growth_yoy),
    help="TTM contra el TTM de un año antes. Geométrico, nunca media de porcentajes (§9.10).",
)

row = st.columns(4)
row[0].metric("Margen operativo", pct(snapshot.operating_margin))
row[1].metric("Margen neto", pct(snapshot.net_margin))
row[2].metric("Margen FCF", pct(snapshot.fcf_margin))
row[3].metric(
    "Conversión a caja",
    f"{snapshot.cash_conversion:.2f}" if snapshot.cash_conversion is not None else MISSING,
    help="FCF / beneficio neto. Por debajo de 1, parte del beneficio contable no llega a "
         "la caja — capex, circulante o contabilidad agresiva.",
)

revenue_series = fun.ttm_series(observations, cik, "revenue", as_of_iso)
fcf_operating = fun.ttm_series(observations, cik, "operating_cash_flow", as_of_iso)
capex_series = fun.ttm_series(observations, cik, "capex", as_of_iso)

if not revenue_series.empty:
    lines = revenue_series.assign(Serie="Ingresos (TTM)")
    if not fcf_operating.empty and not capex_series.empty:
        fcf = fcf_operating.merge(capex_series, on="ts", suffixes=("_ocf", "_capex"))
        fcf["value"] = fcf["value_ocf"] - fcf["value_capex"]
        lines = pd.concat([lines, fcf[["ts", "value"]].assign(Serie="FCF (TTM)")])
    lines["date"] = pd.to_datetime(lines["ts"]).dt.date

    st.altair_chart(
        alt.Chart(lines)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(grid=False)),
            y=alt.Y("value:Q", title="USD", scale=alt.Scale(zero=False),
                    axis=alt.Axis(grid=True, gridOpacity=0.25)),
            color=alt.Color("Serie:N",
                            scale=alt.Scale(domain=["Ingresos (TTM)", "FCF (TTM)"],
                                            range=SERIES_COLORS),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Serie:N"), alt.Tooltip("date:T", title="Trimestre"),
                     alt.Tooltip("value:Q", title="USD", format=",.0f")],
        )
        .interactive(bind_y=False)
        .properties(height=260),
        width="stretch",
    )

# --- 3. Value accrual -----------------------------------------------------------------

st.divider()
st.subheader("3 · ¿Cómo llega eso al accionista?")
st.caption(
    "El concepto rector (§2): una empresa puede crecer 30% al año y destruir valor por "
    "acción si diluye 35%. Esta sección es el análogo de la vista de unlocks."
)

row = st.columns(4)
row[0].metric(
    "Dilución interanual", pct(accrual.dilution_yoy, decimals=2),
    help="Positivo = más acciones para la misma empresa. Negativo = el recuento baja.",
)
row[1].metric("SBC / ingresos", pct(accrual.sbc_over_revenue))
row[2].metric(
    "SBC / FCF", pct(accrual.sbc_over_fcf),
    help="Cuánto del efectivo que genera el negocio ya está comprometido con empleados.",
)
row[3].metric(
    "ROIC (aprox.)", pct(accrual.roic),
    help="NOPAT sobre capital invertido, con la tasa impositiva OBSERVADA en los filings, "
         "no la estatutaria. Capital invertido = patrimonio + deuda largo plazo − caja.",
)

row = st.columns(4)
row[0].metric("Recompras brutas (TTM)", reported_amount(accrual.buybacks_ttm))
row[1].metric("Recompras netas (TTM)", reported_amount(accrual.net_buybacks_ttm),
              help="Recompras − emisión. La bruta engaña si el SBC la anula.")
row[2].metric(
    "Acciones retiradas",
    reported_amount(accrual.shares_removed, unit="acciones")
    if accrual.shares_removed else MISSING,
)
row[3].metric(
    "Coste por acción retirada",
    reported_amount(accrual.buyback_per_share_removed)
    if accrual.buyback_per_share_removed else MISSING,
    help="Recompra neta dividida por la caída real del recuento. Si es mucho mayor que el "
         "precio de mercado, la recompra está compensando dilución, no devolviendo capital.",
)

if accrual.roic is not None:
    if hurdle is None:
        st.caption(
            "Sin tasa exigida configurada, el ROIC no se compara contra nada: escribe "
            "`panel.level3.hurdle_rate` en `settings.local.yaml`. **Desconocido no es "
            "aprobado** (§12)."
        )
    else:
        st.caption(
            f"ROIC {pct(accrual.roic)} frente a la tasa exigida {pct(hurdle)}: "
            f"**{pct(accrual.roic_above_hurdle)}** de diferencia."
        )

shares = fun.known(observations, cik, "diluted_shares", fun.ANNUAL, as_of_iso)
if len(shares) >= 2:
    chart = shares.assign(date=pd.to_datetime(shares["ts"]).dt.date)
    st.markdown("**Acciones diluidas en circulación** — el análogo directo de los unlocks")
    st.altair_chart(
        alt.Chart(chart)
        .mark_line(strokeWidth=2, color=SERIES_COLORS[0], point=True)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(grid=False)),
            y=alt.Y("value:Q", title="acciones", scale=alt.Scale(zero=False),
                    axis=alt.Axis(grid=True, gridOpacity=0.25)),
            tooltip=[alt.Tooltip("date:T", title="Ejercicio"),
                     alt.Tooltip("value:Q", title="Acciones", format=",.0f")],
        )
        .properties(height=200),
        width="stretch",
    )

# --- 4. Filings and calendar ----------------------------------------------------------

st.divider()
st.subheader("4 · Presentaciones y calendario")

company_filings = filings[filings["cik"] == cik] if not filings.empty else pd.DataFrame()
company_events = events[events["cik"] == cik] if not events.empty else pd.DataFrame()

if status.amended_filings:
    st.error(
        "**Estados financieros enmendados.** "
        + ", ".join(f"{f['form']} ({f['filed_date']})" for f in status.amended_filings)
        + ". Un `/A` sobre un 10-K o 10-Q significa que la empresa volvió a presentar "
        "cuentas ya publicadas: bandera de gobernanza **para investigar**, no prueba de "
        "reexpresión (§9.6)."
    )

calendar_column, filings_column = st.columns([1, 1])
with calendar_column:
    st.markdown("**Calendario de resultados**")
    if company_events.empty:
        st.caption("Sin eventos. Ejecuta `python run_ingest.py --only sec_filings`.")
    else:
        upcoming = company_events[company_events["ts"] >= as_of_iso].sort_values("ts")
        past = company_events[company_events["ts"] < as_of_iso].sort_values(
            "ts", ascending=False
        ).head(6)
        for row_ in upcoming.itertuples():
            payload = json.loads(row_.payload) if row_.payload else {}
            if row_.is_estimated:
                st.info(
                    f"**{row_.ts}** — estimado. Método: {payload.get('method', 'n/d')}. "
                    f"Último confirmado: {payload.get('last_confirmed', 'n/d')}."
                )
            else:
                st.success(f"**{row_.ts}** — confirmado.")
        if not past.empty:
            st.caption(
                "Anuncios anteriores (8-K item 2.02): "
                + " · ".join(past["ts"].astype(str))
            )

with filings_column:
    st.markdown("**Últimas presentaciones**")
    if company_filings.empty:
        st.caption("Sin presentaciones ingeridas todavía.")
    else:
        recent = company_filings.sort_values("filed_date", ascending=False).head(12)
        st.dataframe(
            pd.DataFrame({
                "Formulario": recent["form"],
                "Periodo": recent["period_end"],
                "Presentado": recent["filed_date"],
                "Enmienda": recent["is_amended"].map({1: "sí", 0: ""}),
                "Documento": recent["url"],
            }),
            hide_index=True, width="stretch",
            column_config={"Documento": st.column_config.LinkColumn(display_text="abrir")},
        )

# --- 5. Your position (local only) ----------------------------------------------------

if not public:
    st.divider()
    st.subheader("5 · Tu posición")

    account = app_data.account_observations()
    positions = port.latest_positions(account)
    held = positions[positions["ticker"] == ticker] if not positions.empty else pd.DataFrame()

    if held.empty:
        st.caption(
            f"No tienes {ticker} en cartera. Esta empresa está en el universo de "
            "**análisis**, que no es lista de tenencia (§5.1)."
        )
    else:
        prices = app_data.prices_for([ticker])
        valued = port.valuation(held, port.latest_prices(prices))
        position = valued.iloc[0]
        trades = app_data.trades()
        lots, _disposals, _unmatched = port.fifo_lots(
            trades[trades["ticker"] == ticker] if not trades.empty else trades
        )
        costs = port.average_cost(lots)
        avg = float(costs["avg_cost"].iloc[0]) if not costs.empty else None

        row = st.columns(4)
        row[0].metric("Cantidad", f"{position['quantity']:,.4f}".replace(".", ","))
        row[1].metric("Coste medio", money(avg, public=False))
        row[2].metric("Valor", money(position["market_value"], public=False))
        row[3].metric(
            "PnL no realizado", money(position["unrealized_pnl"], public=False),
            delta=pct(position["unrealized_return"]),
        )
