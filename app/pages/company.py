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
from app.format import (
    altair_chart,
    MISSING,
    PUBLIC_NOTE,
    compact_amount,
    money,
    number,
    pct,
    quantity,
    reported_amount,
)
from core.config import load_settings
from transform import bcb
from transform import filing_signals as fs
from transform import per_share
from transform import fundamentals as fun
from transform import portfolio as port
from transform import thesis as th
from transform import price_action as pa
from transform import quarterly as qt
from transform import screen as sc
from transform import valuation as val
from transform.adjustments import split_adjusted, total_return_index
from transform import value_accrual as va
from transform.macro import latest as macro_latest

SERIES_COLORS = ["#2a78d6", "#eb6834"]
MULTIPLE_LABELS = {"ev_sales": "EV / ventas", "ev_ebit": "EV / EBIT", "pe": "P / E",
                   "p_fcf": "P / FCF", "fcf_yield": "Rendimiento FCF"}

settings = load_settings()
public = settings.public_mode
tracked = settings.tracked_companies
watchlist = settings.watchlist_companies
cards = [*tracked, *watchlist]
if public and not cards:
    # The public deployment has no settings.local.yaml. Its list is the ``companies`` table
    # of the public copy, which run_public_sync.py fills with the researched companies only
    # — never a position held without a card — and with no thesis text: numbers, not
    # opinions. So every one of them shows as a company under study.
    published = app_data.companies()
    watchlist = [{"ticker": t, "cik": c} for t, c in zip(published["ticker"], published["cik"])]
    cards = watchlist
# Tickers with no written thesis. This set is what section 1 branches on, and it is the
# only thing on the page that behaves differently — a candidate gets the same audited
# numbers as anything else, because the numbers are what you study it with (section 5.1).
under_study = {str(entry.get("ticker")) for entry in watchlist}
level3 = settings.raw.get("panel", {}).get("level3", {})
hurdle = level3.get("hurdle_rate")
earnings_window = int(level3.get("earnings_window_days", th.EARNINGS_WINDOW_DAYS))

st.title("🏢 Empresa")

# --- No universe yet -----------------------------------------------------------------

if not cards:
    if public:
        st.caption(PUBLIC_NOTE)
    st.warning("No hay ninguna empresa escrita todavía, ni en estudio ni con tesis.")
    st.markdown(
        """
Hay **dos formas** de que una empresa llegue a esta página, y la diferencia importa.

### Para empezar a estudiar una: `watchlist`

Ticker y CIK, nada más:

```yaml
universe:
  watchlist:
    - ticker: HIMS
      cik: "0001773751"      # entre comillas SIEMPRE (§9.13)
```

Con eso el panel baja sus fundamentales auditados y te los enseña. No opina — no tiene con
qué. Es la herramienta con la que haces la investigación.

### Cuando la termines: `tracked`

Ahí la ficha sí es completa, y es lo que activa la vigilancia:

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

El orden importa: **la ficha se escribe después de mirar los números**, no antes. Por eso
existe `watchlist` — un criterio de invalidación inventado para poder guardar la ficha no
protege de nada.

Después, en cualquiera de los dos casos: `python run_ingest.py --only sec sec_filings`.
"""
    )
    st.stop()

# --- Controls -------------------------------------------------------------------------

by_ticker = {str(card.get("ticker", "?")): card for card in cards}
left, right = st.columns([1, 1])
with left:
    ticker = st.selectbox(
        "Empresa", sorted(by_ticker), index=0,
        format_func=lambda t: f"{t} · en estudio" if t in under_study else t,
    )
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
                   hurdle_rate=hurdle, earnings_window_days=earnings_window)
snapshot = fun.snapshot(observations, cik, as_of_iso)
accrual = va.assess(observations, cik, as_of_iso, hurdle_rate=hurdle)

if public:
    st.caption(PUBLIC_NOTE)

# The SEC's registered name from the companies registry, and the sector only when a card
# wrote one (the SIC code is not GICS, section 5.3). It read "HIMS · sector sin definir"
# for every company under study until 2026-09-25.
registry = app_data.companies()
registered = registry.loc[registry["cik"] == cik, "name"] if not registry.empty else []
title = [ticker]
if len(registered) and registered.iloc[0]:
    title.append(str(registered.iloc[0]))
if card.get("sector"):
    title.append(str(card["sector"]))
st.subheader(" · ".join(title))
header = [f"CIK `{cik}`"]
if card.get("thesis_category"):
    header.append(f"categoría de tesis: **{card['thesis_category']}**")
if status.earnings_date:
    kind = "estimada" if status.earnings_is_estimated else "confirmada"
    header.append(f"próximos resultados: **{status.earnings_date}** ({kind})")
if snapshot.stale_days is not None:
    header.append(f"último dato presentado hace {snapshot.stale_days} días")
st.caption(" · ".join(header))

# Why fundamentals are missing decides what to say (all three seen 2026-09-25): a
# foreign issuer files IFRS on 20-F/6-K (Nu Holdings), a pre-revenue company files no
# revenue at all (Nautilus), and only otherwise is it a data gap to fix.
company_forms = set(filings.loc[filings["cik"] == cik, "form"]) if not filings.empty else set()
foreign_issuer = bool(company_forms & {"20-F", "20-F/A", "6-K", "40-F"}) and \
    not company_forms & {"10-K", "10-Q"}
has_facts = not observations.empty and observations["series_id"].str.startswith(cik).any()
if foreign_issuer and not has_facts:
    st.info(
        "**Emisor extranjero**: presenta un 20-F anual y 6-K trimestrales, con normas **IFRS** "
        "en vez de US GAAP. El panel solo lee US GAAP, y los 6-K no llevan XBRL: aquí no hay "
        "fundamentales auditados que mostrar. Precio, valoración de mercado e interés corto "
        "sí; las cuentas, en sus informes trimestrales."
    )
elif has_facts and snapshot.revenue_ttm is None:
    revenue_q = fun.known(observations, cik, "revenue", fun.QUARTER, as_of_iso)
    sold = revenue_q[revenue_q["value"] > 0]
    first = (f" Presentó ingresos por primera vez el trimestre cerrado el "
             f"{str(sold['ts'].iloc[0])[:10]} ({reported_amount(float(sold['value'].iloc[0]))});"
             " menos de cuatro trimestres, así que no hay cifra de doce meses."
             if not sold.empty else "")
    st.info(
        "**Sin ingresos de doce meses**: habitual en una biotecnológica o tecnológica que "
        f"aún no vende, o que acaba de empezar.{first} Márgenes y múltiplos sobre ventas no "
        "existen todavía; lo que manda es la **caja**: cuánta tiene, cuánto quema y cuánto "
        "le dura (balance, abajo), y cuánto diluye."
    )
elif observations.empty or snapshot.revenue_ttm is None:
    st.info(
        "Sin fundamentales calculables para esta empresa a esa fecha. Ejecuta "
        "`python run_ingest.py --only sec sec_filings`, o revisa si hacen falta cuatro "
        "trimestres contiguos (§9.12). No se estima nada (§12)."
    )

# --- 1. The thesis --------------------------------------------------------------------

st.divider()
st.subheader("1 · En estudio" if ticker in under_study else "1 · La tesis")

# The mechanical flags apply to a candidate exactly as they do to a holding: an amended
# 10-K is a governance flag whether or not anyone has written a thesis yet, and results
# inside the blackout window are still results inside the blackout window.
for flag in status.flags:
    st.warning(flag)

if ticker in under_study and public:
    # The visitor is not the owner: no "your list", and no instructions for editing a
    # config file they do not have (the portfolio page drops its operator notes the same way).
    st.info(
        f"**{ticker}: empresa en estudio.** Cifras auditadas completas, tal como se "
        "presentaron a la SEC. No hay tesis publicada: el panel no opina sobre ella (§5.2)."
    )
elif ticker in under_study:
    st.info(
        f"**{ticker} está en tu lista de estudio, sin ficha de tesis.** Debajo tienes sus "
        "cifras auditadas completas — que es con lo que se estudia una empresa. Lo que el "
        "panel **no** hace es opinar: sin criterio de invalidación escrito no hay nada que "
        "evaluar, así que no entra al tablero de tesis ni cuenta como universo (§5.2)."
    )
    st.markdown(
        """
**Cuando termines la investigación**, mueve la entrada de `watchlist` a `tracked` en
`config/settings.local.yaml` y complétala. Ahí es donde el panel empieza a vigilarla por ti:

| Campo | Qué es |
|---|---|
| `thesis` | Una frase: por qué esta empresa gana |
| `value_accrual` | Cómo llega ese resultado **a ti** como accionista |
| `key_metric` | El concepto que hay que vigilar |
| `invalidation` | Qué observación mata la tesis. **Con umbral.** Obligatorio |
| `review_date` | Cuándo toca releerla |

El orden importa: la ficha se escribe **después de mirar los números**, no antes. Un
criterio de invalidación inventado para poder guardar la ficha no protege de nada.
"""
    )
else:
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
row[0].metric("Ingresos (TTM, USD)", compact_amount(snapshot.revenue_ttm),
              help=reported_amount(snapshot.revenue_ttm))
row[1].metric("Beneficio neto (TTM, USD)", compact_amount(snapshot.net_income_ttm),
              help=reported_amount(snapshot.net_income_ttm))
row[2].metric("Flujo de caja libre (TTM, USD)", compact_amount(snapshot.fcf_ttm),
              help=reported_amount(snapshot.fcf_ttm))
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

# Una conversión a caja sobre beneficio negativo no significa nada, y computada saldría
# NEGATIVA justo cuando el hecho es bueno: caja positiva pese a pérdida contable. Se dice
# con palabras (§12).
if (
    snapshot.cash_conversion is None
    and snapshot.fcf_ttm is not None
    and snapshot.fcf_ttm > 0
    and snapshot.net_income_ttm is not None
    and snapshot.net_income_ttm <= 0
):
    st.caption(
        f"**Conversión a caja no definida:** la empresa reporta pérdida contable "
        f"({reported_amount(snapshot.net_income_ttm)}) y **flujo de caja libre positivo** "
        f"({reported_amount(snapshot.fcf_ttm)}). Un cociente sobre base negativa saldría "
        "negativo justo cuando el hecho es favorable, así que no se muestra. El dato es "
        "que genera caja sin dar beneficio — mira qué partidas lo explican."
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

    altair_chart(
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

# Quarter by quarter (§15.1.3): the TTM flattens the trend and the operating leverage,
# which is what an analyst reads first.
table = qt.quarter_table(observations, cik, as_of_iso, quarters=8)
if not table.empty:
    st.markdown("**Trimestre a trimestre**")
    lines = [
        ("Ingresos", "revenue", compact_amount),
        ("Crecimiento interanual", "revenue_growth", pct),
        ("Margen bruto", "gross_margin", pct),
        ("I+D / ingresos", "rd_share", pct),
        ("Marketing / ingresos", "marketing_share", pct),
        ("Margen operativo", "operating_margin", pct),
        ("Margen neto", "net_margin", pct),
        ("Margen FCF", "fcf_margin", pct),
        ("SBC / ingresos", "sbc_share", pct),
    ]
    # A line no quarter has (HIMS files its R&D under its own tag, outside companyfacts)
    # is left out instead of shown as a row of dashes.
    shown = [(label, key, fmt) for label, key, fmt in lines if table[key].notna().any()]
    # Metrics as rows, quarters as columns, each cell formatted by its row's formatter.
    grid = pd.DataFrame(
        [[fmt(None if pd.isna(v) else float(v)) for v in table[key]]
         for _, key, fmt in shown],
        index=[label for label, _, _ in shown], columns=list(table["quarter_end"]),
    )
    st.dataframe(grid, width="stretch")
    missing = [label for label, key, _ in lines if table[key].isna().all()]
    st.caption(
        "Cada columna es un trimestre tal como se conocía a la fecha elegida; los Q4 y los "
        "trimestres escondidos en acumulados se derivan (§9.12, §9.14). El crecimiento es "
        "**interanual** — contra el mismo trimestre del año anterior, no contra el anterior, "
        "que mezclaría la estacionalidad. Margen bruto solo si la empresa presenta "
        "`GrossProfit`: restar el coste de ventas no da lo mismo en todas."
        + (f" No presentadas en XBRL estándar: {', '.join(missing)}." if missing else "")
    )

sheet = qt.balance(observations, cik, as_of_iso)
if sheet.assets is not None:
    st.markdown(f"**Balance** · al {sheet.balance_date}")
    row = st.columns(4)
    row[0].metric("Caja e inversiones a corto (USD)", compact_amount(sheet.liquidity),
                  help=f"Efectivo {reported_amount(sheet.cash)}; inversiones a corto "
                       f"{reported_amount(sheet.short_term_investments)}.")
    row[1].metric("Deuda (USD)", compact_amount(sheet.debt),
                  help=f"A corto {reported_amount(sheet.debt_current)}; a largo "
                       f"{reported_amount(sheet.long_term_debt)}, convertibles incluidos: "
                       "un convertible es deuda y, si convierte, dilución futura.")
    row[2].metric("Caja neta (USD)", compact_amount(sheet.net_cash),
                  help="Caja e inversiones menos deuda. Negativa = más deuda que caja.")
    row[3].metric("Patrimonio (USD)", compact_amount(sheet.equity),
                  help=f"Activos {reported_amount(sheet.assets)} − pasivos "
                       f"{reported_amount(sheet.liabilities)}.")
    row = st.columns(4)
    row[0].metric("Deuda / patrimonio", number(fun.ratio(sheet.debt, sheet.equity))
                  if sheet.equity and sheet.equity > 0 else MISSING,
                  help="Sin definir con patrimonio negativo o nulo.")
    row[1].metric("Cobertura de intereses", number(sheet.interest_coverage, decimals=1),
                  help="Beneficio operativo TTM / gasto por intereses TTM: cuántas veces "
                       "cubre el negocio lo que paga por su deuda. Vacía si no presenta "
                       "intereses o pierde dinero.")
    row[2].metric("Quema de caja (TTM, USD)", compact_amount(sheet.cash_burn_ttm),
                  help="Flujo de caja libre negativo de los últimos cuatro trimestres. Vacía "
                       "si la empresa genera caja.")
    row[3].metric("Autonomía de caja", f"{number(sheet.runway_years, decimals=1)} años"
                  if sheet.runway_years is not None else MISSING,
                  help="Caja e inversiones / quema anual: cuánto dura la caja al ritmo del "
                       "último año antes de tener que conseguir dinero — que, para quien quema "
                       "caja, suele significar emitir acciones (dilución) o deuda.")
    notes = ["Solo cifras del último balance presentado: una partida que no aparece en él "
             "queda vacía en vez de arrastrar la de un balance anterior."]
    if sheet.debt is None and sheet.noncurrent_liabilities:
        notes.append(
            f"⚠️ **Ningún concepto estándar de deuda** en este balance, pero el pasivo no "
            f"corriente es de **{reported_amount(sheet.noncurrent_liabilities)}**. Hay "
            "empresas que presentan su deuda con etiquetas propias, que la SEC no normaliza; "
            "vacía no significa sin deuda — mira el balance del 10-Q.")
    if sheet.equity is not None and sheet.equity < 0:
        notes.append("⚠️ **Patrimonio negativo:** los pasivos superan a los activos.")
    st.caption(" ".join(notes))

# Against its industry (§15.1.4): without a comparison, "SBC 6 % of revenue" says nothing.
# The group is the quarterly screen's (S&P 500 + researched) sharing the SIC code; the
# public copy carries no screen, so publicly the block does not appear.
screen_table = app_data.screen_view(app_data.db_mtime())
if not screen_table.empty:
    registry_now = app_data.companies()
    comparison = sc.peer_comparison(
        screen_table, cik, dict(zip(registry_now["cik"], registry_now["sic"])),
        dict(zip(registry_now["cik"], registry_now["sic_description"])))
    if comparison is None:
        if not public:
            st.caption("Sin grupo de comparación: su código de industria (SIC) no reúne "
                       f"{sc.MIN_PEERS} empresas en el cribado, ni con 3 o 2 dígitos.")
    else:
        multiples = {"cash_conversion", "ev_sales"}
        st.markdown(f"**Frente a su sector** · SIC {comparison.code}"
                    + (f" ({comparison.description})" if comparison.description else
                       f" (primeros {comparison.digits} dígitos)")
                    + f" · {len(comparison.peers)} empresas")
        rows = comparison.rows
        st.dataframe(pd.DataFrame({
            "Métrica": rows["label"],
            ticker: [(number(v, decimals=1) + "×" if m in multiples else pct(v))
                     if v is not None and not pd.isna(v) else MISSING
                     for m, v in zip(rows["metric"], rows["value"])],
            "Mediana del grupo": [(number(v, decimals=1) + "×" if m in multiples else pct(v))
                                  if v is not None and not pd.isna(v) else MISSING
                                  for m, v in zip(rows["metric"], rows["median"])],
            "Por encima de": [f"{pct(p, decimals=0)} del grupo"
                              if p is not None and not pd.isna(p) else MISSING
                              for p in rows["percentile"]],
        }), hide_index=True, width="stretch")
        st.caption(
            f"Cifras anuales del cribado (CY{int(screen_table['fiscal_year'].iloc[0])}). "
            "El grupo son empresas **del S&P 500** con el mismo código de industria de la SEC "
            f"({', '.join(comparison.peers[:10])}{'…' if len(comparison.peers) > 10 else ''}): "
            "para una empresa pequeña son los **grandes del sector**, no sus pares de tamaño. "
            "«Por encima de» dice dónde cae, no si es bueno: más SBC o más crecimiento no se "
            "leen en la misma dirección."
        )

# The supervisor's figures, for a bank the SEC cannot read (RESEARCH.md §2.43): Nu Holdings
# files IFRS without quarterly XBRL, and the Banco Central do Brasil publishes its Brazilian
# conglomerate every quarter. Shown for any company with an institution in sources.bcb.
bcb_institution = next((i for i in settings.source("bcb").get("institutions", [])
                        if str(i.get("ticker")) == ticker), None)
if bcb_institution:
    supervisor = bcb.assess(app_data.observations("bcb_ifdata", app_data.db_mtime()),
                            str(bcb_institution["code"]), as_of_iso)
    if supervisor is not None:
        dated = "observada: la primera vez que el panel la vio" if supervisor.observed else \
            "derivada: cierre + 120 días, tarde a propósito"
        st.markdown(f"**Datos del supervisor · Banco Central do Brasil** · trimestre al "
                    f"{supervisor.quarter} (fecha de publicación {dated})")
        row = st.columns(4)
        row[0].metric("Cartera de crédito (R$)", compact_amount(supervisor.credit_portfolio),
                      delta=None if supervisor.credit_growth is None
                      else f"{pct(supervisor.credit_growth)} interanual",
                      help=f"Base: {supervisor.credit_basis}. El crecimiento solo se calcula "
                           "dentro de la misma base: en 2025 cambió la norma (Res. 4.966).")
        row[1].metric("Clientes con crédito activo", compact_amount(supervisor.credit_clients))
        row[2].metric("Índice de Basilea", pct(supervisor.basel),
                      help="Capital regulatorio sobre activos ponderados por riesgo.")
        row[3].metric("Capital principal", pct(supervisor.cet1))
        row = st.columns(4)
        row[0].metric("Activos problemáticos", pct(supervisor.problem_share),
                      help="Sobre la exposición total. Definición del regulador (Res. 4.966): "
                           "más de 90 días de atraso, reestructurados o con indicios de no "
                           "recuperarse. NO es la morosidad 90+ que publica la empresa.")
        row[1].metric("Inadimplência (BCB)", pct(supervisor.delinquent_share),
                      help="Sobre la exposición total, con la etiqueta y definición del BCB.")
        row[2].metric("Beneficio del trimestre (R$)",
                      compact_amount(supervisor.net_income_quarter),
                      help="El BCB lo publica acumulado en el semestre; el trimestre se deriva "
                           "(2T = semestre − 1T).")
        row[3].metric("ROE anualizado (aprox.)", pct(supervisor.roe_annualized),
                      help="4 × beneficio del trimestre / patrimonio del conglomerado en Brasil. "
                           "No es el ROE del grupo que publica la empresa.")
        credit = supervisor.history.dropna(subset=["credit"])
        if len(credit) > 2:
            altair_chart(
                alt.Chart(credit.assign(date=pd.to_datetime(credit["date"]))).mark_bar().encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("credit:Q", title="Cartera de crédito, R$"),
                    color=alt.Color("basis:N", legend=alt.Legend(orient="top", title="Base"),
                                    scale=alt.Scale(range=["#9aa5b1", SERIES_COLORS[0]])),
                    tooltip=[alt.Tooltip("date:T", title="Trimestre"),
                             alt.Tooltip("credit:Q", title="R$", format=",.0f"),
                             alt.Tooltip("basis:N", title="Base")],
                ).properties(height=200),
                width="stretch",
            )
        sgs = app_data.observations("bcb_sgs", app_data.db_mtime())
        selic = bcb.series(sgs, "BR:selic_target", as_of_iso)
        brl = bcb.series(app_data.fred_observations(), "DEXBZUS", as_of_iso)
        context = []
        if not selic.empty:
            year_ago = selic[selic.index <= pd.Timestamp(as_of_iso) - pd.DateOffset(years=1)]
            context.append(f"meta Selic **{number(selic.iloc[-1])} %**"
                           + (f" (hace un año {number(year_ago.iloc[-1])} %)"
                              if len(year_ago) else ""))
        if not brl.empty:
            year_ago = brl[brl.index <= pd.Timestamp(as_of_iso) - pd.DateOffset(years=1)]
            context.append(f"**{number(brl.iloc[-1])} reales por dólar**"
                           + (f" ({pct(brl.iloc[-1] / year_ago.iloc[-1] - 1)} en un año; "
                              "positivo = el real se debilitó)" if len(year_ago) else ""))
        st.caption(
            ("Contexto: " + " · ".join(context) + ". " if context else "")
            + "Normas contables brasileñas (COSIF), en reales y **solo Brasil**: no cuadra con "
            "los 6-K en IFRS y no debe. Es la mirada del supervisor, que la empresa no elige."
        )

# Growth per share (§5) is also the reference for the implied return (§3): computed once.
PS = dict(settings.raw.get("panel", {}).get("per_share") or {})
growth_ps = per_share.assess(observations, cik, as_of_iso,
                             horizons=tuple(PS.get("horizons", (3, 5))),
                             jump_threshold=float(PS.get("jump_threshold",
                                                         per_share.JUMP_THRESHOLD)))

# --- 3. Valuation ---------------------------------------------------------------------

VAL = settings.raw.get("panel", {}).get("valuation", {})
RDCF = VAL.get("reverse_dcf", {})


def multiple(value: float | None) -> str:
    return MISSING if value is None or pd.isna(value) else f"{number(value, decimals=1)}×"


st.divider()
st.subheader("3 · ¿A qué precio?")
st.caption(
    "Lo que el mercado paga hoy por lo de arriba, con el cierre del día y las cifras "
    "presentadas hasta entonces (point-in-time). Un múltiplo sobre una base negativa no se "
    "muestra: un P/E de −30 parecería barato justo cuando la empresa pierde dinero."
)
years = int(VAL.get("history_years", 5))
now, hist = app_data.valuation_view(cik, ticker, as_of_iso, years, app_data.db_mtime())
treasury = macro_latest(app_data.fred_observations(), "DGS10", pd.Timestamp(as_of_iso)).value

if now.price is None or now.market_cap is None:
    st.info("Sin precio o sin recuento de acciones para esta fecha: la valoración necesita "
            "los dos (`python run_ingest.py --only prices sec`).")
else:
    row = st.columns(4)
    row[0].metric("Precio (USD)", number(now.price), help=f"Cierre crudo del {now.price_date}.")
    row[1].metric("Capitalización (USD)", compact_amount(now.market_cap),
                  help=f"Precio × acciones diluidas medias del periodo cerrado el "
                       f"{now.shares_date} ({compact_amount(now.shares)}), ajustadas por "
                       "splits posteriores. No es el recuento de portada: las empresas con "
                       "varias clases de acciones no lo publican en los datos de la SEC.")
    row[2].metric("EV aprox. (USD)",
                  "incompleto" if now.debt_unidentified else compact_amount(now.enterprise_value),
                  help="Valor de empresa: capitalización + deuda a largo plazo (convertibles "
                       "incluidos) − caja e inversiones a corto (la misma liquidez del "
                       "balance). No incluye la deuda a corto plazo, los arrendamientos ni "
                       "los minoritarios.")
    row[3].metric("EV / ventas", multiple(now.ev_sales))
    row = st.columns(4)
    row[0].metric("EV / EBIT", multiple(now.ev_ebit))
    row[1].metric("P / E", multiple(now.pe))
    row[2].metric("P / FCF", multiple(now.p_fcf))
    row[3].metric("Bono EE. UU. 10 años", pct(treasury / 100 if treasury is not None else None))
    row = st.columns(4)
    row[0].metric("Rend. FCF", pct(now.fcf_yield),
                  help="FCF / capitalización: lo que rinde la empresa en caja al precio de hoy. "
                       "Compáralo con el bono: es lo que cobras por el riesgo.")
    row[1].metric("Rend. FCF tras SBC", pct(now.fcf_after_sbc_yield),
                  help="(FCF − compensación en acciones) / capitalización. El FCF suma de "
                       "vuelta el SBC, que es un coste real pagado en acciones.")
    row[2].metric("Rend. del beneficio", pct(now.earnings_yield),
                  help="Beneficio neto / capitalización (la inversa del P/E, con signo).")
    if now.debt_unidentified:
        st.warning("**EV incompleto:** la empresa no presenta ningún concepto estándar de "
                   "deuda, pero su pasivo no corriente supera con mucho lo normal para "
                   "arrendamientos: la deuda existe con etiquetas propias. Tomarla como cero "
                   "daría un EV y un EV/ventas más bajos de lo real, así que no se calculan. "
                   "La capitalización, el P/FCF y los rendimientos sí valen.")
    elif now.debt is None:
        row[3].caption("Sin deuda a largo plazo registrada: el EV la toma como cero.")

    # History: the multiple on each month-end, and where today sits in it.
    available = [m for m in val.MULTIPLES if hist[m].notna().sum() >= 6]
    if available:
        chosen = st.selectbox("Historia del múltiplo", available,
                              format_func=lambda m: MULTIPLE_LABELS[m])
        series = hist[["date", chosen]].dropna()
        current = getattr(now, chosen)
        percentile = val.percentile_in_history(series[chosen], current)
        altair_chart(
            alt.Chart(series.assign(date=pd.to_datetime(series["date"]))).mark_line(
                strokeWidth=2, color=SERIES_COLORS[0]).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y(f"{chosen}:Q", title=MULTIPLE_LABELS[chosen],
                        scale=alt.Scale(zero=False)),
            ).properties(height=220),
            width="stretch",
        )
        st.caption(
            f"Cierre de cada mes, {years} años, cada punto con lo presentado hasta ese día. "
            + (f"Hoy está en el **percentil {percentile * 100:.0f}** de su propia historia "
               "(100 = el más caro que ha estado)." if percentile is not None else "")
        )

    # Reverse DCF: what growth the price assumes.
    st.markdown("**¿Qué crecimiento descuenta el precio?** — DCF inverso")
    discounts = [float(r) for r in RDCF.get("discount_rates", [0.08, 0.10, 0.12])]
    terminal = float(RDCF.get("terminal_growth", 0.025))
    horizon = int(RDCF.get("years", 10))
    implied = val.reverse_dcf(now, discounts, terminal_growth=terminal, years=horizon)

    def cell(result: val.ImpliedGrowth) -> str:
        return pct(result.growth) + " al año" if result.growth is not None else result.note

    st.dataframe(pd.DataFrame({
        "Tasa de descuento": [pct(r, decimals=0) + (" ← tu tasa" if hurdle is not None
                                                   and abs(r - hurdle) < 1e-9 else "")
                              for r in discounts],
        "Sobre el FCF": [cell(x) for x in implied["fcf"]],
        "Sobre el FCF tras SBC": [cell(x) for x in implied["fcf_after_sbc"]],
    }), hide_index=True, width="stretch")
    if any(x.growth is None and "negativa" in x.note for xs in implied.values() for x in xs):
        st.caption("**Base negativa** = ese flujo es hoy negativo: el precio no se explica por "
                   "el flujo actual, sino por uno que todavía no existe.")
    st.caption(
        f"Crecimiento anual constante del flujo durante {horizon} años —después, "
        f"{pct(terminal)} para siempre— que hace que su valor presente iguale la "
        "capitalización de hoy. Si te parece más de lo que la empresa puede sostener, el "
        "precio ya paga más que tu tesis. La tasa de descuento es tuya (§12): se muestra una "
        "rejilla" + ("" if hurdle is not None or public else
                     " y, si escribes `panel.level3.hurdle_rate`, se marca la tuya") + "."
    )

    # The other way round (§15.4, point 7): the return the price gives for each growth.
    st.markdown("**¿Y al revés? Lo que rinde el precio de hoy según cuánto crezca**")
    scenarios = [float(g) for g in VAL.get("return_scenarios", [0, .05, .10, .15, .20, .25])]
    mine = card.get("growth_assumption")
    if mine is not None and float(mine) not in scenarios:
        scenarios = sorted([*scenarios, float(mine)])
    grid_r = val.return_grid(now, scenarios, terminal_growth=terminal, years=horizon)
    st.dataframe(pd.DataFrame({
        "Crecimiento del FCF por acción, al año": [
            pct(x.growth, decimals=0) + (" ← tu supuesto" if mine is not None
                                         and abs(x.growth - float(mine)) < 1e-9 else "")
            for x in grid_r],
        "Rentabilidad anual a este precio": [
            (pct(x.rate) + (" ✓ supera tu tasa" if hurdle is not None and x.rate >= hurdle
                            else ""))
            if x.rate is not None else x.note for x in grid_r],
    }), hide_index=True, width="stretch")
    past = growth_ps.table.set_index(["metric", "years"])
    references = []
    for metric_r, label_r in (("fcf", "FCF"), ("revenue", "ingresos")):
        value_r = past["per_share"].get((metric_r, 3)) if len(past) else None
        if value_r is not None and not pd.isna(value_r):
            references.append(f"{label_r} por acción {pct(value_r)} al año en los últimos 3")
    st.caption(
        f"El crecimiento es del FCF **por acción** durante {horizon} años (después "
        f"{pct(terminal)}): la dilución va dentro, así que si la empresa emite acciones hay "
        "que restarla. Referencia del pasado, no una previsión: "
        + ("; ".join(references) if references else "sin historia suficiente") + ". "
        + ("Tu supuesto viene de `growth_assumption` en la ficha. " if mine is not None else
           "Escribe tu supuesto en la ficha (`growth_assumption: 0.12`) y se marca. ")
        + ("" if hurdle is not None else "Sin `hurdle_rate`, no hay con qué compararla: el "
           f"bono a 10 años rinde {pct(treasury / 100) if treasury is not None else MISSING}.")
    )

# --- 4. The price ------------------------------------------------------------------------


@st.cache_data(show_spinner="Cargando precios…")
def price_view(tickers: tuple[str, ...], as_of_iso: str, mtime: float) -> dict:
    """Split-adjusted close and total-return index per ticker, up to ``as_of``."""
    rows = app_data.prices_for(list(tickers))
    actions = app_data.corporate_actions()
    out = {}
    for t in tickers:
        own = rows[rows["series_id"] == f"{t}:close_raw"].assign(ts=lambda d: d["ts"].str[:10])
        out[t] = {"price": pa.as_series(split_adjusted(own, actions, t, as_of_iso)),
                  "tr": pa.as_series(total_return_index(own, actions, t, as_of_iso))}
    return out


st.divider()
st.subheader("4 · El precio")
# The benchmark is SPY; a card may name its sector ETF too (`benchmark: XLV`), since a
# stock that beats the index while its whole sector did better has not done much.
benchmarks = ["SPY"] + ([str(card["benchmark"])] if card.get("benchmark") else [])
views = price_view((ticker, *benchmarks), as_of_iso, app_data.db_mtime())
stock = views[ticker]
if stock["price"].empty:
    st.info("Sin precios de esta empresa todavía (`python run_ingest.py --only prices`).")
else:
    years_shown = st.select_slider("Ventana", options=[1, 3, 5], value=1,
                                   format_func=lambda y: f"{y} año{'s' if y > 1 else ''}")
    start = pd.Timestamp(as_of_iso) - pd.DateOffset(years=years_shown)
    price = stock["price"][stock["price"].index >= start]
    altair_chart(
        alt.Chart(pd.DataFrame({"date": price.index, "precio": price.to_numpy()}))
        .mark_line(strokeWidth=1.5, color=SERIES_COLORS[0])
        .encode(x=alt.X("date:T", title=None),
                y=alt.Y("precio:Q", title="USD, ajustado por splits",
                        scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("date:T", title="Fecha"),
                         alt.Tooltip("precio:Q", title="Cierre", format=",.2f")])
        .properties(height=240),
        width="stretch",
    )
    compare = pd.concat(
        {name: pa.rebased(views[name]["tr"], start) for name in (ticker, *benchmarks)},
        axis=1).dropna(how="all")
    long = compare.reset_index(names="date").melt("date", var_name="Serie", value_name="valor")
    altair_chart(
        alt.Chart(long.dropna()).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("valor:Q", title="Base 100, con dividendos", scale=alt.Scale(zero=False)),
            color=alt.Color("Serie:N", legend=alt.Legend(orient="top", title=None),
                            scale=alt.Scale(domain=[ticker, *benchmarks],
                                            range=["#2a78d6", "#eb6834", "#3a9d5d"])),
        ).properties(height=220),
        width="stretch",
    )
    b = pa.behaviour(stock["tr"], views["SPY"]["tr"], as_of_iso, days=365)
    row = st.columns(4)
    row[0].metric("Rentabilidad 1 año", pct(b.total_return), help="Con dividendos.")
    row[1].metric("SPY 1 año", pct(b.benchmark_return))
    row[2].metric("Diferencia", pct(b.excess))
    row[3].metric("Beta frente a SPY", number(b.beta),
                  help="Cuánto amplifica los movimientos del mercado (1 = igual que el "
                       "índice). Un año de rendimientos diarios.")
    row = st.columns(4)
    row[0].metric("Volatilidad anual", pct(b.volatility),
                  help="Desviación típica de los rendimientos diarios, anualizada. SPY "
                       "ronda el 15-20 %.")
    row[1].metric("Caída desde el máximo", pct(b.drawdown_now),
                  help="Del cierre más alto del último año al de hoy.")
    row[2].metric("Peor caída del año", pct(b.max_drawdown))

    events = app_data.events()
    confirmed = events[(events["category"] == "earnings") & (events["cik"] == cik)
                       & (events["is_estimated"].fillna(0).astype(int) == 0)
                       & (events["ts"].str[:10] <= as_of_iso)] if not events.empty else events
    reactions = pa.earnings_reactions(stock["tr"], views["SPY"]["tr"],
                                      list(confirmed["ts"]) if not confirmed.empty else [])
    if not reactions.empty:
        st.markdown("**Reacción a los resultados**")
        st.dataframe(pd.DataFrame({
            "Resultados": reactions["date"],
            "Movimiento": reactions["move"].map(pct),
            "SPY": reactions["benchmark_move"].map(pct),
            "Por encima de SPY": reactions["excess"].map(pct),
        }).head(8), hide_index=True, width="stretch")
        typical = pa.typical_reaction(reactions)
        st.caption(
            "Del cierre anterior al 8-K de resultados (item 2.02) al cierre siguiente: la "
            "SEC no dice si se presentó antes de la apertura o tras el cierre, y esa ventana "
            "recoge la reacción en los dos casos. "
            + (f"**Movimiento típico: ±{pct(typical)}** sobre SPY — es lo que se juega una "
               "posición abierta justo antes de resultados (§2: 5 días de margen)."
               if typical is not None else "")
        )

# Short interest (§15.1.7): a fact about the company, so it lives here and not only in the
# private market block. The public copy carries no FINRA rows (db/public_sync.py: the
# series cover the held tickers too), so publicly the block simply does not appear.
finra = app_data.observations("finra", app_data.db_mtime())
short = macro_latest(finra, f"{ticker}:short_interest", pd.Timestamp(as_of_iso)) \
    if not finra.empty else None
if short is not None and short.value is not None:
    st.markdown("**Interés corto**")
    cover = macro_latest(finra, f"{ticker}:days_to_cover", pd.Timestamp(as_of_iso))
    revised = macro_latest(finra, f"{ticker}:short_interest:revised", pd.Timestamp(as_of_iso))
    shares_now = now.shares if now is not None else None
    row = st.columns(4)
    row[0].metric("Acciones en corto", compact_amount(short.value))
    row[1].metric("De las acciones diluidas", pct(short.value / shares_now)
                  if shares_now else MISSING,
                  help="Sobre el recuento diluido de la valoración, no sobre el *float*: "
                       "la SEC no publica el float en sus datos estructurados.")
    row[2].metric("Días para cubrir", number(cover.value),
                  help="Acciones en corto / volumen medio diario: cuántas sesiones haría "
                       "falta para recomprarlas todas.")
    row[3].metric("Liquidación", short.ts or MISSING)
    history = finra[finra["series_id"] == f"{ticker}:short_interest"]
    history = history[history["ts_release"].astype(str).str[:10] <= as_of_iso]
    if shares_now and len(history) > 4:
        frame = pd.DataFrame({"date": pd.to_datetime(history["ts"].astype(str).str[:10]),
                              "corto": history["value"].astype(float) / shares_now})
        frame = frame[frame["date"] >= pd.Timestamp(as_of_iso) - pd.DateOffset(years=2)]
        altair_chart(
            alt.Chart(frame).mark_line(strokeWidth=1.5, color=SERIES_COLORS[0]).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("corto:Q", title="En corto, % de las diluidas de hoy",
                        axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("date:T", title="Liquidación"),
                         alt.Tooltip("corto:Q", title="En corto", format=".1%")],
            ).properties(height=160),
            width="stretch",
        )
    st.caption(
        "FINRA, dos veces al mes. Publicado (fecha **derivada**, tarde a propósito): "
        f"{short.ts_release or MISSING}. "
        + ("⚠️ FINRA revisó esta cifra después de publicarla. " if revised.value else "")
        + "Mucho interés corto no es una señal de venta ni de compra: dice que hay "
          "inversores apostando dinero a que la tesis contraria es la buena — conviene saber "
          "cuál es. La historia usa el recuento de hoy como denominador."
    )
elif not public:
    st.caption("Sin interés corto de esta empresa (`python run_ingest.py --only short_interest`).")

# --- 5. Value accrual -----------------------------------------------------------------

st.divider()
st.subheader("5 · ¿Cómo llega eso al accionista?")
st.caption(
    "El concepto rector (§2): una empresa puede crecer 30% al año y destruir valor por "
    "acción si diluye 35%. Esta sección es el análogo de la vista de unlocks."
)

row = st.columns(4)
row[0].metric(
    "Dilución interanual (diluidas)", pct(accrual.dilution_yoy, decimals=2),
    help="Acciones diluidas medias del ejercicio frente al anterior. Incluye opciones y "
         "convertibles… solo en los años con beneficio: con pérdida, la norma las iguala a "
         "las básicas (§9.15), así que salta al cruzar de pérdida a ganancia.",
)
row[1].metric(
    "Acciones en circulación, interanual",
    pct(growth_ps.shares_yoy, decimals=2) if growth_ps.basis == "basic_shares" else MISSING,
    help="Acciones básicas medias, fechas emparejadas a un año: la emisión real, sin el "
         "vaivén de las diluidas. Vacía si la empresa no presenta la básica al día.",
)
row[2].metric("SBC / ingresos", pct(accrual.sbc_over_revenue))
row[3].metric(
    "SBC / FCF", pct(accrual.sbc_over_fcf),
    help="Cuánto del efectivo que genera el negocio ya está comprometido con empleados.",
)

# Growth of the business against growth per share (§2, the gap this page exists for).
labels_ps = {"revenue": "Ingresos", "fcf": "Flujo de caja libre", "net_income": "Beneficio neto"}
horizons_ps = sorted({int(y) for y in growth_ps.table["years"]})
grid_ps = {}
for metric_ps, label_ps in labels_ps.items():
    rows_ps = growth_ps.table[growth_ps.table["metric"] == metric_ps].set_index("years")
    grid_ps[label_ps] = {}
    for y in horizons_ps:
        grid_ps[label_ps][f"{y} años · empresa"] = pct(rows_ps.at[y, "total"])
        grid_ps[label_ps][f"{y} años · por acción"] = pct(rows_ps.at[y, "per_share"])
grid_ps["Acciones (" + ("básicas" if growth_ps.basis == "basic_shares" else "diluidas") + ")"] = {
    **{f"{y} años · empresa": pct(growth_ps.shares.get(y)) for y in horizons_ps},
    **{f"{y} años · por acción": "" for y in horizons_ps}}
st.markdown("**Crecimiento del negocio frente a crecimiento por acción** — anual compuesto")
st.dataframe(pd.DataFrame(grid_ps).T.replace({"None": MISSING}), width="stretch")
notes_ps = []
if any(growth_ps.jumps.values()):
    notes_ps.append("Una ventana cruza un **salto de más del 50 % en el recuento** en un solo "
                    "trimestre (salida a bolsa, SPAC o fusión): antes, las preferentes que "
                    "luego convierten no cuentan, y comparar daría una dilución que no lo es. "
                    "Esa ventana queda sin cifra por acción.")
if any(growth_ps.flips.values()):
    notes_ps.append("Con el recuento diluido, una ventana **pasa de pérdida a beneficio**: en "
                    "pérdida las diluidas no cuentan opciones (§9.15), así que no se comparan.")
st.caption(
    "Crecimiento compuesto de los doce meses (TTM), emparejando fechas; «por acción» divide "
    "cada punto por las acciones "
    + ("**básicas** medias del mismo periodo (las diluidas saltan cuando la empresa cruza de "
       "pérdida a ganancia)" if growth_ps.basis == "basic_shares" else
       "**diluidas** medias del mismo periodo (la empresa no presenta la básica al día)")
    + ". La diferencia entre las dos columnas es el crecimiento que se llevaron los nuevos "
      "accionistas. Vacío = base negativa o sin historia suficiente: el crecimiento no está "
      "definido, no es cero. " + " ".join(notes_ps)
)
if len(growth_ps.chart) > 2:
    long_ps = growth_ps.chart.melt("date", var_name="Serie", value_name="valor")
    long_ps["Serie"] = long_ps["Serie"].map({"total": "Ingresos de la empresa",
                                             "per_share": "Ingresos por acción"})
    altair_chart(
        alt.Chart(long_ps).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("valor:Q", title="Índice (inicio = 100), escala log.",
                    scale=alt.Scale(type="log")),
            color=alt.Color("Serie:N", legend=alt.Legend(orient="top", title=None),
                            scale=alt.Scale(domain=["Ingresos de la empresa",
                                                    "Ingresos por acción"],
                                            range=SERIES_COLORS)),
        ).properties(height=200),
        width="stretch",
    )

row = st.columns(4)
row[0].metric(
    "ROIC (aprox.)", pct(accrual.roic),
    help="NOPAT sobre capital invertido, con la tasa impositiva OBSERVADA en los filings, "
         "no la estatutaria. Capital invertido = patrimonio + deuda largo plazo − caja.",
)

row = st.columns(4)
row[0].metric("Recompras brutas (TTM, USD)", compact_amount(accrual.buybacks_ttm),
              help=reported_amount(accrual.buybacks_ttm))
row[1].metric("Recompras netas (TTM, USD)", compact_amount(accrual.net_buybacks_ttm),
              help="Recompras − emisión. La bruta engaña si el SBC la anula. "
                   + reported_amount(accrual.net_buybacks_ttm))
# The count grew for most companies that pay in stock: a tile reading "retired: −21 M"
# made the reader do the sign. The label says which way it went.
issued = accrual.shares_removed is not None and accrual.shares_removed < 0
row[2].metric(
    "Acciones emitidas netas (año)" if issued else "Acciones retiradas (año)",
    compact_amount(abs(accrual.shares_removed) if issued else accrual.shares_removed),
    help="Variación del recuento diluido medio en un año. Emitidas netas = la emisión "
         "(SBC, conversiones, ampliaciones) superó a las recompras.",
)
row[3].metric(
    "Coste por acción retirada (USD)",
    compact_amount(accrual.buyback_per_share_removed),
    help="Recompra neta dividida por la caída real del recuento. Si es mucho mayor que el "
         "precio de mercado, la recompra está compensando dilución, no devolviendo capital.",
)

if accrual.roic is not None:
    if hurdle is None:
        st.caption(
            "Sin tasa exigida de referencia, el ROIC no se compara contra nada. "
            "**Desconocido no es aprobado** (§12)." if public else
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
    altair_chart(
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
st.subheader("6 · Presentaciones y calendario")

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

# What the filings say before the numbers do (transform/filing_signals.py): late filings,
# non-reliance, delisting notices, auditor changes — and a shelf or sale by a company that
# burns cash, which is almost always new shares.
LOOKBACK = int(settings.raw.get("panel", {}).get("level3", {}).get("signal_lookback_days", 365))
found = fs.signals(filings, cik, pd.Timestamp(as_of_iso) - pd.Timedelta(days=LOOKBACK),
                   as_of_iso)
if not found.empty:
    burns = None if snapshot.fcf_ttm is None else snapshot.fcf_ttm < 0
    for w in fs.warnings(found, burns):
        (st.error if w.severity == fs.RED else st.warning)(
            f"**{w.date} · {w.form}** — {w.text}." + (f" [Documento]({w.url})" if w.url else ""))
    st.markdown(f"**Señales en las presentaciones** · últimos {LOOKBACK} días")
    st.dataframe(pd.DataFrame({
        "Fecha": found["date"], "Formulario": found["form"],
        "Qué es": found["label"],
        "Nivel": found["severity"].map({fs.RED: "🔴", fs.YELLOW: "🟡", fs.INFO: "·"}),
        "Documento": found["url"],
    }), hide_index=True, width="stretch",
        column_config={"Documento": st.column_config.LinkColumn(display_text="abrir")})
    st.caption(
        "Hechos que registra la SEC el día que llegan. Un registro para vender valores o una "
        "venta bajo él puede ser deuda (así financian las grandes) o acciones: el panel no lee "
        "el documento, y solo lo sube a aviso cuando la empresa **quema caja**, que es cuando "
        "casi siempre son acciones nuevas. Un 13D puede ser un activista o un accionista de "
        "control; léelo."
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
    st.subheader("7 · Tu posición")

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
        row[0].metric("Cantidad", quantity(position["quantity"]))
        row[1].metric("Coste medio", money(avg, public=False))
        row[2].metric("Valor", money(position["market_value"], public=False))
        row[3].metric(
            "PnL no realizado", money(position["unrealized_pnl"], public=False),
            delta=pct(position["unrealized_return"]),
        )
