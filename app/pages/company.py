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
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import (
    BAD,
    CAUTION,
    GOOD,
    MISSING,
    PUBLIC_NOTE,
    altair_chart,
    color_by_sign,
    color_rows_by_sign,
    compact_amount,
    money,
    number,
    pct,
    quantity,
    reported_amount,
    tone,
    verdict_delta,
)
from core.config import load_settings
from transform import bcb
from transform import catalysts as cat
from transform import entry_context as ec
from transform import filing_signals as fs
from transform import inflection as infl
from transform import ownership as own
from transform import insider_activity as ia
from transform import per_share
from transform import fundamentals as fun
from transform import portfolio as port
from transform import thesis as th
from transform import price_action as pa
from transform import quarterly as qt
from transform import screen as sc
from transform import scenarios as scn
from transform import segments as sg
from transform import sfc
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
# Positions held with no card yet (local only: holdings are private). They get the page as a
# candidate does — the numbers are what a card is written from — marked so it is never
# forgotten that the position already exists (§5.1).
held_no_card: set[str] = set()
if not public and app_data.database_ready():
    account_now = app_data.account_observations()
    held_now = port.latest_positions(account_now) if not account_now.empty else pd.DataFrame()
    registry_held = app_data.companies()
    carded = {str(c.get("ticker")) for c in cards}
    for held_ticker in (held_now["ticker"] if not held_now.empty else []):
        match = registry_held[registry_held["ticker"] == held_ticker] \
            if not registry_held.empty else registry_held
        if held_ticker not in carded and len(match):
            cards = [*cards, {"ticker": held_ticker, "cik": str(match["cik"].iloc[0])}]
            held_no_card.add(str(held_ticker))
under_study |= held_no_card
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
        format_func=lambda t: (f"{t} · en cartera, sin ficha" if t in held_no_card else
                               f"{t} · en estudio" if t in under_study else t),
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
if ticker in held_no_card:
    st.warning(f"**Tienes {ticker} en cartera y no tiene ficha de tesis.** La posición existe "
               "antes que la razón escrita para tenerla: sin criterio de invalidación no hay "
               "forma de saber cuándo venderla (§5.2).")

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
              **verdict_delta(tone(snapshot.net_income_ttm), "beneficio"
                              if (snapshot.net_income_ttm or 0) > 0 else "pérdida"),
              help=reported_amount(snapshot.net_income_ttm))
row[2].metric("Flujo de caja libre (TTM, USD)", compact_amount(snapshot.fcf_ttm),
              **verdict_delta(tone(snapshot.fcf_ttm), "genera caja"
                              if (snapshot.fcf_ttm or 0) > 0 else "quema caja"),
              help=reported_amount(snapshot.fcf_ttm))
row[3].metric(
    "Crecimiento de ingresos",
    pct(snapshot.revenue_growth_yoy),
    **verdict_delta(tone(snapshot.revenue_growth_yoy), "crece"
                    if (snapshot.revenue_growth_yoy or 0) > 0 else "decrece"),
    help="TTM contra el TTM de un año antes. Geométrico, nunca media de porcentajes (§9.10).",
)

row = st.columns(4)
for column, label, value in ((row[0], "Margen operativo", snapshot.operating_margin),
                             (row[1], "Margen neto", snapshot.net_margin),
                             (row[2], "Margen FCF", snapshot.fcf_margin)):
    column.metric(label, pct(value), **verdict_delta(
        tone(value), "gana dinero" if (value or 0) > 0 else "pierde dinero"))
# Below 1 is the help's own definition of "part of the profit does not reach cash"; common
# in heavy-capex companies (MSFT 0,50), so a caution, not a red.
row[3].metric(
    "Conversión a caja",
    number(snapshot.cash_conversion) if snapshot.cash_conversion is not None else MISSING,
    **verdict_delta(None if snapshot.cash_conversion is None else
                    GOOD if snapshot.cash_conversion >= 1 else CAUTION,
                    "todo llega a caja" if (snapshot.cash_conversion or 0) >= 1
                    else "parte no llega a caja"),
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
        width="stretch", amounts=True,
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
    st.dataframe(color_rows_by_sign(grid, {
        "Crecimiento interanual": True, "Margen bruto": True, "Margen operativo": True,
        "Margen neto": True, "Margen FCF": True}), width="stretch")
    missing = [label for label, key, _ in lines if table[key].isna().all()]
    st.caption(
        "Cada columna es un trimestre tal como se conocía a la fecha elegida; los Q4 y los "
        "trimestres escondidos en acumulados se derivan (§9.12, §9.14). El crecimiento es "
        "**interanual** — contra el mismo trimestre del año anterior, no contra el anterior, "
        "que mezclaría la estacionalidad. Margen bruto solo si la empresa presenta "
        "`GrossProfit`: restar el coste de ventas no da lo mismo en todas."
        + (f" No presentadas en XBRL estándar: {', '.join(missing)}." if missing else "")
    )
    # The latest quarter's change of pace (§15.5 point 5), with the alert's own thresholds.
    rule = next((r for r in settings.raw.get("alerts", {}).get("rules", [])
                 if r.get("kind") == "fundamental_inflection"), {})
    turn = infl.assess(observations, cik, as_of_iso,
                       acceleration=float(rule.get("acceleration", 0.10)),
                       gross_margin=float(rule.get("gross_margin", 0.03)))
    if turn is not None and turn.signals:
        for _kind, text in turn.signals:
            st.info(f"**Inflexión, trimestre al {turn.quarter}.** {text}")
    elif turn is not None and turn.acceleration is not None:
        st.caption(f"Ritmo del último trimestre: el crecimiento interanual cambió "
                   f"{number(turn.acceleration * 100, decimals=1)} puntos frente al trimestre "
                   "anterior, por debajo de tu umbral de inflexión "
                   f"({number(float(rule.get('acceleration', 0.10)) * 100, decimals=0)} puntos).")

# Segments (§15.5 point 15): where the growth is. From each filing's XBRL, which is the
# only free source of dimensional facts; not in the public copy, so publicly absent.
SEG_CFG = dict(settings.source("segments"))
segment_obs = app_data.observations("sec_segments", app_data.db_mtime())
segments = sg.assess(segment_obs, cik, as_of_iso,
                     settings.source("sec").get("concepts", {}).get("revenue", {}).get("tags", []),
                     SEG_CFG.get("profit_concepts", [])) if not segment_obs.empty else None
if segments is not None and len(segments.members) == 1:
    only = segments.members[0]
    generic = only.lower() in {"singlereporting", "reportable", "consolidated", "single"}
    st.caption("Segmentos: la empresa reporta **uno solo**"
               + ("" if generic else f" ({only})") + "; sus cifras son las de la empresa entera.")
elif segments is not None:
    profit_name = sg.labels(SEG_CFG.get("profit_labels", {}), segments.profit_concept)
    st.markdown(f"**Por segmento** · trimestre al {segments.quarter}")
    st.dataframe(pd.DataFrame([{
        "Segmento": r["member"],
        "Ingresos": compact_amount(r["revenue"]),
        "Crec. interanual": pct(r["revenue_growth"]),
        "Peso en los ingresos": pct(r["share"], decimals=0),
        (profit_name or "Resultado"): compact_amount(r["profit"]),
        "Margen": pct(r["margin"]),
        "Crec. del resultado": pct(r["profit_growth"]),
    } for r in segments.table.to_dict("records")]), hide_index=True, width="stretch")
    chart_data = segments.history[segments.history["date"] >= segments.history["date"].max()
                                  - pd.DateOffset(months=24)]
    palette = ["#2a78d6", "#eb6834", "#3aa76d", "#8e6bd8", "#c9a227", "#9aa5b1"]
    altair_chart(
        alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("revenue:Q", title="Ingresos por segmento, USD", stack=True),
            color=alt.Color("member:N", sort=segments.members,
                            scale=alt.Scale(domain=segments.members,
                                            range=palette[:len(segments.members)]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("date:T", title="Trimestre"),
                     alt.Tooltip("member:N", title="Segmento"),
                     alt.Tooltip("revenue:Q", title="USD", format=",.0f"),
                     alt.Tooltip("derived:N", title="Derivado")],
        ).properties(height=220),
        width="stretch", amounts=True,
    )
    notes = ["Del XBRL de cada 10-Q y 10-K: la única fuente gratuita de cifras por segmento. "
             "Crecimiento interanual; el cuarto trimestre, cuando la empresa solo lo da dentro "
             "del año, se deriva (año − nueve meses, §9.12)."]
    if segments.profit_concept is None:
        notes.append("Sin resultado por segmento: la empresa usa una medida que el panel no "
                     "conoce todavía (se añade en `sources.segments.profit_concepts`).")
    if segments.older_concepts:
        notes.append(f"⚠️ **Cambio de medida:** antes de este trimestre la empresa medía el "
                     f"segmento con «{', '.join(sg.labels(SEG_CFG.get('profit_labels', {}), c) for c in segments.older_concepts)}», y "
                     f"ahora con «{profit_name}». El crecimiento del resultado solo compara "
                     "periodos con la medida nueva; las dos series no se empalman (§9.8).")
    if segments.restated:
        notes.append(f"{segments.restated} cifra(s) por segmento aparecen con otro valor en una "
                     "presentación posterior (reexpresión o reclasificación): manda la última.")
    notes.append("Las métricas operativas (reservas, viajes, usuarios) no van etiquetadas en "
                 "el XBRL: solo en el texto y el comunicado de resultados.")
    st.caption(" ".join(notes))

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
                  **verdict_delta(tone(sheet.net_cash), "más caja que deuda"
                                  if (sheet.net_cash or 0) > 0 else "más deuda que caja"),
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
                  **verdict_delta(BAD if sheet.cash_burn_ttm else None, "quema caja"),
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

# Against the peers the user chose (§15.5 point 11): the SIC group above fails on the
# companies that matter (HIMS is "offices of doctors", UBER a catch-all code). The list lives
# in settings.local.yaml, so the public deployment never shows it.
peer_list = settings.peers.get(ticker, [])
if peer_list:
    peer_rows, peer_median = app_data.peers_view(ticker, tuple(peer_list), as_of_iso,
                                                 app_data.db_mtime())
    shown = peer_rows[peer_rows["found"]]
    missing = [str(t) for t in peer_rows.loc[~peer_rows["found"], "ticker"]]
    # A peer whose last readable period is old (DiDi: US GAAP until FY2023, IFRS since) is
    # shown with its period and said out loud: compared as if current, it would mislead.
    old_rows = [f"{r['ticker']} ({r['quarter']})" for r in shown.to_dict("records")
                if r.get("quarter_end") and (pd.Timestamp(as_of_iso)
                                             - pd.Timestamp(r["quarter_end"])).days > 460]
    st.markdown(f"**Frente a tus pares** · {len(shown) - int(shown['role'].eq('empresa').sum())}"
                f" de {len(peer_list)} con cifras")
    PEER_COLUMNS = {
        "revenue_growth": "Crec. ingresos", "acceleration": "Aceleración (pp)",
        "gross_margin": "Margen bruto", "operating_margin": "Margen operativo",
        "incremental_margin": "Margen op. incremental", "fcf_margin": "Margen FCF (año)",
        "sbc_over_revenue": "SBC / ingresos", "dilution": "Dilución",
        "rule_of_40": "Regla del 40", "price_to_sales": "P / ventas",
        "market_cap": "Capitalización (mil M)",
    }

    def scaled(metric: str, value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if metric == "acceleration":
            return value * 100
        return value / 1e9 if metric == "market_cap" else value

    table_rows = [{"Empresa": ("▶ " if r["role"] == "empresa" else "") + str(r["ticker"]),
                   "Nombre": r.get("name"),
                   "Periodo": str(r.get("quarter") or "").replace("CY", "").replace("Q", " T"),
                   **{label: scaled(m, r.get(m)) for m, label in PEER_COLUMNS.items()}}
                  for r in peer_rows.to_dict("records") if r["found"]]
    table_rows.append({"Empresa": "Mediana de los pares", "Nombre": None, "Periodo": None,
                       **{label: scaled(m, peer_median.get(m))
                          for m, label in PEER_COLUMNS.items()}})
    percent = st.column_config.NumberColumn(format="percent")
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch", column_config={
        "Crec. ingresos": percent, "Margen bruto": percent, "Margen operativo": percent,
        "Margen op. incremental": percent, "Margen FCF (año)": percent,
        "SBC / ingresos": percent, "Dilución": percent, "Regla del 40": percent,
        "Aceleración (pp)": st.column_config.NumberColumn(format="%.1f"),
        "P / ventas": st.column_config.NumberColumn(format="%.1f×"),
        "Capitalización (mil M)": st.column_config.NumberColumn(format="%.1f"),
    })
    st.caption(
        "Cada empresa en su último trimestre presentado frente al mismo del año anterior "
        "(🔎 Cribado, modo crecimiento); las que solo presentan un informe anual, en su último "
        "ejercicio («FY»), con cocientes que no dependen de la moneda. La mediana es la de los "
        "pares, sin la empresa. Nada aquí dice cuál es mejor: más crecimiento y más dilución "
        "no se leen en la misma dirección. En un banco o una financiera, el margen FCF y la "
        "regla del 40 quedan vacíos: su flujo de caja lleva dentro los préstamos que concede. "
        "Los pares los eliges tú en `universe.peers`."
        + (f" **Sin cifras:** {', '.join(missing)} (emisor IFRS, sin ingresos presentados o "
           "fuera del cribado todavía)." if missing else "")
        + (f" ⚠️ **Cifras antiguas:** {', '.join(old_rows)} — su último periodo legible tiene "
           "más de 15 meses (por ejemplo, un emisor extranjero que pasó a IFRS, que esta "
           "fuente no lee): no lo compares como si fuera actual." if old_rows else ""))

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
                width="stretch", amounts=True,
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

# The same lender against its competitors, by the same supervisor (§15.5 point 11): in
# Brazil the IF.data institutions marked `peer_of`, in Colombia the CUIF entities. The same
# definitions for all, which is what makes these the cleanest comparison there is.
bcb_group = [i for i in settings.source("bcb").get("institutions", [])
             if ticker in (str(i.get("ticker")), str(i.get("peer_of")))]
if len(bcb_group) > 1:
    ifdata = app_data.observations("bcb_ifdata", app_data.db_mtime())
    lines = []
    for inst in bcb_group:
        view = bcb.assess(ifdata, str(inst["code"]), as_of_iso)
        if view is None:
            continue
        lines.append({
            "Institución": ("▶ " + ticker) if inst.get("ticker") == ticker else inst.get("name"),
            "Trimestre": view.quarter,
            "Cartera (mil M R$)": None if view.credit_portfolio is None
            else view.credit_portfolio / 1e9,
            "Crec. cartera": view.credit_growth,
            "Clientes con crédito (M)": None if view.credit_clients is None
            else view.credit_clients / 1e6,
            "Basilea": view.basel,                       # IF.data: a fraction (0,155)
            "Activos problemáticos": view.problem_share,
            "ROE anualizado": view.roe_annualized,
        })
    if len(lines) > 1:
        st.markdown("**Frente a los bancos de Brasil** · Banco Central do Brasil (IF.data)")
        percent = st.column_config.NumberColumn(format="percent")
        st.dataframe(pd.DataFrame(lines), hide_index=True, width="stretch", column_config={
            "Cartera (mil M R$)": st.column_config.NumberColumn(format="%.1f"),
            "Clientes con crédito (M)": st.column_config.NumberColumn(format="%.2f"),
            "Crec. cartera": percent, "Basilea": percent, "Activos problemáticos": percent,
            "ROE anualizado": percent})
        st.caption("Mismos informes y mismas definiciones del supervisor para todos: la "
                   "comparación más limpia. Conglomerados prudenciales, solo Brasil. «Activos "
                   "problemáticos» sobre la exposición total (Res. 4.966); ROE = 4 × beneficio "
                   "del trimestre / patrimonio, aproximado. Crecimiento de cartera solo dentro "
                   "de la misma base contable; uno enorme es una entidad que empezó a prestar "
                   "hace poco (Revolut Brasil).")

sfc_group = [e for e in settings.source("sfc").get("entities", [])
             if ticker in (str(e.get("ticker")), str(e.get("peer_of")))]
if sfc_group:
    cuif = app_data.observations("sfc_cuif", app_data.db_mtime())
    snaps = sfc.compare(cuif, sfc_group, as_of_iso) if not cuif.empty else []
    if snaps:
        st.markdown("**Frente a los de Colombia** · Superintendencia Financiera (CUIF)")
        percent = st.column_config.NumberColumn(format="percent")
        billions = st.column_config.NumberColumn(format="%.2f")
        st.dataframe(pd.DataFrame([{
            "Entidad": ("▶ " if next((e for e in sfc_group if str(e.get("name")) == x.name),
                                     {}).get("ticker") == ticker else "") + x.name,
            "Mes": x.month,
            "Activo (billones COP)": None if x.total_assets is None else x.total_assets / 1e12,
            "Cartera neta (billones)": None if x.loans_net is None else x.loans_net / 1e12,
            "Crec. cartera": x.loans_growth,
            "Depósitos (billones)": None if x.deposits is None else x.deposits / 1e12,
            "Crec. depósitos": x.deposits_growth,
            "Cartera / depósitos": x.loans_to_deposits,
            "Cartera C, D y E": x.risk_share,
            "ROE anualizado": x.roe_annualized,
        } for x in snaps]), hide_index=True, width="stretch", column_config={
            "Activo (billones COP)": billions, "Cartera neta (billones)": billions,
            "Depósitos (billones)": billions, "Crec. cartera": percent,
            "Crec. depósitos": percent, "Cartera / depósitos": percent,
            "Cartera C, D y E": percent, "ROE anualizado": percent})
        st.caption(
            "Normas colombianas, en pesos y **solo Colombia**. «Cartera C, D y E»: la parte de "
            "la cartera bruta en riesgo apreciable, significativo o de incobrabilidad, según la "
            "calificación del supervisor (no es la morosidad a 90 días que publica la empresa). "
            "El resultado es acumulado desde enero; el ROE lo anualiza (× 12 / mes) y es "
            "aproximado. Un crecimiento de miles por ciento es una entidad que empezó a "
            "prestar hace menos de un año (base casi nula), no un error. Fecha de publicación: "
            "primera vista o cierre + 60 días.")

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
                # A yield is a fraction (0,057 = 5,7 %); a multiple reads "30×", not "30".
                y=alt.Y(f"{chosen}:Q", title=MULTIPLE_LABELS[chosen],
                        scale=alt.Scale(zero=False),
                        axis=alt.Axis(format="%") if chosen == "fcf_yield"
                        else alt.Axis(labelExpr="datum.label + '×'")),
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

# Scenarios and size (§15.5 point 3): the card's three cases, what they imply at today's
# price, and how large the position can be for the loss the user accepts. Every number comes
# from the card; without one the block says how to write it.
SCEN = dict(settings.raw.get("panel", {}).get("scenarios") or {})
PORT_CFG = dict(settings.raw.get("portfolio") or {})
spec = card.get("scenarios")
st.markdown("**Tus escenarios y el tamaño que admiten**")
if not spec:
    if not public:
        st.caption("Sin escenarios en la ficha. Tres casos con su probabilidad, cada uno como "
                   "crecimiento del FCF por acción (`growth`) o como múltiplo del precio de hoy "
                   "a N años (`multiple`, que sirve también para una empresa que quema caja o "
                   "que la SEC no lee):")
        st.code("scenarios:\n  years: 5\n"
                "  bear: {prob: 0.25, multiple: 0.4, note: \"la tesis falla\"}\n"
                "  base: {prob: 0.5, growth: 0.12}\n"
                "  bull: {prob: 0.25, multiple: 4.0}", language="yaml")
else:
    budget = PORT_CFG.get("loss_budget")
    result = scn.assess(spec, now if now.market_cap is not None else None,
                        years=int(SCEN.get("horizon_years", 5)),
                        terminal_growth=float(RDCF.get("terminal_growth", 0.025)),
                        dcf_years=int(RDCF.get("years", 10)),
                        kelly_share=float(SCEN.get("kelly_fraction", 0.25)),
                        loss_budget=budget, max_position=PORT_CFG.get("max_position"))
    n = result.years
    st.dataframe(pd.DataFrame([{
        "Escenario": scn.LABELS[x.name], "Probabilidad": pct(x.prob, decimals=0),
        "Lo que escribiste": (f"FCF por acción {pct(x.growth, decimals=0)} al año"
                              if x.growth is not None else
                              f"precio × {number(x.multiple_written, decimals=2)} en {n} años"),
        "Rentabilidad anual": pct(x.annual_return) if x.annual_return is not None else MISSING,
        f"Tu dinero en {n} años": (f"× {number(x.multiple, decimals=2)}"
                                   if x.multiple is not None else MISSING),
        "Nota": x.note,
    } for x in result.scenarios]), hide_index=True, width="stretch")
    row = st.columns(4)
    row[0].metric("Rentabilidad esperada, anual", pct(result.expected_return),
                  help="Del dinero esperado al final ((Σ prob × múltiplo)^(1/años) − 1), no "
                       "de la media de las rentabilidades anuales, que la exagera (§9.10).")
    row[1].metric("Probabilidad de perder dinero", pct(result.loss_probability, decimals=0))
    row[2].metric("Peor caída escrita", pct(-result.worst_loss) if result.worst_loss else
                  ("ninguna" if result.worst_loss == 0 else MISSING))
    row[3].metric("Techo de tamaño", pct(result.ceiling, decimals=1),
                  help="El menor de: presupuesto de pérdida / peor caída, Kelly fraccional y tu "
                       "`portfolio.max_position`. Un techo, nunca un objetivo.")
    lines = []
    if result.size_by_budget is not None:
        lines.append(f"con tu presupuesto de pérdida del **{pct(budget, decimals=1)}** de la "
                     f"cuenta, el escenario pesimista cuesta eso con un peso del "
                     f"**{pct(result.size_by_budget, decimals=1)}**; si la tesis se fuera a cero, "
                     f"con uno del **{pct(result.size_if_zero, decimals=1)}**")
    if result.kelly is not None:
        lines.append(f"Kelly completo sobre tus escenarios: {pct(result.kelly, decimals=0)} "
                     f"(se enseña × {number(float(SCEN.get('kelly_fraction', 0.25)), decimals=2)}"
                     f" = **{pct(result.kelly_ceiling, decimals=1)}**, como techo)")
    if not public:
        account_now = app_data.account_observations()
        if not account_now.empty:
            held_now = port.latest_positions(account_now)
            held_now = held_now[held_now["ticker"] == ticker]
            nav_now = port.nav_series(account_now)
            if not held_now.empty and not nav_now.empty:
                valued_now = port.valuation(held_now, port.latest_prices(
                    app_data.prices_for([ticker])))
                value_now = valued_now["market_value"].iloc[0]
                if pd.notna(value_now):
                    weight_now = float(value_now) / float(nav_now["value"].iloc[-1])
                    lines.append(f"hoy pesa el **{pct(weight_now, decimals=1)}** de la cuenta")
                    if result.ceiling is not None and weight_now > result.ceiling:
                        st.warning(f"Pesa el {pct(weight_now, decimals=1)} de la cuenta, por "
                                   f"encima del techo que sale de lo que escribiste "
                                   f"({pct(result.ceiling, decimals=1)}).")
    st.caption(("Tamaño: " + "; ".join(lines) + ". " if lines else "")
               + " ".join(result.notes)
               + " Los escenarios y sus probabilidades son tuyos: el panel solo hace la "
                 "aritmética. Una probabilidad optimista hace optimista todo lo de arriba.")

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
    row[0].metric("Rentabilidad 1 año", pct(b.total_return), help="Con dividendos.",
                  **verdict_delta(tone(b.total_return),
                                  "sube" if (b.total_return or 0) > 0 else "baja"))
    row[1].metric("SPY 1 año", pct(b.benchmark_return))
    row[2].metric("Diferencia", pct(b.excess),
                  **verdict_delta(tone(b.excess), "por encima de SPY"
                                  if (b.excess or 0) > 0 else "por debajo de SPY"))
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

    # Where the price stands before buying (§15.5 points 8 and 12): the conditions of the
    # knife study, as description. They became no gate: the study's single run found no
    # evidence for either (RESEARCH.md §2.65), and the verdict lives in config.
    KNIFE = dict(settings.raw.get("knife_study") or {})
    entry = ec.assess(stock["price"], as_of_iso, KNIFE["conditions"]) if KNIFE else None
    if entry is not None:
        row[3].metric("Frente a su media de 200 sesiones", pct(entry.distance_to_sma),
                      help="Precio ajustado por splits sobre la media de las últimas 200 "
                           "sesiones.")
        state = ("cae con fuerza (bajo su media y −20 % o peor en 3 meses)" if entry.knife
                 else "muy estirada (+30 % o más sobre su media)" if entry.overextended
                 else "ni cae con fuerza ni está muy estirada")
        st.caption(
            f"**Antes de comprar:** 1 mes {pct(entry.return_1m)} · 3 meses "
            f"{pct(entry.return_3m)} · 12 meses {pct(entry.return_12m)} · hoy {state}. "
            "Probado en el S&P 500 (2018-2026) comparando cada fecha con el resto del índice: "
            "comprar una acción que cae con fuerza rindió 1,9 puntos menos a 6 meses, **sin "
            "significación** y con signo distinto según el periodo; una muy estirada rindió "
            "**más** (+8,5 puntos, casi todo desde 2022). Ninguna de las dos es regla: el "
            "precio solo no decide; la tesis y el 10-Q sí.")
    # Relative strength (§15.5 point 8): the percentile within the growth screen's whole
    # universe, at the screen's own price date — the company's return on that same day,
    # never today's return against the universe of another day. Context, not a signal.
    universe = app_data.growth_view(app_data.db_mtime())
    mine = universe[universe["cik"] == cik] if not universe.empty else universe
    if not mine.empty and pd.notna(mine.iloc[0]["rs_12m"]):
        r = mine.iloc[0]
        ranked = f"{int((universe['investable'] & universe['return_12m'].notna()).sum()):,}" \
            .replace(",", ".")
        six = (f"; a 6 meses {pct(r['return_6m'])}, percentil {r['rs_6m'] * 100:.0f}"
               if pd.notna(r["rs_6m"]) else "")
        st.caption(
            f"**Fuerza relativa** (cribado de crecimiento, precios del {r['price_date']}): a 12 "
            f"meses {pct(r['return_12m'])}, **percentil {r['rs_12m'] * 100:.0f}** frente a "
            f"{ranked} empresas invertibles (capitalización y volumen mínimos del "
            f"cribado){six}. "
            + "Contexto, no señal: el momentum a 12 meses cambió de signo en el S&P 500 "
            "entre periodos, así que haber subido más que el resto no dice que vaya a seguir.")

    events = app_data.events()
    confirmed = events[(events["category"] == "earnings") & (events["cik"] == cik)
                       & (events["is_estimated"].fillna(0).astype(int) == 0)
                       & (events["ts"].str[:10] <= as_of_iso)] if not events.empty else events
    reactions = pa.earnings_reactions(stock["tr"], views["SPY"]["tr"],
                                      list(confirmed["ts"]) if not confirmed.empty else [])
    if not reactions.empty:
        st.markdown("**Reacción a los resultados**")
        st.dataframe(color_by_sign(pd.DataFrame({
            "Resultados": reactions["date"],
            "Movimiento": reactions["move"].map(pct),
            "SPY": reactions["benchmark_move"].map(pct),
            "Por encima de SPY": reactions["excess"].map(pct),
        }).head(8), {"Movimiento": True, "Por encima de SPY": True}),
            hide_index=True, width="stretch")
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

# Who owns and who trades it (user request, 2026-09-29): institutions from the 13F data sets
# and a retail-participation proxy from FINRA. Two charts on purpose — quarterly holdings and
# weekly volume do not share an axis, and retail has no direction to show.
inst_obs = app_data.observations("sec_13f", app_data.db_mtime())
if not inst_obs.empty:
    basic = observations[observations["series_id"] == f"{cik}:basic_shares:q"] \
        if not observations.empty else pd.DataFrame()
    outstanding = (pd.Series(basic["value"].astype(float).to_numpy(),
                             index=pd.to_datetime(basic["ts"].astype(str).str[:10])).sort_index()
                   if len(basic) else None)
    registry_filers = app_data.filers()
    filer_names = dict(zip(registry_filers["cik"], registry_filers["name"])) \
        if not registry_filers.empty else {}
    whales = own.institutional(inst_obs, cik, as_of_iso, outstanding, filer_names)
    if whales is not None and not whales.quarters.empty:
        st.markdown("**Institucionales (13F)** · posiciones de los gestores con más de 100 M USD, "
                    "por trimestre")
        q = whales.quarters.assign(quarter=pd.to_datetime(whales.quarters["quarter"]))
        last = q[q["complete"]].iloc[-1] if q["complete"].any() else q.iloc[-1]
        cols = st.columns(3)
        cols[0].metric("En manos de fondos", pct(last["share_of_company"], decimals=0)
                       if pd.notna(last["share_of_company"]) else compact_amount(last["shares"]),
                       help="Acciones declaradas en 13F sobre las acciones básicas del "
                            "trimestre. Sin cifras de la SEC (NU), el número de acciones.")
        cols[1].metric("Fondos con posición", f"{int(last['holders']):,}".replace(",", "."))
        cols[2].metric("Cambio del trimestre", compact_amount(last["change"]) + " acc."
                       if pd.notna(last["change"]) else MISSING,
                       help="Acciones en manos de fondos frente al trimestre anterior: la "
                            "compra (o venta) neta institucional.")
        altair_chart(
            alt.Chart(q.dropna(subset=["change"])).mark_bar().encode(
                x=alt.X("quarter:T", title=None),
                y=alt.Y("change:Q", title="Compra (+) o venta (−) neta de fondos, acciones"),
                color=alt.condition("datum.change >= 0", alt.value("#2a78d6"),
                                    alt.value("#eb6834")),
                opacity=alt.condition("datum.complete", alt.value(1.0), alt.value(0.4)),
                tooltip=[alt.Tooltip("quarter:T", title="Trimestre", format="%Y-%m"),
                         alt.Tooltip("change:Q", title="Acciones", format=",.0f"),
                         alt.Tooltip("holders:Q", title="Fondos")],
            ).properties(height=180),
            width="stretch", amounts=True,
        )
        if not whales.movers.empty:
            top = whales.movers.sort_values("change")
            buyers = top.tail(5).iloc[::-1]
            sellers = top.head(5)
            left, right = st.columns(2)
            for column, title, frame in ((left, "Más compraron", buyers),
                                         (right, "Más vendieron", sellers)):
                column.markdown(f"*{title} · trimestre al {whales.latest}*")
                column.dataframe(pd.DataFrame({
                    "Gestor": frame["name"] + frame["entity_change"].map(
                        lambda flag: " · ¿cambio de entidad?" if flag else ""),
                    "Acciones": frame["change"].map(lambda v: compact_amount(v)),
                    "Ahora": frame["after"].map(lambda v: compact_amount(v) if v else "fuera"),
                }), hide_index=True, width="stretch")
        st.caption(
            " ".join(whales.notes)
            + " Posiciones largas declaradas 45 días después del cierre de cada trimestre; no "
              "incluye cortos, derivados ni gestores pequeños. El cambio es **neto**: no dice si "
              "entraron unos y salieron otros. «¿Cambio de entidad?»: una salida y una entrada "
              "de tamaño parecido con el mismo nombre suelen ser el mismo gestor presentando "
              "con otra sociedad (Pershing Square en 2026), no una venta y una compra. En "
              "empresas con varias clases de acción el porcentaje compara una clase con todas. "
              "Contexto, no señal: no se ha validado que anticipe nada.")

finra_weekly = app_data.observations("finra_otc", app_data.db_mtime())
if not finra_weekly.empty:
    daily = app_data.daily_volume(ticker, app_data.db_mtime())
    retail = own.retail_participation(finra_weekly, daily, ticker, as_of_iso)
    if not retail.empty:
        st.markdown("**Participación retail** · volumen semanal fuera de bolsa y fuera de "
                    "ATS, sobre el total")
        altair_chart(
            alt.Chart(retail.tail(156)).mark_line(color="#8e6bd8").encode(
                x=alt.X("week:T", title=None),
                y=alt.Y("otc_share:Q", title="Parte del volumen", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("week:T", title="Semana"),
                         alt.Tooltip("otc_share:Q", title="Fuera de bolsa (retail)",
                                     format=".1%"),
                         alt.Tooltip("ats_share:Q", title="ATS", format=".1%")],
            ).properties(height=160),
            width="stretch",
        )
        st.caption(
            f"Última semana publicada: {retail.iloc[-1]['week']:%Y-%m-%d}, "
            f"{pct(retail.iloc[-1]['otc_share'], decimals=0)} del volumen. Ese volumen lo "
            "ejecutan sobre todo mayoristas (Citadel Securities, Virtu, G1…) que reciben las "
            "órdenes minoristas, así que mide **cuánto opera el retail, no si compra o vende**: "
            "la dirección solo existe en datos de pago. Un pico suele acompañar a noticias o a "
            "movimientos grandes del precio. FINRA publica cada semana con 2-4 semanas de "
            "retraso.")

# --- 5. Value accrual -----------------------------------------------------------------

st.divider()
st.subheader("5 · ¿Cómo llega eso al accionista?")
st.caption(
    "El concepto rector (§2): una empresa puede crecer 30% al año y destruir valor por "
    "acción si diluye 35%. Esta sección es el análogo de la vista de unlocks."
)

row = st.columns(4)
# Dilution is the unlock of §2: more shares is red, fewer is green.
row[0].metric(
    "Dilución interanual (diluidas)", pct(accrual.dilution_yoy, decimals=2),
    **verdict_delta(tone(accrual.dilution_yoy, higher_is_better=False),
                    "diluye" if (accrual.dilution_yoy or 0) > 0 else "reduce acciones"),
    help="Acciones diluidas medias del ejercicio frente al anterior. Incluye opciones y "
         "convertibles… solo en los años con beneficio: con pérdida, la norma las iguala a "
         "las básicas (§9.15), así que salta al cruzar de pérdida a ganancia.",
)
row[1].metric(
    "Acciones en circulación, interanual",
    pct(growth_ps.shares_yoy, decimals=2) if growth_ps.basis == "basic_shares" else MISSING,
    **verdict_delta(tone(growth_ps.shares_yoy, higher_is_better=False)
                    if growth_ps.basis == "basic_shares" else None,
                    "diluye" if (growth_ps.shares_yoy or 0) > 0 else "reduce acciones"),
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
shares_row = [k for k in grid_ps if k.startswith("Acciones")]
st.dataframe(color_rows_by_sign(pd.DataFrame(grid_ps).T.replace({"None": MISSING}), {
    **{label: True for label in labels_ps.values()},
    **{label: False for label in shares_row}}), width="stretch")
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
    **verdict_delta(tone(accrual.roic), "crea rentabilidad" if (accrual.roic or 0) > 0
                    else "destruye capital"),
    help="NOPAT sobre capital invertido, con la tasa impositiva OBSERVADA en los filings, "
         "no la estatutaria. Capital invertido = patrimonio + deuda largo plazo − caja.",
)

row = st.columns(4)
row[0].metric("Recompras brutas (TTM, USD)", compact_amount(accrual.buybacks_ttm),
              help=reported_amount(accrual.buybacks_ttm))
row[1].metric("Recompras netas (TTM, USD)", compact_amount(accrual.net_buybacks_ttm),
              **verdict_delta(tone(accrual.net_buybacks_ttm), "devuelve capital"
                              if (accrual.net_buybacks_ttm or 0) > 0
                              else "emite más de lo que recompra"),
              help="Recompras − emisión. La bruta engaña si el SBC la anula. "
                   + reported_amount(accrual.net_buybacks_ttm))
# The count grew for most companies that pay in stock: a tile reading "retired: −21 M"
# made the reader do the sign. The label says which way it went.
issued = accrual.shares_removed is not None and accrual.shares_removed < 0
row[2].metric(
    "Acciones emitidas netas (año)" if issued else "Acciones retiradas (año)",
    compact_amount(abs(accrual.shares_removed) if issued else accrual.shares_removed),
    **verdict_delta(tone(accrual.shares_removed), "diluye" if issued else "recompra"),
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
            y=alt.Y("value:Q", title="Acciones", scale=alt.Scale(zero=False),
                    axis=alt.Axis(grid=True, gridOpacity=0.25)),
            tooltip=[alt.Tooltip("date:T", title="Ejercicio"),
                     alt.Tooltip("value:Q", title="Acciones", format=",.0f")],
        )
        .properties(height=200),
        width="stretch", amounts=True,
    )

# --- 4. Filings and calendar ----------------------------------------------------------

st.divider()
st.subheader("6 · Presentaciones y calendario")

# Insider buying and selling (user request, 2026-09-29): context, not a signal — the studies
# found purchases do not precede better returns (RESEARCH.md §2.42, §2.62). Neutral colours
# on purpose: green would say "good for the holder", which the evidence does not support.
form4 = app_data.observations("sec_form4", app_data.db_mtime())
insiders = ia.assess(form4, cik, as_of_iso) if not form4.empty else None
if insiders is not None and not insiders.monthly.empty:
    st.markdown("**Compras y ventas de directivos** · Form 4, consejeros y directivos, en bolsa")
    order = list(ia.KINDS.values())
    palette = ["#2a78d6", "#eb6834", "#f2b48a", "#9aa5b1"]
    altair_chart(
        alt.Chart(insiders.monthly).mark_bar().encode(
            x=alt.X("month:T", title=None),
            y=alt.Y("value:Q", title="USD al mes (ventas hacia abajo)", stack=True),
            color=alt.Color("label:N", sort=order,
                            scale=alt.Scale(domain=order, range=palette),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("month:T", title="Mes", format="%Y-%m"),
                     alt.Tooltip("label:N", title="Tipo"),
                     alt.Tooltip("value:Q", title="USD", format=",.0f")],
        ).properties(height=200),
        width="stretch", amounts=True,
    )
    def usd(value: float) -> str:
        return reported_amount(value) if value else "ninguna"

    st.caption(
        f"Últimos 12 meses: compras {usd(insiders.buy_12m)}"
        + (f" ({insiders.buyers_12m} compra(s) de directivos)" if insiders.buyers_12m else "")
        + f" · ventas {usd(insiders.sell_12m)}, de ellas discrecionales "
        f"{usd(insiders.discretionary_12m)}. Última presentación: "
        f"{insiders.last_filed}. **Contexto, no señal:** en los estudios del panel, las compras de "
        "directivos no precedieron mejor rentabilidad, ni en el S&P 500 ni en las pequeñas "
        "(suelen llegar tras caídas fuertes). Las ventas con plan 10b5-1 se programan meses "
        "antes y dicen poco: los directivos venden acciones que cobran como sueldo. Lo que más "
        "informa es una compra, o una ola de ventas discrecionales. Fuera: ejercicios de "
        "opciones, retenciones por impuestos, regalos y fondos que solo son dueños del 10 %.")

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
    # Hand-written catalysts (§15.5 point 9): local only, like settings.local.yaml.
    if not public:
        today_ = dt.date.fromisoformat(as_of_iso)
        mine_ = [c for c in cat.from_settings(settings.raw) if c.ticker == ticker]
        ahead = cat.upcoming(mine_, today_)
        st.markdown("**Catalizadores**")
        for c in ahead:
            st.warning(f"**{c.when}** ({cat.countdown(c, today_)}) — {c.kind_label}: "
                       f"{c.label}. Fuente: {c.source}.")
        gone = [c for c in mine_ if c not in ahead]
        if gone:
            st.caption("Pasados: " + " · ".join(f"{c.end} {c.kind_label.lower()}"
                                                 for c in gone))
        if not mine_:
            st.caption("Ninguno escrito. Las fechas de la FDA (PDUFA), de resultados de un "
                       "ensayo o del fin de un lockup no están en ninguna fuente gratuita "
                       "estructurada: se escriben a mano en `catalysts` de "
                       "`settings.local.yaml`, con la fuente de cada fecha.")

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
