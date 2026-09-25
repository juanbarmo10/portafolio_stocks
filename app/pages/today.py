"""🏠 Hoy — level 1 of the checklist: is there risk appetite? (CLAUDE.md section 2).

Everything on this page is point-in-time: values are shown as they were *publicly known*
on the selected date, never as they were later revised (section 9.4). That is why the
table carries a publication column and a staleness column — a CPI print is six weeks old
by construction, and a panel that hides that invites reading it as today's inflation.

Below it, the landing page summarizes the rest of the checklist — the regime verdict, the
written universe, and what is coming in the next two weeks — so the one page opened first
says what needs attention. (Until 2026-09-25 that block still read "sin construir" for
three levels that had been built long before.)
"""

from __future__ import annotations

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from app import data as app_data
from app.data import last_ingest, macro_observations
from app.format import altair_chart, MISSING, number
from core.config import load_settings
from ingest.macro_calendar import current_calendar
from transform import macro
from transform import portfolio as port
from transform import regime as rg

# Categorical slots 1 and 2 of the reference palette — a validated adjacent pair.
# Colour carries series identity only; every number on the page wears text ink.
SERIES_COLORS = ["#2a78d6", "#eb6834"]

settings = load_settings()
public = settings.public_mode
level1 = settings.raw.get("panel", {}).get("level1", {})
readings_cfg = level1.get("readings", [])
charts_cfg = level1.get("charts", [])
lookback_days = int(level1.get("lookback_days", 30))

st.title("🏠 Hoy")

df = macro_observations()

if df.empty:
    st.warning(
        "La base de datos no tiene series macro todavía. Ejecuta `python run_ingest.py` "
        "con `FRED_API_KEY` configurada en `config/.env`."
    )
    st.stop()

# --- Controls ------------------------------------------------------------------------
# The date picker is the point-in-time control: it answers "what did the panel say back
# then?" without ever showing data that was not public yet.
latest_ts = pd.to_datetime(df["ts"]).max().date()
col_date, col_meta = st.columns([1, 3])
with col_date:
    as_of = st.date_input(
        "Ver el panel al día",
        value=dt.date.today(),
        max_value=dt.date.today(),
        help="Solo se muestra lo que ya estaba publicado en esa fecha (§9.4).",
    )
with col_meta:
    ingest_at = last_ingest()
    st.caption(
        f"Última ingesta: **{ingest_at:%Y-%m-%d %H:%M}**  ·  "
        f"dato macro más reciente en la base: **{latest_ts}**  ·  "
        "resolución diaria, panel *pull* (sin auto-refresco)."
        if ingest_at else "La base de datos aún no se ha escrito."
    )

# --- What is coming (first: it is what can change today's decision) -----------
st.subheader("Lo que viene — 14 días")
events = app_data.events()
horizon = pd.Timestamp(as_of) + pd.Timedelta(days=14)
upcoming = []
calendar = current_calendar(events)
for row in calendar.to_dict("records"):
    when = pd.Timestamp(row["ts"])
    if pd.Timestamp(as_of, tz="UTC") <= when <= horizon.tz_localize("UTC"):
        upcoming.append((when.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
                         row["label"], "hora de Nueva York"))
# Results of the researched companies, and — privately — of what is held: publicly the
# list of positions is exactly what must not show (RESEARCH.md §2.33).
ciks = {str(c["cik"]).zfill(10): str(c.get("ticker")) for c in settings.researched_companies if c.get("cik")}
if not public:
    companies = app_data.companies()
    by_ticker = dict(zip(companies["ticker"], companies["cik"])) if not companies.empty else {}
    for ticker in port.latest_positions(app_data.account_observations())["ticker"]:
        if by_ticker.get(ticker):
            ciks.setdefault(by_ticker[ticker], ticker)
if not events.empty:
    earnings = events[(events["category"] == "earnings") & events["cik"].isin(ciks)]
    for row in earnings.to_dict("records"):
        day = pd.Timestamp(str(row["ts"])[:10])
        if pd.Timestamp(as_of) <= day <= horizon:
            kind = "estimada" if int(row.get("is_estimated") or 0) else "confirmada"
            upcoming.append((day.strftime("%Y-%m-%d"), f"Resultados de {ciks[row['cik']]}",
                             f"fecha {kind}"))
if upcoming:
    st.dataframe(pd.DataFrame(sorted(upcoming), columns=["Cuándo", "Qué", "Nota"]),
                 hide_index=True, width="stretch")
    st.caption("Con un dato macro de alto impacto en las 48 h siguientes, §2 propone "
               "posponer el aporte; con resultados en 5 días, no abrir posición sin decisión "
               "explícita. Las alertas de Telegram avisan de ambos.")
else:
    st.caption("Nada de alto impacto en los próximos 14 días.")

# --- The rest of the checklist -------------------------------------------------------
st.subheader("El checklist")

view = app_data.regime_view(app_data.db_mtime())
if view is None:
    regime_state = "sin datos para el semáforo todavía"
else:
    r = view["reading"]
    # verdict_label already carries its icon (it read "🟡 🟡 Mixto" with one added here).
    regime_state = (f"{rg.verdict_label(r.verdict)} al {r.date}: {r.on} a favor, "
                    f"{r.off} en contra, {r.neutral} neutros")
researched = settings.researched_companies
theses = f"{len(settings.tracked_companies)} tesis escritas, " \
         f"{len(settings.watchlist_companies)} empresas en estudio"
rows = [
    ("2 · ¿El mercado está sano o es un rally estrecho?", f"{regime_state} — 📈 Mercado"),
    ("3 · ¿La tesis de la empresa sigue viva?", f"{theses} — 🏢 Empresa"),
]
if not public:
    rows.append(("4 · ¿Toca ejecutar según el plan?", "posiciones, coste y resultados — 🔬 Cartera"))
st.markdown("| Pregunta | Estado |\n|---|---|\n"
            + "\n".join(f"| **{q}** | {a} |" for q, a in rows))
st.caption(
    "Regla dura de §2: si los niveles 1 y 2 están claramente en rojo, no se compra aunque "
    "el nivel 3 sea perfecto. Es un freno de disciplina: la validación (fase 4) mostró que "
    "no anticipa la rentabilidad, pero sí caídas más hondas en el mes siguiente."
)

st.divider()
# --- 1. Risk appetite ----------------------------------------------------------------
st.subheader("1 · ¿Hay apetito por riesgo?")

series_ids = [r["series_id"] for r in readings_cfg]
labels = {r["series_id"]: r.get("label", r["series_id"]) for r in readings_cfg}
units = {r["series_id"]: r.get("unit") for r in readings_cfg}
notes = {r["series_id"]: r.get("note") for r in readings_cfg}

snapshot = macro.snapshot(df, series_ids, as_of, lookback_days=lookback_days)
def signed(value: float | None) -> str:
    """Change with its sign, in Spanish notation."""
    if value is None or pd.isna(value):
        return MISSING
    return ("+" if value >= 0 else "") + number(value, decimals=3)


# The readings that decide level 1 at a glance (§15.2: the table of eleven series was noise
# to someone choosing stocks). The rest stays one click away, not gone.
key_ids = [s for s in level1.get("key_readings", []) if s in labels]
by_id = {r.series_id: r for r in snapshot}
if key_ids:
    cols = st.columns(len(key_ids))
    for col, sid in zip(cols, key_ids):
        reading = by_id.get(sid)
        col.metric(labels[sid], number(reading.value, decimals=2) if reading else MISSING,
                   delta=None if reading is None or reading.change is None
                   else f"{'+' if reading.change >= 0 else ''}"
                        f"{number(reading.change, decimals=2)} en {lookback_days} d",
                   delta_color="off", help=notes.get(sid) or None)
    st.caption("Qué mirar: el spread de crédito **ensanchándose** es el mejor aviso temprano "
               "de aversión al riesgo; la curva negativa, recesión descontada; el dólar fuerte "
               "y el VIX alto, estrés. El veredicto conjunto está en 📈 Mercado.")

with st.expander("Las series macro, una por una, y sus gráficas"):
    # Formatted as text so the table reads 2,730 like the rest of the panel, not 2.730.
    table = pd.DataFrame([
        {
            "Serie": labels[r.series_id],
            "Valor": number(r.value, decimals=3),
            f"Cambio {lookback_days}d": signed(r.change),
            "Unidad": units.get(r.series_id),
            "Fecha de referencia": r.ts,
            "Publicado": r.ts_release,
            "Antigüedad (d)": r.staleness_days,
            "Nota": notes.get(r.series_id) or "",
        }
        for r in snapshot
    ])

    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            f"Cambio {lookback_days}d": st.column_config.TextColumn(
                help="Cambio absoluto en las unidades de la serie, con ambos extremos "
                     "tomados point-in-time. «—» = no hay punto de comparación.",
            ),
            "Antigüedad (d)": st.column_config.NumberColumn(
                help="Días entre la fecha de referencia del dato y la fecha del panel.",
            ),
        },
    )

    missing = [r.series_id for r in snapshot if r.value is None]
    if missing:
        st.info(
            "Sin dato publicado a esa fecha: " + ", ".join(labels.get(m, m) for m in missing)
            + ". Se muestra el hueco; no se estima (§12)."
        )

    # --- Charts --------------------------------------------------------------------------
    known = macro.point_in_time(df, as_of)
    known = known.assign(date=pd.to_datetime(known["ts"]).dt.date)

    window_years = st.select_slider(
        "Ventana de las gráficas", options=[1, 2, 3, 5, 10], value=3,
        format_func=lambda y: f"{y} año{'s' if y > 1 else ''}",
    )
    window_start = as_of - dt.timedelta(days=365 * window_years)

    for chart_cfg in charts_cfg:
        chart_series = chart_cfg.get("series", [])
        data = known[
            known["series_id"].isin(chart_series)
            & (known["date"] >= window_start)
            & (known["date"] <= as_of)
        ].copy()
        if data.empty:
            st.caption(f"{chart_cfg.get('title', '')}: sin datos en la ventana elegida.")
            continue

        data["Serie"] = data["series_id"].map(lambda s: labels.get(s, s))
        # A single series needs no legend — the title names it (dataviz: accessibility pass).
        n_series = data["Serie"].nunique()

        encoding = {
            "x": alt.X("date:T", title=None, axis=alt.Axis(grid=False)),
            "y": alt.Y(
                "value:Q",
                title=chart_cfg.get("unit"),
                scale=alt.Scale(zero=False),
                axis=alt.Axis(grid=True, gridOpacity=0.25),
            ),
            "tooltip": [
                alt.Tooltip("Serie:N"),
                alt.Tooltip("date:T", title="Fecha"),
                alt.Tooltip("value:Q", title="Valor", format=".3f"),
                alt.Tooltip("ts_release:N", title="Publicado"),
            ],
        }
        if n_series > 1:
            encoding["color"] = alt.Color(
                "Serie:N",
                scale=alt.Scale(
                    domain=[labels.get(s, s) for s in chart_series],
                    range=SERIES_COLORS[:len(chart_series)],
                ),
                legend=alt.Legend(orient="top", title=None),
            )

        line = alt.Chart(data).mark_line(strokeWidth=2).encode(**encoding)
        layers = [line]

        if chart_cfg.get("zero_rule"):
            # Zero is the fact here (an inverted curve), not a threshold anyone chose.
            layers.append(
                alt.Chart(pd.DataFrame({"y": [0.0]}))
                .mark_rule(strokeDash=[4, 4], strokeWidth=1, opacity=0.6)
                .encode(y="y:Q")
            )

        st.markdown(f"**{chart_cfg.get('title', '')}**")
        altair_chart(
            alt.layer(*layers).interactive(bind_y=False).properties(height=260),
            width="stretch",
        )

        if n_series == 1:
            color = SERIES_COLORS[0]
            st.markdown(
                f"<span style='color:{color}'>●</span> "
                f"{labels.get(chart_series[0], chart_series[0])}",
                unsafe_allow_html=True,
            )

    st.caption(
        "⚠️ El spread HY (BAMLH0A0HYM2) solo existe en FRED desde el 2023-08-29: es una "
        "ventana rodante por licencia de ICE, no un límite del point-in-time. La prima "
        "Baa − 10a tiene serie desde 1986, pero su historia point-in-time empieza en 2014."
    )

