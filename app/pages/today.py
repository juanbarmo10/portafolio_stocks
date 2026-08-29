"""🏠 Hoy — level 1 of the checklist: is there risk appetite? (CLAUDE.md section 2).

Everything on this page is point-in-time: values are shown as they were *publicly known*
on the selected date, never as they were later revised (section 9.4). That is why the
table carries a publication column and a staleness column — a CPI print is six weeks old
by construction, and a panel that hides that invites reading it as today's inflation.

The other three questions of section 2 are listed but not answered yet: their state says
which phase builds them. An empty chart would be worse than an explicit hole.
"""

from __future__ import annotations

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from app.data import last_ingest, macro_observations
from core.config import load_settings
from transform import macro

# Categorical slots 1 and 2 of the reference palette — a validated adjacent pair.
# Colour carries series identity only; every number on the page wears text ink.
SERIES_COLORS = ["#2a78d6", "#eb6834"]

settings = load_settings()
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

# --- 1. Risk appetite ----------------------------------------------------------------
st.subheader("1 · ¿Hay apetito por riesgo?")

series_ids = [r["series_id"] for r in readings_cfg]
labels = {r["series_id"]: r.get("label", r["series_id"]) for r in readings_cfg}
units = {r["series_id"]: r.get("unit") for r in readings_cfg}
notes = {r["series_id"]: r.get("note") for r in readings_cfg}

snapshot = macro.snapshot(df, series_ids, as_of, lookback_days=lookback_days)
table = pd.DataFrame([
    {
        "Serie": labels[r.series_id],
        "Valor": r.value,
        f"Cambio {lookback_days}d": r.change,
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
        "Valor": st.column_config.NumberColumn(format="%.3f"),
        f"Cambio {lookback_days}d": st.column_config.NumberColumn(
            format="%+.3f",
            help="Cambio absoluto en las unidades de la serie, con ambos extremos "
                 "tomados point-in-time. Vacío = no hay punto de comparación.",
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
    st.altair_chart(
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
    "Baa − 10a cubre el histórico largo (desde 1986) para lo que necesite serie larga."
)

# --- The rest of the checklist -------------------------------------------------------
st.divider()
st.subheader("Resto del checklist")
st.markdown(
    """
| Pregunta | Estado |
|---|---|
| **2 · ¿El mercado está sano o es un rally estrecho?** | Sin construir — fase 3 (amplitud, RSP/SPY, rotación, estructura VIX) |
| **3 · ¿La tesis de la empresa sigue viva?** | Sin construir — fase 2 (fundamentales SEC XBRL, dilución, earnings) |
| **4 · ¿Toca ejecutar según el plan?** | Sin construir — falta conectar la cuenta de IBKR (ver *Cartera*) |
"""
)
st.caption(
    "Regla dura de §2: si los niveles 1 y 2 están claramente en rojo, no se compra "
    "aunque el nivel 3 sea perfecto. Con el nivel 2 sin construir, esa regla todavía "
    "no se puede aplicar entera."
)
