"""📈 Mercado — level 2, and the regime light that combines it with level 1 (section 2).

"Is the market healthy, or is it a narrow rally?" — and, one step further, the verdict
that can block a purchase. Section 2's hierarchy: if levels 1 and 2 are clearly red,
nothing is bought however good the company looks.

Everything on this page is read point-in-time and at daily resolution. Nothing here
refreshes by itself and nothing moves intraday (section 2: pull, not push).

**Public since 2026-09-24, by the user's decision.** Everything here is public market data
except one block: short interest, which lists the held tickers alongside the ones under
study, and so reveals positions. That block renders only in private mode. The page's one
dollar figure — the Fed's net liquidity — is a published macro aggregate, the same
asymmetry the company page has (RESEARCH.md section 2.16): the account is protected, the
market is not.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import MISSING, number, pct, reported_amount
from core.config import load_settings
from transform import regime as rg
from transform.breadth import (
    breadth_series,
    defensive_rotation,
    equal_weight_ratio,
    sector_breadth,
    splits_by_ticker,
    usable_from,
    wide_closes,
)
from transform.macro import latest

LINE = "#2a78d6"
VERDICT_COLORS = {
    rg.RISK_ON: "#3a9d5d", rg.NEUTRAL: "#e0a526",
    rg.RISK_OFF: "#d64545", rg.INSUFFICIENT: "#b8b8b8",
}
VERDICT_SHORT = {
    rg.RISK_ON: "risk-on", rg.NEUTRAL: "mixto", rg.RISK_OFF: "risk-off",
    rg.INSUFFICIENT: "insuficiente",
}

settings = load_settings()
public = settings.public_mode
L2 = settings.raw["panel"]["level2"]
R, B, SB = L2["regime"], L2["breadth"], L2["sector_breadth"]
EW, ROT = L2["equal_weight"], L2["rotation"]
ETFS = rg.regime_tickers(L2)

st.title("📈 Mercado")
st.caption("Nivel 2 — ¿está sano el mercado o es un rally estrecho? — y el semáforo que lo "
           "combina con el nivel 1. Resolución diaria; nada se actualiza solo (§2).")


@st.cache_data(show_spinner="Calculando el semáforo…")
def regime_view(mtime: float) -> dict | None:
    """Everything the light needs, cached until the database changes."""
    built = rg.build(app_data.fred_observations(), app_data.prices_for(ETFS),
                     app_data.corporate_actions(), L2)
    if built is None:
        return None
    return {
        "reading": rg.reading_at(built.calendar[-1], built.rules, built.components,
                                 built.votes, built.frame),
        "frame": built.frame[["verdict", "available"]],
        "evaluation": rg.evaluate(built.frame, built.benchmark),
        "closes": built.closes, "splits": built.splits,
    }


@st.cache_data(show_spinner="Calculando la amplitud por empresas…")
def constituent_view(mtime: float) -> list | None:
    """Breadth over the real S&P 500 membership of each day, with its coverage."""
    intervals = app_data.universe_membership()
    if intervals.empty:
        return None
    tickers = sorted(set(intervals["ticker"]))
    prices = app_data.prices_for(tickers)
    if prices.empty:
        return None
    return breadth_series(
        intervals, wide_closes(prices), splits_by_ticker(app_data.corporate_actions()),
        window=B["window"], threshold=B["coverage_threshold"],
        min_window_fraction=B["min_window_fraction"],
    )


view = regime_view(app_data.db_mtime())

if view is None:
    st.info(
        "Sin datos para el semáforo todavía. Necesita la macro y los precios de las "
        "referencias: `python run_ingest.py --only fred prices`. Para la amplitud por "
        "empresas, además, `--only universe`."
    )
    st.stop()

# --- 1. The light ---------------------------------------------------------------------

reading = view["reading"]
st.subheader("1 · Semáforo de régimen")
st.markdown(f"### {rg.verdict_label(reading.verdict)}")
st.caption(
    f"Al cierre del {reading.date}: **{reading.on}** componentes dicen risk-on, "
    f"**{reading.off}** risk-off, **{reading.neutral}** neutros, de **{reading.available}** "
    f"que votan. Pesos iguales, mayoría simple. **Bloquea, no dispara:** en rojo no se "
    "compra aunque la empresa sea perfecta (§2); en verde no significa comprar, significa "
    "que el entorno no lo impide."
)

VOTE_TEXT = {1: "🟢 risk-on", 0: "🟡 neutro", -1: "🔴 risk-off", None: "⚪ se abstiene"}


def shown(key: str, value: float | None) -> str:
    """Each component in its own unit, in Spanish notation."""
    if value is None:
        return MISSING
    if key == "net_liquidity":
        return reported_amount(value * 1e6)          # H.4.1 is in millions of USD
    if key == "breadth":
        return pct(value, decimals=0)
    return number(value, decimals=3 if key in ("equal_weight", "volatility") else 2)


rows = []
for vote in reading.votes:
    rows.append({
        "Componente": vote.label,
        "Valor": shown(vote.key, vote.value),
        "Comparado con": shown(vote.key, vote.reference),
        "Voto": VOTE_TEXT[vote.vote],
        "Dato del": vote.value_date or MISSING,
        "Por qué": vote.detail,
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(
    "Reglas **escritas antes de mirar ningún resultado** y no retocadas después (§9.7). "
    "Nivel contra una referencia natural (NFCI 0 = condiciones medias; curva y "
    "volatilidad invierten en 0 y 1; amplitud 50 % = mayoría) o tendencia contra su media "
    "de 200 sesiones. Todo leído tal como se había publicado cada día (§9.4)."
)

# --- 2. How it behaved -------------------------------------------------------------------

st.subheader("2 · Cómo se comportó en el pasado")
ev = view["evaluation"]
cols = st.columns(3)
cols[0].metric("Rojo en correcciones", pct(ev["red_in_correction"]),
               help="Sesiones a ≥10 % del máximo previo en las que el semáforo estaba en rojo.")
cols[1].metric("Rojo o ámbar en correcciones",
               pct(None if ev["red_in_correction"] is None
                   else ev["red_in_correction"] + ev["amber_in_correction"]))
cols[2].metric("Rojo en calma (falsas alarmas)", pct(ev["red_in_calm"]),
               help="Sesiones a menos del 5 % del máximo en las que estaba en rojo.")

episodes = pd.DataFrame([{
    "Pico": e["peak"], "Suelo": e["trough"], "Caída": pct(e["depth"]),
    "Primer rojo": e["first_red"] or ("—" if e["judged"] else "sin veredicto"),
    "% en rojo hasta el suelo": pct(e["red_share_to_trough"]),
    "Componentes votando": e["components_voting"],
} for e in ev["episodes"]])
if not episodes.empty:
    st.dataframe(episodes, hide_index=True, width="stretch")
st.caption(
    "**Lectura honesta:** casi nunca da falsa alarma, y se pone en rojo en la mayoría de "
    "las correcciones desde 2012 — pero **confirma, no anticipa**: llega tarde, y en las "
    "correcciones pasa más tiempo en ámbar que en rojo. Antes de 2012 no hay veredicto: con "
    "datos point-in-time solo votan los dos componentes construidos con ETF. En esta "
    "muestra, **NFCI no votó risk-off nunca** y **RSP/SPY no distingue** calma de "
    "corrección; no se han cambiado tras verlo — sería ajustar las reglas al pasado. Es una "
    "descripción, no una prueba: la validación estadística es la fase 4 (RESEARCH.md §2.27)."
)

history = view["frame"].reset_index().rename(columns={"index": "date"})
history["Veredicto"] = history["verdict"].map(VERDICT_SHORT)
history = history[history["verdict"] != rg.INSUFFICIENT]
if not history.empty:
    st.altair_chart(
        alt.Chart(history).mark_rect().encode(
            x=alt.X("date:T", title=None),
            color=alt.Color(
                "Veredicto:N",
                scale=alt.Scale(domain=[VERDICT_SHORT[k] for k in VERDICT_COLORS],
                                range=list(VERDICT_COLORS.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[alt.Tooltip("date:T", title="Sesión"), "Veredicto:N"],
        ).properties(height=60),
        width="stretch",
    )

# --- 3. Breadth ----------------------------------------------------------------------------

st.subheader("3 · Amplitud")
readings = constituent_view(app_data.db_mtime())
if not readings:
    st.caption("Amplitud por empresas sin calcular: `python run_ingest.py --only universe`.")
else:
    last = readings[-1]
    start = usable_from(readings)
    cols = st.columns(3)
    if last.source_hole:
        cols[0].metric("Empresas sobre su media de 200", MISSING,
                       help="Hueco en la fuente ese día: no dice nada del índice.")
    else:
        cols[0].metric("Empresas sobre su media de 200", pct(last.breadth))
    cols[1].metric("Cobertura", pct(last.coverage),
                   help="Miembros de ese día sobre los que descansa el cálculo.")
    cols[2].metric("Serie fiable desde", start or MISSING)
    if not last.meets_threshold:
        st.warning(
            "La lectura de hoy no llega a la cobertura mínima "
            f"({pct(B['coverage_threshold'], decimals=0)}): no describe el índice entero."
        )
    series = pd.DataFrame([
        {"date": r.date, "value": r.breadth} for r in readings
        if start and r.date >= start and r.breadth is not None and not r.source_hole
    ])
    if not series.empty:
        series["date"] = pd.to_datetime(series["date"])
        st.altair_chart(
            alt.Chart(series).mark_line(color=LINE).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("value:Q", title="% de miembros", axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1])),
            ).properties(height=220),
            width="stretch",
        )
    st.caption(
        "Sobre la composición real del S&P 500 de cada día (`fja05680/sp500`, MIT). Los "
        "precios de las empresas que ya salieron no son gratis, así que cada lectura lleva "
        "su cobertura y solo vale desde el 85 %. Hacia atrás, el semáforo usa la amplitud "
        "sectorial, que no tiene ese sesgo (§9.5)."
    )

closes, splits = view["closes"], view["splits"]
sb = sector_breadth(closes, splits, SB["sectors"], window=SB["window"],
                    min_window_fraction=SB["min_window_fraction"]).dropna(subset=["share"])
if not sb.empty:
    sb = sb.assign(date=pd.to_datetime(sb["date"]), share=sb["share"].astype(float))
    st.markdown(f"**Amplitud sectorial** — {int(sb['above'].iloc[-1])} de "
                f"{len(SB['sectors'])} sectores sobre su media de 200 sesiones")
    st.altair_chart(
        alt.Chart(sb).mark_line(color=LINE).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("share:Q", title="% de sectores", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
        ).properties(height=180),
        width="stretch",
    )

# --- 4. Narrowness and rotation ---------------------------------------------------------------

st.subheader("4 · Estrechez y rotación")
left, right = st.columns(2)
ew = equal_weight_ratio(closes, splits, equal=EW["equal"], cap=EW["cap"])
rot = defensive_rotation(closes, splits, defensive=ROT["defensive"], cyclical=ROT["cyclical"])
with left:
    st.markdown(f"**{EW['equal']} / {EW['cap']}** — equiponderado frente a capitalización")
    if not ew.empty:
        st.altair_chart(
            alt.Chart(ew.assign(date=pd.to_datetime(ew["date"]))).mark_line(color=LINE)
            .encode(x=alt.X("date:T", title=None),
                    y=alt.Y("ratio:Q", title="cociente", scale=alt.Scale(zero=False)))
            .properties(height=200),
            width="stretch",
        )
    st.caption("Cae = la empresa media se queda atrás frente a las gigantes. Se lee por su "
               "dirección, no por su nivel.")
with right:
    st.markdown("**Rotación defensiva** — " + " + ".join(ROT["defensive"]) + " frente a "
                + " + ".join(ROT["cyclical"]))
    if not rot.empty:
        st.altair_chart(
            alt.Chart(rot.assign(date=pd.to_datetime(rot["date"]))).mark_line(color=LINE)
            .encode(x=alt.X("date:T", title=None),
                    y=alt.Y("rotation:Q", title="base 100", scale=alt.Scale(zero=False)))
            .properties(height=200),
            width="stretch",
        )
    st.caption("Sube = los defensivos baten a los cíclicos: el mercado descuenta "
               "desaceleración. Medias geométricas, no suma de precios (THEORY.md §2.3).")

# --- 5. Volatility ----------------------------------------------------------------------------

st.subheader("5 · Estructura de volatilidad")
fred = app_data.fred_observations()
vix = latest(fred, "VIXCLS", pd.Timestamp.today())
vix3m = latest(fred, "VXVCLS", pd.Timestamp.today())
cols = st.columns(3)
cols[0].metric("VIX (30 días)", number(vix.value))
cols[1].metric("VIX3M (3 meses)", number(vix3m.value))
if vix.value is not None and vix3m.value:
    ratio = vix.value / vix3m.value
    cols[2].metric("VIX / VIX3M", number(ratio, decimals=3),
                   help="Por debajo de 1 es lo normal. Por encima, el miedo a corto plazo "
                        "supera al de largo: estrés presente.")
st.caption(f"Datos del {vix.ts or MISSING}. Point-in-time del VIX3M solo desde 2014.")

# --- 6. Short interest (private) ----------------------------------------------------------------

# Private only: the list is the watchlist plus whatever FINRA was asked about because it
# is HELD, so publishing it would publish the positions.
if not public:
    st.subheader("6 · Interés corto")
    tickers = sorted({str(c["ticker"]) for c in settings.researched_companies if c.get("ticker")})
    finra = app_data.observations("finra", app_data.db_mtime())
    # FINRA was asked about the written universe AND what is held (its ingester adds the
    # held tickers); showing every ticker it has data for covers both.
    with_data = {s.split(":")[0] for s in finra["series_id"]} if not finra.empty else set()
    tickers = sorted(set(tickers) | with_data)
    rows = []
    for t in tickers:
        si = latest(finra, f"{t}:short_interest", pd.Timestamp.today())
        dtc = latest(finra, f"{t}:days_to_cover", pd.Timestamp.today())
        if si.value is None:
            continue
        rows.append({"Empresa": t, "Acciones en corto": number(si.value, decimals=0),
                     "Días para cubrir": number(dtc.value),
                     "Liquidación": si.ts, "Publicado (derivado)": si.ts_release})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "FINRA, dos veces al mes. **La fecha de publicación es derivada**: FINRA no la "
            "publica, y se estima con el rezago máximo medido en su calendario (8 días "
            "hábiles), tarde a propósito (THEORY.md §7.4)."
        )
    else:
        st.caption("Sin interés corto todavía: `python run_ingest.py --only short_interest`.")
