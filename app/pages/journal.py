"""📓 Diario — every decision, what was expected, and what happened (CLAUDE.md §15.1.10).

Private: it is the account's trades and the user's own reasoning. Not in ``PUBLIC_PAGES``.
The logic lives in ``transform/journal.py``; the reasons live in ``settings.local.yaml``
under ``journal`` — the panel never writes them, it only reads them back.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import MISSING, money, number, pct
from core.config import load_settings
from transform import journal as jr
from transform import regime as rg
from transform.adjustments import total_return_index
from transform.price_action import as_series

settings = load_settings()
cfg = dict(settings.raw.get("panel", {}).get("journal") or {})
entries = list(settings.raw.get("journal") or [])

st.title("📓 Diario de decisiones")
st.caption(
    "Un buen resultado de una mala decisión enseña la lección equivocada, y al revés. La "
    "única defensa es haber escrito **entonces** por qué y qué esperabas, y releerlo "
    "después contra lo que pasó. Las decisiones salen solas de tu cuenta; las razones las "
    "escribes tú en `config/settings.local.yaml` (`journal`)."
)

if not app_data.database_ready():
    st.info("Todavía no hay base de datos. Ejecuta `python run_ingest.py`.")
    st.stop()

today = pd.Timestamp(dt.date.today())
table = jr.match_entries(jr.decisions(app_data.trades()), entries,
                         tolerance_days=int(cfg.get("tolerance_days", 3)))
if table.empty:
    st.info("Sin operaciones en el extracto de IBKR ni entradas escritas todavía.")
    st.stop()

view = app_data.regime_view(app_data.db_mtime())
frame = view["frame"] if view else None
tickers = sorted({str(t) for t in table["ticker"]} | {"SPY"})
prices = app_data.prices_for(tickers)
actions = app_data.corporate_actions()


def total_return(ticker: str) -> pd.Series:
    own = prices[prices["series_id"] == f"{ticker}:close_raw"]
    if own.empty:
        return pd.Series(dtype=float)
    return as_series(total_return_index(own.assign(ts=own["ts"].str[:10]), actions, ticker,
                                        today.date().isoformat()))


spy = total_return("SPY")
executed = table[table["side"] != "sin ejecutar"]
unwritten = executed[executed["entry"].isna()]
due = [row for row in table.to_dict("records") if jr.review_due(row["entry"], today)]
cols = st.columns(3)
cols[0].metric("Decisiones registradas", len(executed))
cols[1].metric("Sin razón escrita", len(unwritten),
               help="Una posición que existe antes que la razón escrita para tenerla (§5.1).")
cols[2].metric("Revisiones que tocan", len(due),
               help="Entradas cuya fecha `review_after` ya llegó: relee y juzga la decisión.")

side_label = {"buy": "Compra", "sell": "Venta", "sin ejecutar": "Plan sin ejecutar"}
for row in table.to_dict("records"):
    entry = row["entry"]
    flags = []
    if row["side"] != "sin ejecutar" and entry is None:
        flags.append("⚠️ sin razón escrita")
    if jr.review_due(entry, today):
        flags.append("🔁 toca revisar")
    header = (f"{row['date']} · {side_label.get(row['side'], row['side'])} "
              f"{row['ticker']}" + (f" · {' · '.join(flags)}" if flags else ""))
    with st.expander(header, expanded=bool(flags) and len(table) <= 12):
        if row["side"] != "sin ejecutar":
            stock = total_return(str(row["ticker"]))
            move = jr.since(stock, row["date"], today)
            market = jr.since(spy, row["date"], today)
            verdict = jr.verdict_on(frame, row["date"])
            c = st.columns(4)
            c[0].metric("Importe", money(row["value"], public=False),
                        help=f"{number(row['quantity'], decimals=4)} acciones a "
                             f"{money(row['price'], public=False)} de media, "
                             f"{int(row['executions'])} ejecución(es), comisión "
                             f"{money(abs(row['commission'] or 0), public=False)}.")
            c[1].metric("Semáforo ese día", rg.verdict_label(verdict) if verdict else MISSING,
                        help="El veredicto vigente esa fecha, con lo publicado entonces.")
            c[2].metric("Desde entonces", pct(move), help="Con dividendos, hasta hoy.")
            c[3].metric("Frente a SPY", pct(None if move is None or market is None
                                             else move - market),
                        help="Para una venta, positivo = la acción siguió subiendo después: "
                             "vender costó; negativo = vender evitó esa caída.")
        if entry:
            for key, label in (("why", "Por qué"), ("expectation", "Qué esperaba, y para "
                                                                    "cuándo"),
                               ("premortem", "Si sale mal, habrá sido porque"),
                               ("review_after", "Revisar a partir de"),
                               ("review", "Revisión escrita")):
                if entry.get(key):
                    st.markdown(f"**{label}.** {entry[key]}")
            if jr.review_due(entry, today) and not entry.get("review"):
                st.caption("Toca revisarla: escribe en `review` qué pasó y si la **decisión** "
                           "fue buena con lo que sabías entonces — no si salió bien.")
        else:
            st.markdown("Escríbela en `config/settings.local.yaml` (aunque sea después: "
                        "más vale tarde que sin razón):")
            st.code(
                "journal:\n"
                f"  - date: {row['date']}\n"
                f"    ticker: {row['ticker']}\n"
                "    why: \"\"            # por qué, en una o dos frases\n"
                "    expectation: \"\"    # qué esperas que pase, y en qué plazo\n"
                "    premortem: \"\"      # si sale mal, ¿por qué habrá sido?\n"
                f"    review_after: {(pd.Timestamp(row['date']) + pd.DateOffset(months=6)).date()}",
                language="yaml",
            )

st.caption(
    "Las decisiones son las ejecuciones de un mismo valor, sentido y día. Una entrada se "
    f"empareja con su decisión por valor y fecha (±{int(cfg.get('tolerance_days', 3))} días). "
    "Una entrada sin operación aparece como plan sin ejecutar. El extracto de IBKR cubre "
    "365 días, pero la base guarda toda operación que haya visto: no se pierden."
)
