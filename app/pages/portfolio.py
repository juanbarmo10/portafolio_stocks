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
from app.format import (
    MISSING,
    PORTFOLIO_PUBLIC_NOTE as PUBLIC_NOTE,
    altair_chart,
    color_by_sign,
    color_cells,
    money,
    number,
    pct,
    rebase_100,
    tone,
    verdict_delta,
)
from core.config import load_settings
from transform import corporate_actions as reorg
from transform import bcb
from transform import portfolio
from transform import risk
from transform import themes as themes_tf
from transform.adjustments import total_return_index
from transform.price_action import as_series

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

# Section 5.2 biting where it should. The watchlist buys data, not permission: a candidate
# you are still reading about is not a company with a thesis, and holding one means the
# position exists before the reason to hold it was written down. Said once, plainly, and
# never in public — which positions exist is account information.
if not public:
    held_tickers = set(positions["ticker"]) if not positions.empty else set()
    studying = {str(c.get("ticker")) for c in settings.watchlist_companies}
    written = {str(c.get("ticker")) for c in settings.tracked_companies}
    premature = sorted(held_tickers & studying)
    unwritten = sorted(held_tickers - studying - written)
    if premature:
        st.warning(
            "**En cartera sin tesis terminada: " + ", ".join(premature) + ".** Están en "
            "tu lista de estudio, así que la posición existe antes que la razón escrita "
            "para tenerla. §5.2 pide el criterio de invalidación *antes* de comprar, "
            "porque sin él no hay forma de saber cuándo vender."
        )
    if unwritten:
        st.warning(
            "**En cartera y sin escribir en ninguna parte: " + ", ".join(unwritten)
            + ".** Ni ficha de tesis ni lista de estudio. Añádelos al menos a "
            "`universe.watchlist` para que el panel pueda decirte algo sobre ellos."
        )

# Section 9.2: an unrecognized or contradicted corporate action puts the quantity and the
# cost base of that position in doubt, so it is raised here — next to the other reasons a
# number below might not mean what it says — and never resolved automatically.
unmapped = app_data.unmapped_actions()
unmapped_labels = [
    f"{row['ticker']} el {row['ex_date'] or '¿fecha?'}: código «{row['code']}»"
    + (f" — «{row['description']}»" if row.get("description") else "")
    + f" (visto desde el {row['first_seen']})"
    for row in (unmapped.to_dict("records") if not unmapped.empty else [])
]
for review in reorg.review_positions(app_data.corporate_actions(), positions,
                                     unmapped=unmapped_labels,
                                     since=reorg.held_since(open_lots, positions)):
    st.error(
        f"**{review.ticker} — revisar.** "
        + " ".join(finding.detail for finding in review.reviews)
    )

# Only the securities this panel is about. The breadth universe adds ~700 index members to
# corporate_actions, and their spin-offs-coded-as-splits are noise on this page: a
# permanent list of irrelevant flags trains the reader to skip the relevant ones.
relevant = {
    *settings.market_references,
    *(str(c["ticker"]) for c in settings.researched_companies if c.get("ticker")),
    *(set(positions["ticker"]) if not positions.empty else set()),
}
all_actions = app_data.corporate_actions()
notes = reorg.data_quality_notes(
    all_actions[all_actions["ticker"].isin(relevant)] if not all_actions.empty else all_actions
)
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

cash_series = portfolio.nav_series(account, "NAV:cash")
cash_now = float(cash_series["value"].iloc[-1]) if not cash_series.empty else None
shares_value = float(valued["market_value"].sum(skipna=True)) if not valued.empty else 0.0
account_total = shares_value + cash_now if cash_now is not None else None

st.subheader("Posiciones")

table = valued.merge(costs[["ticker", "avg_cost"]], on="ticker", how="left")
# Rounded before display: the "localized" format shows up to three decimals, and a cost of
# "216,949" reads as two hundred thousand to a Spanish eye.
for column in ("avg_cost", "price", "market_value", "unrealized_pnl"):
    table[column] = table[column].round(2)
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
        "Precio": st.column_config.NumberColumn(format="localized"),
        "Coste medio": st.column_config.NumberColumn(format="localized"),
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
        "Peso en el patrimonio": table["market_value"] / account_total
        if account_total else float("nan"),
        "Fecha precio": table["price_ts"].str[:10],
    })
    column_config = {
        "Cantidad": st.column_config.NumberColumn(format="localized"),
        "Coste medio": st.column_config.NumberColumn(format="localized"),
        "Precio": st.column_config.NumberColumn(format="localized"),
        "Valor": st.column_config.NumberColumn(format="localized"),
        "PnL no realizado": st.column_config.NumberColumn(format="localized"),
        "Retorno": st.column_config.NumberColumn(format="percent"),
        "Peso": st.column_config.NumberColumn(format="percent",
                                              help="Sobre las acciones, sin el efectivo."),
        "Peso en el patrimonio": st.column_config.NumberColumn(
            format="percent", help="Sobre el total: acciones más efectivo."),
    }

st.dataframe(color_by_sign(display, {"PnL no realizado": True, "Retorno": True,
                                     "Retorno no realizado": True}),
             hide_index=True, width="stretch", column_config=column_config)
if not public and cash_now is not None and account_total:
    # The weights above add up to 100 % of the shares. Until 2026-09-25 that was the only
    # weight shown, while most of the account was cash — a concentrated-looking portfolio
    # that was a small fraction of what the account holds.
    st.caption(f"Efectivo: {money(cash_now, public=False)}, el "
               f"{pct(cash_now / account_total)} del patrimonio.")

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
            "Valor": st.column_config.NumberColumn(format="localized"),
        },
    )
    if "sin clasificar" in set(grouped["group"]):
        st.info(
            "Hay posiciones **sin ficha de tesis** (§5.2). Son justo aquellas cuyo modo "
            "de fallo nadie ha escrito: añádelas a `universe.tracked` en "
            "`config/settings.local.yaml`."
        )

# --- 4a. Thematic exposure (§15.5 point 10) -------------------------------------------
# Private: the themes are the user's own grouping, from settings.local.yaml. Each theme's ETF
# against SPY, and each holding's return against SPY split into its theme's part and its own.
THEMES = settings.themes
if not public:
    st.subheader("Exposición temática")
    if not THEMES:
        st.caption("Sin temas escritos. En `universe.themes` de `settings.local.yaml`, un ETF "
                   "de referencia por tema (por ejemplo XBI para biotecnología) con sus tickers: "
                   "el panel separa entonces si una posición ganó por la empresa o por su tema.")
    else:
        theme_tickers = sorted({"SPY", *(t.etf for t in THEMES),
                                *(m for t in THEMES for m in t.tickers)})
        theme_rows = app_data.prices_for(theme_tickers)
        theme_actions = app_data.corporate_actions()
        as_of_theme = pd.Timestamp.today().normalize()

        def theme_tr(ticker: str) -> pd.Series:
            own = theme_rows[theme_rows["series_id"] == f"{ticker}:close_raw"]
            if own.empty:
                return pd.Series(dtype=float)
            return as_series(total_return_index(own.assign(ts=own["ts"].str[:10]),
                                                theme_actions, ticker,
                                                as_of_theme.date().isoformat()))

        theme_series = {t: theme_tr(t) for t in theme_tickers}
        held_weights = {row["ticker"]: float(row["weight"]) for row in valued.to_dict("records")
                        if pd.notna(row.get("weight"))} if not valued.empty else {}
        table_t = themes_tf.theme_table(THEMES, theme_series, as_of_theme, held_weights)
        st.dataframe(
            color_by_sign(pd.DataFrame({
                "Tema": table_t["theme"], "ETF": table_t["etf"],
                "3 m frente a SPY": table_t["vs_spy_3m"],
                "6 m frente a SPY": table_t["vs_spy_6m"],
                "12 m frente a SPY": table_t["vs_spy_12m"],
                "En cartera": table_t["held"], "Peso en acciones": table_t["weight"],
                "Miembros": table_t["members"],
            }), {"3 m frente a SPY": True, "6 m frente a SPY": True, "12 m frente a SPY": True}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="percent") for c in (
                "3 m frente a SPY", "6 m frente a SPY", "12 m frente a SPY",
                "Peso en acciones")},
        )
        rows_split = []
        for ticker in held_weights:
            for theme in themes_tf.themes_of(ticker, THEMES):
                one = themes_tf.split(theme_series.get(ticker, pd.Series(dtype=float)),
                                      theme_series[theme.etf], theme_series["SPY"],
                                      as_of_theme, {"12m": 365})[0]
                rows_split.append({"Posición": ticker, "Tema": f"{theme.name} ({theme.etf})",
                                   "Frente a SPY": one.company_vs_spy,
                                   "Del tema frente a SPY": one.theme_vs_spy,
                                   "De la empresa frente a su tema": one.company_vs_theme})
        if rows_split:
            st.markdown("**12 meses: ¿la empresa o su tema?**")
            split_frame = pd.DataFrame(rows_split)
            st.dataframe(
                color_by_sign(split_frame, {c: True for c in split_frame.columns[2:]}),
                hide_index=True, width="stretch",
                column_config={c: st.column_config.NumberColumn(format="percent")
                               for c in split_frame.columns[2:]})
        missing = [t for t in held_weights if not themes_tf.themes_of(t, THEMES)]
        st.caption(
            "Con dividendos. Mide la **acción** en los últimos 12 meses, no tu posición (que "
            "pudiste comprar después). «Frente a SPY» = (1 + acción) / (1 + SPY) − 1, y se "
            "descompone "
            "**exactamente** en el tema frente a SPY por la empresa frente a su tema. Si casi "
            "todo lo explica el tema, la posición es una apuesta al tema; si lo explica la "
            "empresa, es tu tesis. Un ETF sube o baja con decenas de empresas: no dice nada "
            "de la tuya, solo separa las dos apuestas."
            + (f" Sin tema: {', '.join(sorted(missing))}." if missing else ""))

# --- 4b. Risk, size and shared failure modes (§15.1.8, §5.3) --------------------------
# Private: its tables are not in the public allow-list of columns (test_public_mode), and
# the page is not published anyway (PUBLIC_PAGES).
if not public:

    st.subheader("Riesgo y modos de fallo compartidos")
    RISK = settings.raw.get("panel", {}).get("risk", {})
    LIMITS = settings.raw.get("portfolio", {})
    as_of_risk = pd.Timestamp.today().normalize()
    held = list(valued["ticker"]) if not valued.empty else []
    studied = [str(c["ticker"]) for c in settings.researched_companies if c.get("ticker")]
    factor_etfs = ["SPY", "IWM", "XLK", "XLY", "XLP"]
    metal_etfs = ["GLD", "SLV"]      # §15.5 point 14: would they have diversified the account?
    risk_prices = app_data.prices_for(sorted(set(held) | set(studied) | set(factor_etfs)
                                             | set(metal_etfs)))
    risk_actions = app_data.corporate_actions()


    def tr_index(ticker: str) -> pd.Series:
        own = risk_prices[risk_prices["series_id"] == f"{ticker}:close_raw"]
        if own.empty:
            return pd.Series(dtype=float)
        return as_series(total_return_index(own.assign(ts=own["ts"].str[:10]), risk_actions,
                                            ticker, as_of_risk.date().isoformat()))


    indices = {t: tr_index(t) for t in sorted(set(held) | set(studied) | set(factor_etfs)
                                              | set(metal_etfs))}

    if held and account_total:
        weights = {row["ticker"]: float(row["market_value"]) / account_total
                   for row in valued.to_dict("records") if pd.notna(row["market_value"])}
        weights["CASH"] = (cash_now or 0.0) / account_total
        breakdown = risk.risk_breakdown(
            weights, risk.daily_returns({t: indices[t] for t in held}, as_of_risk,
                                        int(RISK.get("window_days", 365))))
        st.dataframe(pd.DataFrame({
            "Posición": breakdown.table["ticker"].replace({"CASH": "Efectivo"}),
            "Peso (con efectivo)": breakdown.table["weight"],
            "Volatilidad anual": breakdown.table["volatility"],
            "Parte del riesgo": breakdown.table["share_of_risk"],
        }), hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent")
            for c in ("Peso (con efectivo)", "Volatilidad anual", "Parte del riesgo")})
        corr = breakdown.correlation
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns)
                 for b in corr.columns[i + 1:]] if not corr.empty else []
        st.caption(
            f"Volatilidad de la cuenta: **{pct(breakdown.portfolio_volatility)} anual** "
            f"(un año de rendimientos diarios, efectivo incluido). «Parte del riesgo» reparte "
            "esa volatilidad entre las posiciones: pesa más quien más se mueve y más se mueve "
            "con las demás."
            + (" Correlaciones: " + "; ".join(f"{a}-{b} {number(c)}" for a, b, c in pairs) + "."
               if pairs else "")
        )
        # Gold and silver against the account as it is today (§15.5 point 14): the weekly
        # return of today's holdings, at today's weights, over the risk window. A context for
        # the policy decision, not a recommendation to hold either.
        invested = {t: w for t, w in weights.items() if t != "CASH" and t in indices}
        metal_corr = risk.correlation_with_basket(
            invested, {t: indices[t] for t in [*invested, *metal_etfs]}, metal_etfs,
            as_of_risk, int(RISK.get("window_days", 365)))
        if metal_corr:
            st.caption(
                "Correlación semanal de un año con tus acciones de hoy (a los pesos de hoy): "
                + "; ".join(f"{t} {number(c)}" for t, c in metal_corr.items())
                + ". Cerca de 0 o negativa = se mueve por su cuenta y habría diversificado; "
                "cerca de 1 = es la misma apuesta. Cómo se portaron el oro y la plata en las "
                "caídas del mercado: 📈 Mercado, sección 6.")
        categories = {t: (metadata.get(t) or {}).get("thesis_category") for t in weights}
        breaches = risk.limit_breaches(weights, categories, LIMITS.get("max_position"),
                                       LIMITS.get("max_per_thesis_category"))
        if breaches:
            st.warning("Por encima de tus límites escritos: " + "; ".join(breaches) + ".")
        elif LIMITS.get("max_position") is None and LIMITS.get("max_per_thesis_category") is None:
            st.caption("Sin límites de tamaño escritos (`portfolio.max_position`, "
                       "`portfolio.max_per_thesis_category` en `settings.local.yaml`): el panel "
                       "no inventa uno.")

    fred_now = app_data.fred_observations()
    factors = risk.factor_returns(
        {t: indices[t] for t in factor_etfs},
        bcb.series(fred_now, "DGS10", as_of_risk), bcb.series(fred_now, "DTWEXBGS", as_of_risk),
        bcb.series(fred_now, "DCOILWTICO", as_of_risk))
    factors = factors[factors.index >= as_of_risk - pd.DateOffset(years=int(RISK.get("factor_years", 3)))]
    names = [t for t in dict.fromkeys([*held, *studied]) if not indices.get(t, pd.Series()).empty]
    sens = [x for x in (risk.sensitivity(indices[t], factors, t) for t in names) if x]
    if sens:
        modes = risk.failure_modes(sens, t_threshold=float(RISK.get("t_threshold", 2.0)))
        grid = modes.pivot_table(index="scenario", columns="ticker", values="move",
                                 aggfunc="first", sort=False)
        reliable = modes.pivot_table(index="scenario", columns="ticker", values="reliable",
                                     aggfunc="first", sort=False)
        shown = pd.DataFrame({
            (t + (" · cartera" if t in held else "")): [
                (pct(grid.at[sc, t]) + ("" if bool(reliable.at[sc, t]) else " (ruido)"))
                if pd.notna(grid.at[sc, t]) else MISSING for sc in grid.index]
            for t in grid.columns}, index=grid.index)
        st.markdown("**Qué pasaría en cada escenario** — movimiento estimado de cada acción")
        # Coloured only where the estimate is not noise: a red "(ruido)" would still alarm.
        styled = shown.style
        for t in grid.columns:
            label = t + (" · cartera" if t in held else "")
            styled = color_cells(styled, label, [
                tone(grid.at[sc, t]) if pd.notna(grid.at[sc, t]) and bool(reliable.at[sc, t])
                else None for sc in grid.index])
        st.dataframe(styled, width="stretch")
        together = [(sc, [t for t in grid.columns if bool(reliable.at[sc, t])
                          and grid.at[sc, t] < 0]) for sc in grid.index]
        shared = [(sc, ts) for sc, ts in together if len(ts) >= 2 and sc != "mercado −20 %"]
        st.caption(
            ("**Caen juntas:** " + "; ".join(f"si {sc}: {', '.join(ts)}" for sc, ts in shared)
             + ". " if shared else "")
            + f"Regresión de los rendimientos semanales de {int(RISK.get('factor_years', 3))} "
            "años sobre el mercado, el tipo a 10 años, el dólar, pequeñas − grandes, tecnología − "
            "mercado, consumo cíclico − defensivo y el petróleo (WTI). Es cómo se movieron "
            "**juntas en el pasado**, no una predicción. «(ruido)» = no distinguible de cero "
            f"(|t| < {number(float(RISK.get('t_threshold', 2.0)), decimals=0)}). Con siete "
            "factores y "
            "varias empresas, alguna saldrá fiable por azar: úsalo para hacer preguntas a tu "
            "tesis, no para responderlas."
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

    altair_chart(
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
        "Sin aportes, y contra el mercado, justo debajo."
    )

# --- 5b. Against the market ----------------------------------------------------------

st.subheader("¿Batiendo al mercado?")
spy_rows = app_data.prices_for(["SPY"])
spy = total_return_index(spy_rows.assign(ts=spy_rows["ts"].str[:10]), app_data.corporate_actions(),
                         "SPY", "2099-01-01") if not spy_rows.empty else pd.DataFrame()
benchmark = (pd.Series(spy["value"].to_numpy(), index=pd.to_datetime(spy["ts"]))
             if not spy.empty else pd.Series(dtype=float))
perf, curves = portfolio.performance(nav, portfolio.external_flows(cash), benchmark)
if perf is None:
    st.info("Sin NAV o sin precios de SPY suficientes para comparar.")
else:
    cols = st.columns(3)
    cols[0].metric("Tu rentabilidad (sin aportes)", pct(perf.twr),
                   **verdict_delta(tone(perf.twr), "gana" if (perf.twr or 0) > 0 else "pierde"),
                   help="Ponderada en el tiempo: encadena los días quitando cada aporte, así "
                        "que mide las decisiones y no el dinero que entró.")
    cols[1].metric("S&P 500 con dividendos (SPY)", pct(perf.benchmark_return))
    cols[2].metric("Diferencia", pct(perf.excess),
                   **verdict_delta(tone(perf.excess), "por encima de SPY"
                                   if (perf.excess or 0) > 0 else "por debajo de SPY"),
                   help="Tu rentabilidad menos la del índice, en esta ventana. Positiva = "
                        "lo batiste; con menos de varios años es sobre todo ruido.")
    long = curves.melt("date", var_name="Serie", value_name="valor")
    long["Serie"] = long["Serie"].map({"account": "Tu cuenta", "benchmark": "SPY"})
    altair_chart(
        alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("valor:Q", title="Base 100, sin aportes", scale=alt.Scale(zero=False)),
            color=alt.Color("Serie:N", legend=alt.Legend(orient="top", title=None),
                            scale=alt.Scale(domain=["Tu cuenta", "SPY"],
                                            range=[LINE_COLOR, "#eb6834"])),
        ).properties(height=240),
        width="stretch",
    )
    if not public:
        st.caption(
            f"Del {perf.start} al {perf.end}. **Si el saldo inicial y cada aporte "
            f"({money(perf.contributions, public=False)} en total) hubieran ido a SPY el "
            f"mismo día, hoy tendrías {money(perf.shadow_end, public=False)}**; tienes "
            f"{money(perf.nav_end, public=False)}. Es la comparación honesta con "
            "«comprar el índice». Una ventana de un año dice poco sobre la habilidad."
        )

    # Money-weighted return and where the gap comes from (§15.1.5).
    mwr = portfolio.money_weighted_return(nav, portfolio.external_flows(cash))
    cash_cost = portfolio.cash_drag(nav, portfolio.nav_series(account, portfolio.NAV_CASH),
                                    benchmark)
    cols = st.columns(3)
    cols[0].metric("TIR (ponderada por el dinero)",
                   pct(mwr.period_return) if mwr else MISSING,
                   **verdict_delta(tone(mwr.period_return) if mwr else None,
                                   "gana" if mwr and mwr.period_return > 0 else "pierde"),
                   help="La tasa interna de retorno de tus propios flujos: cuenta cuándo "
                        "entró cada aporte. Sobre la ventana, sin anualizar"
                        + (f"; anualizada, {pct(mwr.annual_rate)}." if mwr else "."))
    # Positive = what the cash failed to earn next to SPY (a cost); negative = it avoided a loss.
    cols[1].metric("Coste del efectivo frente a SPY", pct(cash_cost),
                   **verdict_delta(tone(cash_cost, higher_is_better=False),
                                   "le costó" if (cash_cost or 0) > 0 else "le ahorró"),
                   help="Aproximado: cada día, la parte de la cuenta en efectivo por lo que "
                        "hizo SPY ese día, sumado. Lo que el efectivo dejó de ganar (o evitó "
                        "perder) frente a tenerlo en el índice.")
    rest = None if cash_cost is None else perf.excess + cash_cost
    cols[2].metric("Resto de la diferencia", pct(rest),
                   **verdict_delta(tone(rest), "suma" if (rest or 0) > 0 else "resta"),
                   help="La diferencia con SPY sin la parte del efectivo: lo que explican las "
                        "acciones elegidas y cuándo se compraron y vendieron. Aproximado.")
    st.caption(
        "La rentabilidad sin aportes mide las decisiones; la TIR, tu experiencia — un "
        "aporte que llegó justo antes de una caída pesa más. La diferencia con el índice se "
        "parte en dos: el efectivo sin invertir y la selección."
    )

    if not public:
        attribution = portfolio.position_attribution(trades, cash, valued)
        if not attribution.empty:
            st.markdown("**Qué aportó cada posición**")
            # Rounded to cents like the positions table: "764,455" reads as thousands.
            cents = attribution.round({c: 2 for c in ("bought", "sold", "market_value",
                                                      "income", "commissions", "pnl")})
            st.dataframe(color_by_sign(pd.DataFrame({
                "Posición": cents["ticker"],
                "Comprado": cents["bought"],
                "Vendido": cents["sold"],
                "Valor hoy": cents["market_value"],
                "Dividendos netos": cents["income"],
                "Comisiones": cents["commissions"],
                "Resultado": cents["pnl"],
                "Parte del total": attribution["share"],
                "Historia completa": attribution["complete"],
            }), {"Resultado": True, "Parte del total": True}),
                hide_index=True, width="stretch", column_config={
                **{c: st.column_config.NumberColumn(format="localized")
                   for c in ("Comprado", "Vendido", "Valor hoy", "Dividendos netos",
                             "Comisiones", "Resultado")},
                "Parte del total": st.column_config.NumberColumn(format="percent"),
                "Historia completa": st.column_config.CheckboxColumn(
                    help="Sin marcar: vendió más de lo que el extracto muestra comprado — "
                         "la compra es anterior a la ventana y su resultado saldría inflado."),
            })
            st.caption("Resultado = valor hoy + ventas − compras + comisiones + dividendos "
                       "netos de retención, en USD, desde que empieza el extracto. Incluye lo "
                       "ya vendido: una posición cerrada también sumó o restó.")

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
    realized_total = float(realized["realized_pnl"].sum()) if not realized.empty else 0.0
    columns[0].metric("PnL realizado (FIFO)", money(realized_total, public=False),
                      **verdict_delta(tone(realized_total),
                                      "ganancia" if realized_total > 0 else "pérdida"))
    columns[1].metric("Dividendos netos", money(income.dividends_net, public=False),
                      help=f"Brutos {money(income.dividends_gross, public=False)}, "
                           f"retención {money(income.withholding, public=False)}.")
    columns[2].metric("Comisiones", money(income.commissions, public=False))
    columns[3].metric("Comisiones + gastos / NAV", pct(drag, decimals=2) if drag else "—")

    if not realized.empty:
        st.dataframe(
            color_by_sign(pd.DataFrame({
                "Posición": realized["ticker"],
                "PnL realizado": realized["realized_pnl"].round(2),
                "Cantidad vendida": realized["quantity_sold"],
                "Ventas": realized["disposals"],
            }), {"PnL realizado": True}),
            hide_index=True, width="stretch",
            column_config={
                "PnL realizado": st.column_config.NumberColumn(format="localized"),
                "Cantidad vendida": st.column_config.NumberColumn(format="localized"),
            },
        )

    st.caption(
        f"Retención **observada** sobre dividendos: "
        f"{pct(income.effective_withholding_rate)}. Es el dato que reporta IBKR, no una "
        "tasa supuesta; qué significa fiscalmente es pregunta para el contador (§11). "
        "El PnL realizado sale del motor FIFO propio, no del resumen del bróker — "
        "coinciden, y esa coincidencia es la prueba."
    )
