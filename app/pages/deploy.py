"""📋 Desplegar capital — level 4: is it time to execute the plan? (CLAUDE.md section 2).

The last step of the checklist, and the only page that talks about putting money in — or
taking it out. In order, and the order is the point: whether today is a day to execute at
all (levels 1-3 as gates), how large the tranche is, whether a held position deserves it
before a new one (§15.5 point 1), what and at what price, what it costs to bring the pesos
to IBKR, and — last — what the written plan says before selling.

Private: it shows the account's cash and deposits. Not in ``PUBLIC_PAGES``.

Nothing here places an order or moves money (section 12): the page prepares a decision
that the user executes in IBKR by hand.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data as app_data
from app.format import (BAD, CAUTION, GOOD, MISSING, colored, money, number, pct,
                        reported_amount)
from core.config import load_settings
from ingest.macro_calendar import current_calendar
from transform import allocation as al
from transform import behavior as bh
from transform import discipline as dc
from transform import funding as fx
from transform import per_share
from transform import portfolio as port
from transform import regime as rg
from transform import scenarios as scn
from transform import thesis as th
from transform import valuation as val
from transform.adjustments import total_return_index
from transform.price_action import as_series

settings = load_settings()
local = settings.raw
level3 = local.get("panel", {}).get("level3", {})
earnings_window = int(level3.get("earnings_window_days", 5))
funding_cfg = dict(local.get("funding") or {})
contribution = dict(local.get("contributions") or {})

st.title("📋 Desplegar capital")
st.caption("Nivel 4 del checklist. La página prepara la decisión; la orden la pones tú en "
           "IBKR. El panel es de solo lectura y nunca opera (§12).")

if not app_data.database_ready():
    st.info("Todavía no hay base de datos. Ejecuta `python run_ingest.py`.")
    st.stop()

today = pd.Timestamp.today().normalize()

# --- 1. Gates ---------------------------------------------------------------------------
st.subheader("1 · ¿Es día de ejecutar?")
gates: list[tuple[str, str, str]] = []

view = app_data.regime_view(app_data.db_mtime())
if view is None:
    gates.append(("⚪", "Semáforo (niveles 1 y 2)", "sin datos todavía"))
else:
    r = view["reading"]
    icon = {rg.RISK_OFF: "🔴", rg.RISK_ON: "🟢"}.get(r.verdict, "🟡")
    gates.append((icon, "Semáforo (niveles 1 y 2)",
                  f"{rg.verdict_label(r.verdict)} al {r.date}"))

events = app_data.events()
calendar = current_calendar(events) if not events.empty else pd.DataFrame()
soon = []
for row in calendar.to_dict("records"):
    when = pd.Timestamp(row["ts"])
    if today.tz_localize("UTC") <= when <= (today + pd.Timedelta(hours=48)).tz_localize("UTC"):
        soon.append(f"{row['label']} ({when.tz_convert('America/New_York'):%Y-%m-%d %H:%M} NY)")
gates.append(("🟡" if soon else "🟢", "Dato macro de alto impacto en 48 h",
              "; ".join(soon) + " — §2 propone posponer el aporte" if soon else "ninguno"))

companies = app_data.companies()
by_ticker = dict(zip(companies["ticker"], companies["cik"])) if not companies.empty else {}
positions = port.latest_positions(app_data.account_observations())
watched = {str(c["cik"]).zfill(10): str(c["ticker"]) for c in settings.researched_companies
           if c.get("cik")}
for ticker in positions["ticker"] if not positions.empty else []:
    if by_ticker.get(ticker):
        watched.setdefault(by_ticker[ticker], ticker)
near = []
if not events.empty:
    earnings = events[(events["category"] == "earnings") & events["cik"].isin(watched)]
    for row in earnings.to_dict("records"):
        day = pd.Timestamp(str(row["ts"])[:10])
        if today <= day <= today + pd.Timedelta(days=earnings_window):
            kind = "estimada" if int(row.get("is_estimated") or 0) else "confirmada"
            near.append(f"{watched[row['cik']]} el {day:%Y-%m-%d} ({kind})")
gates.append(("🟡" if near else "🟢", f"Resultados en {earnings_window} días",
              "; ".join(near) + " — no abrir posición en ellas sin decisión explícita"
              if near else "ninguno en cartera ni en estudio"))

ICON_TONES = {"🟢": GOOD, "🟡": CAUTION, "🔴": BAD, "✅": GOOD, "⚠️": CAUTION}
st.markdown("| | Puerta | Estado |\n|---|---|---|\n"
            + "\n".join(f"| {i} | **{g}** | {colored(s, ICON_TONES.get(i))} |"
                         for i, g, s in gates))
if view is not None and view["reading"].verdict == rg.RISK_OFF:
    st.error("Regla dura de §2: con el semáforo en rojo no se compra, aunque la empresa sea "
             "perfecta. El dinero espera en efectivo; en la validación, esperar costó poco.")

# --- 2. The tranche ---------------------------------------------------------------------
st.subheader("2 · El tramo")
nav = app_data.account_observations()
cash = port.nav_series(nav, port.NAV_CASH) if not nav.empty else pd.DataFrame()
cash_now = float(cash["value"].iloc[-1]) if not cash.empty else None
fee = fx.order_fee(app_data.trades())
default_amount = float(contribution.get("monthly_amount_usd") or 100.0)

c1, c2, c3 = st.columns(3)
amount = c1.number_input("Tramo (USD)", min_value=1.0, value=default_amount, step=10.0,
                         help="Definido ANTES de abrir IBKR (§2, nivel 4).")
orders = c2.number_input("Órdenes", min_value=1, max_value=5, value=1, step=1)
c3.metric("Efectivo en IBKR, último extracto", money(cash_now, public=False))
share = None if fee is None else orders * fee / amount
st.markdown(
    f"Comisión mediana por ejecución en tu cuenta: **{money(fee, public=False)}**. "
    f"{orders} orden(es) sobre {money(amount, public=False)} = **{pct(share, decimals=2)}** "
    "del tramo. La comisión mínima es por orden, así que con tramos pequeños **una orden "
    "por aporte** — a la posición más por debajo de su peso objetivo — y rotar entre "
    "aportes, en vez de repartir cada tramo entre varias.")
if cash_now and cash_now >= amount:
    st.info(f"Ya hay {money(cash_now, public=False)} en efectivo en IBKR. Desplegar eso no "
            "cuesta ninguna transferencia: va antes que cualquier depósito nuevo.")

# Core and satellite (§15.5 point 4): where this tranche goes by the written policy. The
# rebalancing is done with contributions, never with sales (transform/allocation.py).
policy = al.Policy.from_config(local.get("portfolio"))
held_all = sorted(positions["ticker"]) if not positions.empty else []
valued_all = port.valuation(positions, port.latest_prices(app_data.prices_for(held_all))) \
    if held_all else pd.DataFrame()
nav_frame = port.nav_series(nav) if not nav.empty else pd.DataFrame()
nav_now = float(nav_frame["value"].iloc[-1]) if not nav_frame.empty else None
split = al.allocate(valued_all, nav_now, cash_now, policy)
st.markdown("**Núcleo y satélite**")
if policy.written:
    target = float(policy.core_target)
    st.markdown(
        f"Núcleo ({', '.join(policy.core_tickers)}): **{pct(split.core_share, decimals=0)}** "
        f"de la cuenta frente a tu objetivo del **{pct(target, decimals=0)}** · satélite "
        f"({len(split.satellites)} acciones): {pct(split.satellite_share, decimals=0)} · "
        f"efectivo: {pct(split.cash_share, decimals=0)}.")
    if split.destination == "core":
        months = None if not split.to_target else split.to_target / default_amount
        st.info(f"**Este tramo va al núcleo.** Faltan unos "
                f"{money(split.to_target, public=False)} de aportes para llegar al objetivo"
                + (f" (≈ {number(months, decimals=0)} aportes de "
                   f"{money(default_amount, public=False)})" if months else "")
                + ". No se vende ningún satélite para rebalancear: se corrige con aportes.")
    elif split.destination == "satellite":
        st.info("**El núcleo está en su objetivo: este tramo puede ir al satélite.** Primero, "
                "reforzar una tesis que ya tienes y mejora (sección 3); una nueva, solo si "
                "ninguna cumple.")
else:
    st.caption("Sin política núcleo-satélite escrita (`portfolio.core` en "
               "`settings.local.yaml`): el panel no sabe qué parte de la cuenta es núcleo.")
for note in split.notes if policy.written else split.notes[1:]:
    st.caption(note)
days_above, since = al.cash_spell(nav_frame, port.nav_series(nav, port.NAV_CASH)
                                  if not nav.empty else pd.DataFrame(), policy.cash_max)
if days_above:
    grace = policy.cash_grace_days
    message = (f"El efectivo está por encima de tu límite del {pct(policy.cash_max, decimals=0)} "
               f"desde el {since} ({days_above} días).")
    if grace is not None and days_above > grace:
        st.warning(message + f" Tu regla permite {grace} días: toca desplegarlo.")
    else:
        st.caption(message)

targets = dict(local.get("portfolio", {}).get("target_weights") or {})
suggested = None
if targets and not positions.empty:
    valued = port.valuation(positions, port.latest_prices(app_data.prices_for(
        sorted(set(positions["ticker"]) | set(targets)))))
    drift = port.target_drift(valued, targets)
    if not drift.empty:
        under = drift.dropna(subset=["drift"]).iloc[0]
        suggested = str(under["ticker"])
        st.markdown(f"Más por debajo de su objetivo: **{under['ticker']}** — pesa "
                    f"{pct(under['weight'])} de las acciones frente a un objetivo de "
                    f"{pct(under['target_weight'])}.")
else:
    st.caption("Sin cartera objetivo escrita (`portfolio.target_weights`, entrada de usuario "
               "nº4), el panel no puede decir a qué posición va el tramo: un drift contra un "
               "objetivo inexistente parecería equilibrio.")

# --- 3. Reinforce before opening (§15.5, point 1) ---------------------------------------
st.subheader("3 · ¿Reforzar antes de abrir?")
DISCIPLINE = dict(local.get("panel", {}).get("discipline") or {})
BEHAVIOR = dict(local.get("panel", {}).get("behavior") or {})
held = sorted(positions["ticker"]) if not positions.empty else []
tracked_cards = {str(c["ticker"]): c for c in settings.tracked_companies if c.get("ticker")}
valued_held = port.valuation(positions, port.latest_prices(app_data.prices_for(held))) \
    if held else pd.DataFrame()
nav_total_frame = port.nav_series(nav) if not nav.empty else pd.DataFrame()
nav_total = float(nav_total_frame["value"].iloc[-1]) if not nav_total_frame.empty else None
portfolio_cfg = dict(local.get("portfolio") or {})
max_position = portfolio_cfg.get("max_position")
sec_obs = app_data.sec_observations()

# Metrics a rule may name that are amounts; the rest are ratios (thesis.evaluable_metrics).
AMOUNT_METRICS = {"revenue_ttm", "net_income_ttm", "fcf_ttm", "net_buybacks_ttm",
                  "long_term_debt", "equity", "cash"}


def metric_value(metric, value) -> str:
    if value is None or pd.isna(value):
        return MISSING
    return reported_amount(value) if metric in AMOUNT_METRICS else pct(value)


REINFORCE_LABELS = {
    "improving": ("Tesis intacta y su métrica mejora", GOOD),
    "flat": ("Tesis intacta, métrica sin cambio", None),
    "unknown": ("No evaluable", None),
    "worsening": ("Intacta, pero la métrica va hacia el umbral", CAUTION),
    "at_limit": ("En tu tope de tamaño", CAUTION),
    "no_card": ("Sin ficha de tesis", CAUTION),
    "breached": ("Invalidación CRUZADA", BAD),
}
if not held:
    st.caption("Sin posiciones abiertas en el último extracto: el tramo va a una posición "
               "nueva (sección 4).")
else:
    table3 = dc.reinforce(held, tracked_cards, by_ticker, sec_obs, valued_held, today,
                          lookback_days=int(DISCIPLINE.get("lookback_days", 100)),
                          hurdle_rate=level3.get("hurdle_rate"), nav_total=nav_total,
                          max_position=max_position)
    st.markdown(
        "| Posición | Estado | Métrica de tu regla | Hace ~un trimestre | Hoy | Margen hasta "
        "el umbral | Peso en la cuenta | Nota |\n|---|---|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| **{r.ticker}** | {colored(*REINFORCE_LABELS[r.status])} | "
            f"{f'`{r.metric}` {r.operator} {r.threshold}' if r.metric else MISSING} | "
            f"{metric_value(r.metric, r.value_before)} | {metric_value(r.metric, r.value_now)} | "
            f"{pct(r.headroom) if r.headroom is not None and pd.notna(r.headroom) else MISSING} | "
            f"{pct(r.weight) if r.weight is not None and pd.notna(r.weight) else MISSING} | "
            f"{r.note or ''} |"
            for r in table3.itertuples()))
    st.caption(
        "Antes de abrir una posición nueva, reforzar una que ya tienes **si su tesis sigue "
        "viva y su propia métrica mejora**. La métrica es la de tu `invalidation_rule`: el "
        "panel no elige por ti qué cifra importa. «Hace ~un trimestre» y «Hoy» son cifras "
        "presentadas a cada fecha (sin mirar el futuro, §9.4); «Margen» es la distancia "
        "relativa al umbral, positiva en el lado seguro. **El orden es el de la tesis, nunca "
        "el del precio**: que haya subido no es razón para añadir, ni que haya bajado para "
        "promediar. Si ninguna mejora, abrir una nueva es coherente con el plan."
        + ("" if max_position is not None else
           " Sin tope escrito (`portfolio.max_position`) no se comprueba el tamaño."))

# --- 4. What, and at what price ---------------------------------------------------------
st.subheader("4 · ¿Qué, y a qué precio?")
cards = {str(c["ticker"]): c for c in settings.researched_companies if c.get("ticker")}
candidates = sorted(set(cards) | (set(positions["ticker"]) if not positions.empty else set())
                    | set(targets))
if candidates:
    # Every candidate on one scale (§15.4, point 7): the return the price gives today if free
    # cash flow per share grows at a given rate — the same question for all of them.
    VALN = settings.raw.get("panel", {}).get("valuation", {})
    RDCF_ = VALN.get("reverse_dcf", {})
    hurdle_ = settings.raw.get("panel", {}).get("level3", {}).get("hurdle_rate")
    today_iso_ = today.date().isoformat()
    observations_ = app_data.sec_observations()
    scale = []
    for t in candidates:
        cik_t = str((cards.get(t) or {}).get("cik") or by_ticker.get(t) or "").zfill(10)
        v = app_data.valuation_now(cik_t, t, today_iso_, app_data.db_mtime())
        g = per_share.assess(observations_, cik_t, today_iso_, horizons=(3,))
        past = g.table.set_index(["metric", "years"])["per_share"]
        mine = (cards.get(t) or {}).get("growth_assumption")
        growths = [0.0, 0.10] + ([float(mine)] if mine is not None else [])
        r = val.return_grid(v, growths, terminal_growth=float(RDCF_.get("terminal_growth", 0.025)),
                            years=int(RDCF_.get("years", 10)))

        def cell(x, v=v):
            if x.rate is not None:
                return pct(x.rate)
            # No market cap for a company with no SEC figures at all (NU files IFRS).
            return "sin cifras de la SEC" if v.shares is None and v.fcf_ttm is None else x.note

        scale.append({
            "Empresa": t + (" · cartera" if not positions.empty
                            and t in set(positions["ticker"]) else ""),
            "Rend. FCF": pct(v.fcf_yield),
            "Acciones/año": pct(g.shares_yoy) if g.basis == "basic_shares" else MISSING,
            "Ingresos/acción, 3 años": pct(past.get(("revenue", 3))),
            "Si el FCF/acción no crece": cell(r[0]),
            "Si crece un 10 %": cell(r[1]),
            "Con tu supuesto": (f"{cell(r[2])} ({pct(float(mine), decimals=0)})"
                                if mine is not None else "sin escribir"),
        })
    st.markdown("**Todas las candidatas en una misma escala** — rentabilidad anual que da el "
                "precio de hoy")
    st.dataframe(pd.DataFrame(scale), hide_index=True, width="stretch")
    st.caption(
        "Misma pregunta para todas: si el flujo de caja libre **por acción** crece a ese ritmo "
        f"{int(RDCF_.get('years', 10))} años (y luego {pct(float(RDCF_.get('terminal_growth', 0.025)))}), "
        "¿qué rentabilidad anual te da comprar hoy? La dilución va dentro del crecimiento: "
        "réstale la columna de acciones. «Ingresos/acción» es el pasado, no una previsión. "
        + (f"Tu tasa exigida: {pct(hurdle_)}." if hurdle_ is not None else
           "Sin tasa exigida escrita (`panel.level3.hurdle_rate`), la tabla ordena pero no "
           "decide.")
        + " Una empresa que quema caja no tiene flujo que crecer: su precio descansa en un "
          "flujo que aún no existe."
    )

if not candidates:
    st.caption("Sin empresas en estudio ni posiciones: nada que valorar todavía.")
else:
    pick = st.selectbox("Qué piensas comprar", candidates,
                        index=candidates.index(suggested) if suggested in candidates else 0,
                        help="Por defecto, la más por debajo de su peso objetivo, si lo hay.")
    cik = str((cards.get(pick) or {}).get("cik") or by_ticker.get(pick) or "").zfill(10)
    card = cards.get(pick) or {}
    tracked = pick in {str(c["ticker"]) for c in settings.tracked_companies}
    ladder = card.get("exit_ladder") or []
    checks = [
        ("✅" if tracked else "⚠️", "Ficha de tesis",
         "escrita" if tracked else "sin ficha: la posición existiría antes que la razón escrita"),
        ("✅" if ladder else "⚠️", "Regla de salida (`exit_ladder`)",
         f"{len(ladder)} escrita(s)" if ladder else "sin escribir — §2 la pide ANTES de comprar"),
    ]
    near_pick = [n for n in near if n.startswith(pick + " ")]
    checks.append(("⚠️" if near_pick else "✅", f"Resultados en {earnings_window} días",
                   near_pick[0] if near_pick else "no"))
    # Scenarios and size (§15.5 point 3): the weight this tranche would leave, against the
    # ceiling that follows from what the card says could go wrong.
    SCEN = dict(local.get("panel", {}).get("scenarios") or {})
    RDCF_S = dict(local.get("panel", {}).get("valuation", {}).get("reverse_dcf") or {})
    val_pick = app_data.valuation_now(cik, pick, today.date().isoformat(), app_data.db_mtime())
    outlook = scn.assess(card.get("scenarios"),
                         val_pick if val_pick.market_cap is not None else None,
                         years=int(SCEN.get("horizon_years", 5)),
                         terminal_growth=float(RDCF_S.get("terminal_growth", 0.025)),
                         dcf_years=int(RDCF_S.get("years", 10)),
                         kelly_share=float(SCEN.get("kelly_fraction", 0.25)),
                         loss_budget=portfolio_cfg.get("loss_budget"),
                         max_position=max_position)
    if outlook is None:
        checks.append(("⚠️", "Escenarios (`scenarios`)",
                       "sin escribir — sin ellos no hay techo de tamaño (ver 🏢 Empresa)"))
    else:
        checks.append(("✅" if outlook.expected_return is not None else "⚠️",
                       "Escenarios (`scenarios`)",
                       f"rentabilidad esperada {pct(outlook.expected_return)} al año; "
                       f"probabilidad de perder {pct(outlook.loss_probability, decimals=0)}; "
                       f"peor caída {pct(-outlook.worst_loss) if outlook.worst_loss else MISSING}"))
        held_value = 0.0
        if not valued_held.empty and pick in set(valued_held["ticker"]):
            v = valued_held.loc[valued_held["ticker"] == pick, "market_value"].iloc[0]
            held_value = float(v) if pd.notna(v) else 0.0
        after = None if not nav_total else (held_value + amount) / nav_total
        over = after is not None and outlook.ceiling is not None and after > outlook.ceiling
        checks.append(("⚠️" if over or outlook.ceiling is None else "✅", "Tamaño tras el tramo",
                       (f"pesaría {pct(after, decimals=1)} de la cuenta" if after is not None
                        else MISSING)
                       + (f" frente a un techo de {pct(outlook.ceiling, decimals=1)}"
                          if outlook.ceiling is not None else
                          " — sin techo: escribe `portfolio.loss_budget`")))
    st.markdown("| | Comprobación | Estado |\n|---|---|---|\n"
                + "\n".join(f"| {i} | {c} | {colored(v, ICON_TONES.get(i))} |"
                             for i, c, v in checks))

    years = int(settings.raw.get("panel", {}).get("valuation", {}).get("history_years", 5))
    today_iso = today.date().isoformat()
    now, hist = app_data.valuation_view(cik, pick, today_iso, years, app_data.db_mtime())
    table = val.band(now, hist) if now.market_cap is not None else pd.DataFrame()
    if table.empty:
        st.caption(f"Sin banda de valoración para {pick}: hacen falta precio y fundamentales "
                   "de la SEC con historia (una empresa que presenta en IFRS, como NU, no los "
                   "tiene). Mira su página de Empresa.")
    else:
        labels = {"ev_sales": "EV / ventas", "ev_ebit": "EV / EBIT", "pe": "P / E",
                  "p_fcf": "P / FCF", "fcf_yield": "Rend. FCF"}

        def show(name: str, value: float) -> str:
            return pct(value) if name == "fcf_yield" else f"{number(value, decimals=1)}×"

        st.dataframe(pd.DataFrame({
            "Múltiplo": table["multiple"].map(labels),
            "Hoy": [show(m, v) for m, v in zip(table["multiple"], table["today"])],
            f"Mínimo {years} años": [show(m, v) for m, v in zip(table["multiple"], table["low"])],
            "Mediana": [show(m, v) for m, v in zip(table["multiple"], table["median"])],
            "Máximo": [show(m, v) for m, v in zip(table["multiple"], table["high"])],
            "Percentil de hoy": [f"{p * 100:.0f}" if p is not None else MISSING
                                 for p in table["percentile"]],
        }), hide_index=True, width="stretch")
        rdcf = settings.raw.get("panel", {}).get("valuation", {}).get("reverse_dcf", {})
        hurdle = settings.raw.get("panel", {}).get("level3", {}).get("hurdle_rate")
        rate = float(hurdle) if hurdle is not None else 0.10
        implied = val.reverse_dcf(now, [rate],
                                  terminal_growth=float(rdcf.get("terminal_growth", 0.025)),
                                  years=int(rdcf.get("years", 10)))["fcf"][0]
        st.caption(
            f"Percentil 100 = el más caro que ha estado en {years} años (en el rendimiento FCF "
            "al revés: alto = barato). Cada punto, con lo presentado ese día. "
            + (f"Al {pct(rate, decimals=0)} "
               + ("(tu tasa)" if hurdle is not None else "(tasa de referencia; la tuya en "
                  "`panel.level3.hurdle_rate`)")
               + ", el precio descuenta que el FCF crezca "
               + (f"**{pct(implied.growth)} al año** durante {int(rdcf.get('years', 10))} "
                  "años." if implied.growth is not None else f"— {implied.note}."))
            + " La banda no dice «compra»: dice si pagas más o menos que de costumbre por lo "
              "mismo. Detalle completo en 🏢 Empresa."
        )

# --- 5. The deposit ---------------------------------------------------------------------
st.subheader("5 · El depósito: de pesos a IBKR")
st.markdown(
    "Dos costes viajan con cada aporte y piden hábitos opuestos:\n"
    "- **Proporcional** — el diferencial del cambio frente a la TRM. Es el mismo porcentaje "
    "con 100 que con 1.000 dólares; agrupar no lo reduce. Solo lo baja un mejor cambio.\n"
    "- **Fijo** — una tarifa por transferencia, la comisión mínima por orden. Pesa más cuanto "
    "más pequeño el aporte; **agrupar es la única palanca**.")

cash_tx = app_data.cash_transactions()
deposits = cash_tx[cash_tx["kind"] == "deposit"] if not cash_tx.empty else pd.DataFrame()
trm = app_data.observations("banrep", app_data.db_mtime())
trm = trm[trm["series_id"] == "TRM:COP_USD"] if not trm.empty else trm
costs = fx.deposit_costs(deposits, funding_cfg.get("deposits") or [], trm)

if costs.empty:
    st.caption("Todavía no hay depósitos en el extracto de IBKR.")
else:
    st.dataframe(pd.DataFrame({
        "Abono en IBKR": costs["date"],
        "USD recibidos": costs["usd_received"],
        "COP pagados": costs["cop_paid"],
        "TRM": costs["trm"],
        "TRM del": costs["trm_date"],
        "Coste (COP)": costs["cost_cop"],
        "Coste": costs["cost_pct"],
    }), hide_index=True, width="stretch", column_config={
        "USD recibidos": st.column_config.NumberColumn(format="localized"),
        "COP pagados": st.column_config.NumberColumn(format="localized"),
        "TRM": st.column_config.NumberColumn(format="localized"),
        "Coste (COP)": st.column_config.NumberColumn(format="localized"),
        "Coste": st.column_config.NumberColumn(format="percent"),
    })
    st.caption("Coste de punta a punta = COP pagados − USD abonados × TRM. Recoge el "
               "diferencial del cambio y **todas** las tarifas del camino (intermediario, "
               "bancos corresponsales, IBKR), sin suponer ninguna. IBKR no cargó nada por "
               "recibir estos depósitos: no hay ninguna tarifa en el extracto.")
    missing = costs[costs["cop_paid"].isna()]
    if not missing.empty:
        snippet = "funding:\n  deposits:\n" + "".join(
            f"    - {{date: {d}, cop_paid: null, bought_on: null}}\n" for d in missing["date"])
        st.markdown("Para medir el coste, apunta en `config/settings.local.yaml` los pesos que "
                    "pagaste por cada depósito (lo único que el extracto de IBKR no sabe). "
                    "`bought_on` es el día que compraste los dólares, si no fue el del abono:")
        st.code(snippet, language="yaml")

split = fx.split_costs(costs)
fixed_usd = funding_cfg.get("fixed_fee_usd")
proportional = funding_cfg.get("proportional_cost")
source = "config"
if split is not None and (fixed_usd is None or proportional is None):
    rate_now, _ = fx.trm_on(trm, today)
    fixed_usd = split.fixed_cop / rate_now if rate_now else None
    proportional = split.proportional
    source = f"ajuste lineal sobre {split.deposits} depósitos"
if fixed_usd is not None or proportional is not None:
    st.markdown(f"Coste fijo por transferencia **{money(fixed_usd, public=False)}** · "
                f"proporcional **{pct(proportional, decimals=2)}** — {source}.")

monthly = float(contribution.get("monthly_amount_usd") or amount)
table = fx.batching(monthly, every_months=[1, 2, 3, 6],
                    fixed_usd=None if fixed_usd is None else float(fixed_usd),
                    proportional=None if proportional is None else float(proportional),
                    fee_per_order=fee, orders=1)
st.markdown(f"**Aportando {money(monthly, public=False)} al mes, transferir cada…**")
st.dataframe(pd.DataFrame({
    "Cada (meses)": table["every_months"],
    "Transferencia (USD)": table["amount_usd"],
    "Fijo": table["fixed"],
    "Cambio": table["proportional"],
    "Comisión (1 orden)": table["commission"],
    "Total": table["total"],
    "Fijos al año (USD)": table["fixed_and_commission_usd_year"],
}), hide_index=True, width="stretch", column_config={
    "Transferencia (USD)": st.column_config.NumberColumn(format="localized"),
    "Fijo": st.column_config.NumberColumn(format="percent"),
    "Cambio": st.column_config.NumberColumn(format="percent"),
    "Comisión (1 orden)": st.column_config.NumberColumn(format="percent"),
    "Total": st.column_config.NumberColumn(format="percent"),
    "Fijos al año (USD)": st.column_config.NumberColumn(format="localized"),
})
if fixed_usd is None:
    st.caption(f"Columnas vacías: {MISSING} = coste todavía no medido. Se llenan al apuntar los "
               "pesos pagados de al menos tres depósitos de distinto tamaño, o al escribir "
               "`funding.fixed_fee_usd` y `funding.proportional_cost` si conoces la tarifa.")
st.caption(
    "Lo que la tabla no pone en número: agrupar deja el dinero esperando en pesos en vez de "
    "invertido. Con un horizonte de años eso cuesta poco — la validación del freno (fase 4) "
    "midió que aplazar aportes apenas cambió el resultado —, pero ponerle cifra exigiría un "
    "rendimiento esperado, que es un supuesto tuyo y no una medición. Regla práctica: si el "
    "fijo por transferencia pesa más que un mes de rentabilidad esperada del dinero que "
    "espera, agrupa.")


# --- 6. Before selling (§15.5, point 1) -------------------------------------------------
st.subheader("6 · Antes de vender")
all_trades = app_data.trades()
sold_or_held = sorted({str(t) for t in all_trades["ticker"]} | set(held) | {"SPY"}) \
    if not all_trades.empty else sorted(set(held) | {"SPY"})
history_prices = app_data.prices_for(sold_or_held)
history_actions = app_data.corporate_actions()


def total_return(ticker: str) -> pd.Series:
    own = history_prices[history_prices["series_id"] == f"{ticker}:close_raw"]
    if own.empty:
        return pd.Series(dtype=float)
    return as_series(total_return_index(own.assign(ts=own["ts"].str[:10]), history_actions,
                                        ticker, today.date().isoformat()))


sales = bh.sales_after(all_trades, {t: total_return(t) for t in sold_or_held}, today)
kept_rising = int((sales["after"] > 0).sum()) if not sales.empty else 0
missed = sales["after_usd"].sum(min_count=1) if not sales.empty else None
missed_spy = (sales["spy_after"] * sales["proceeds"]).sum(min_count=1) \
    if not sales.empty else None
history_line = (
    f"Tus ventas anteriores: **{kept_rising} de {len(sales)} siguieron subiendo** después de "
    f"venderlas, y en conjunto lo vendido "
    + (f"ganó {money(missed, public=False)} más" if missed >= 0 else
       f"perdió {money(-missed, public=False)}: vender lo evitó")
    + (f" (ese dinero en SPY: {money(missed_spy, public=False)})"
       if missed_spy is not None and pd.notna(missed_spy) else "") + "."
    if not sales.empty and missed is not None and pd.notna(missed) else "")

if not held:
    st.caption("Sin posiciones abiertas: nada que vender.")
else:
    min_days = int(BEHAVIOR.get("min_holding_days", 90))
    open_lots, _disposals, _unmatched = port.fifo_lots(all_trades)
    sell_pick = st.selectbox("Qué piensas vender", held, key="sell_pick")
    sell_cik = str(by_ticker.get(sell_pick) or (tracked_cards.get(sell_pick) or {}).get("cik")
                   or "").zfill(10)
    fundamentals = th.evaluable_metrics(sec_obs, sell_cik, today, level3.get("hurdle_rate")) \
        if sell_cik.strip("0") else {}
    check = dc.before_selling(sell_pick, tracked_cards.get(sell_pick), fundamentals,
                              valued_held, open_lots, today, min_holding_days=min_days)
    rule_state = {True: ("CRUZADO", BAD), False: ("no cruzado", GOOD),
                  None: ("no evaluable" if check.has_card else "sin ficha", None)}
    state, state_tone = rule_state[check.invalidation_breached]
    rows6 = [
        ("✅" if check.has_card else "⚠️", "Ficha de tesis",
         "escrita" if check.has_card else "sin ficha — no hay criterio escrito"),
        ("🔴" if check.invalidation_breached else "⚪", "Criterio de invalidación",
         colored(state, state_tone)
         + (f" — «{check.invalidation}»" if check.invalidation else "")),
        ("🎯" if check.exit_triggered else "⚪", "Reglas de salida",
         (f"{len(check.exit_triggered)} alcanzada(s) de {check.exit_rules}"
          if check.exit_rules else "ninguna escrita — §2 las pide antes de comprar")
         + (f"; sin evaluar: {', '.join(check.exit_unevaluable)}"
            if check.exit_unevaluable else "")),
        ("⚠️" if check.before_horizon else "⚪", "Tiempo en cartera",
         (f"{check.held_days} días el lote más antiguo, {check.newest_lot_days} el más nuevo; "
          f"tu horizonte mínimo escrito: {min_days} días")
         if check.held_days is not None else MISSING),
        ("⚪", "Resultado abierto", pct(check.unrealized_return)),
    ]
    st.markdown("| | Comprobación | Estado |\n|---|---|---|\n"
                + "\n".join(f"| {i} | {c} | {v} |" for i, c, v in rows6))
    for rule in check.exit_triggered:
        st.markdown(f"🎯 **{rule['rule_id']}** ({rule['kind']}) — disparador escrito: "
                    f"«{rule['trigger']}» → acción escrita: «**{rule['action']}**». "
                    f"{rule['detail']}")
    for rule in check.exit_prose:
        st.markdown(f"📝 **{rule.get('rule_id')}** (solo texto, la juzgas tú): "
                    f"«{rule.get('trigger')}» → «{rule.get('action')}»")
    reasons = " ".join(check.reasons)
    if check.backing == dc.BACKED:
        st.info(f"**Una regla que escribiste antes respalda esta venta.** {reasons} Ejecuta "
                "lo que dice la acción escrita: una regla de rebalanceo pide vender una "
                "parte, no cerrar.")
    else:
        st.warning(
            ("**Ninguna regla escrita respalda esta venta.** " if check.backing == dc.UNBACKED
             else "**Sin ficha no hay criterio de venta.** ")
            + reasons + (" " + history_line if history_line else "")
            + " Si aun así vendes, escribe por qué en 📓 Diario **antes** de hacerlo.")
    st.caption("El panel no bloquea ninguna orden y no dice «vende» ni «mantén»: pone delante "
               "lo que escribiste y lo que hicieron tus ventas anteriores. La decisión es "
               "tuya (§2, §12).")

if not sales.empty:
    with st.expander(f"Tus ventas y lo que pasó después ({len(sales)})"):
        st.dataframe(pd.DataFrame({
            "Fecha": sales["day"].dt.date.astype(str),
            "Empresa": sales["ticker"],
            "Días tenida": sales["holding_days"].round(0),
            "Resultado de la venta": sales["realized_return"],
            "Lo que hizo después": sales["after"],
            "SPY después": sales["spy_after"],
            "Dejado de ganar (USD)": sales["after_usd"],
        }), hide_index=True, width="stretch", column_config={
            "Resultado de la venta": st.column_config.NumberColumn(format="percent"),
            "Lo que hizo después": st.column_config.NumberColumn(format="percent"),
            "SPY después": st.column_config.NumberColumn(format="percent"),
            "Dejado de ganar (USD)": st.column_config.NumberColumn(format="localized"),
        })
        st.caption("Rentabilidad total (con dividendos) desde el día de la venta hasta hoy. "
                   "«Dejado de ganar» negativo = vender evitó esa pérdida. Días tenida, "
                   "ponderados por importe cuando la venta cerró varios lotes (FIFO).")
