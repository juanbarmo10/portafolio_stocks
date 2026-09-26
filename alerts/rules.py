"""Alert rules, declared in config and evaluated after each ingest (CLAUDE.md sections 2, 8).

Section 2 allows alerts **only for actionable events, under rules written in advance**. The
panel is pull, not push, and an alert is the one exception — so every rule here answers
"is there something to *do*, or to *not do*, today?", and none of them is about price
movement (section 12 forbids price alerts outright).

The rules (``alerts.rules`` in settings.yaml), in the order of section 8, phase 4:

=====================  ==================================================================
macro_release_soon     CPI, PCE, payrolls or FOMC within N hours → do not contribute today
earnings_soon          results of a held company within the section-2 window
regime_risk_off        the regime light turned red → purchases blocked
invalidation_crossed   a held company's structured invalidation rule is met
review_overdue         a written thesis passed its ``review_date``
exit_ladder_triggered  a held company's structured exit rule is met
amended_filing         a held company filed an amended 10-K/10-Q → possible restatement
filing_signal          a held or studied company filed something actionable: late filing,
                       non-reliance, delisting notice, auditor change, impairment, or a
                       shelf / sale while burning cash (transform/filing_signals.py)
(ingest failure)       not a rule: ``run_ingest.py`` raises it from its own failure list
=====================  ==================================================================

Shape: :func:`load_snapshot` reads the database once into a :class:`Snapshot`; each rule is
a pure function of the snapshot; :func:`dispatch` deduplicates against ``alerts_log`` and
sends. Every alert carries a ``key`` built so that a *persistent* condition is sent once —
the key names the event (a release, a filing, a review date, a red spell's first day), not
the day it was noticed.

A condition that cannot be evaluated is not a quiet "all clear" (section 12): a held
company with no CIK or no earnings date is said once, instead of the earnings rule silently
skipping it forever.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import pandas as pd

from core.config import Settings
from core.logging_setup import get_logger
from ingest.macro_calendar import current_calendar
from transform import portfolio
from transform import regime as rg
from transform import thesis as th

log = get_logger(__name__)

NEW_YORK = "America/New_York"


@dataclass(frozen=True)
class Alert:
    """One fired condition. ``key`` is its ``alert_id`` in ``alerts_log``."""

    rule: str
    key: str
    text: str


@dataclass
class Snapshot:
    """Everything the rules look at, read once. Plain data: no connection inside."""

    now: pd.Timestamp                                   # UTC
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    filings: pd.DataFrame = field(default_factory=pd.DataFrame)
    held: dict[str, str | None] = field(default_factory=dict)   # ticker → CIK or None
    valued: pd.DataFrame = field(default_factory=pd.DataFrame)  # portfolio.valuation()
    tracked: list[Mapping[str, Any]] = field(default_factory=list)
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)  # source='sec'
    regime: pd.DataFrame | None = None                  # regime_frame(), or None
    regime_labels: dict[str, str] = field(default_factory=dict)
    earnings_window_days: int = th.EARNINGS_WINDOW_DAYS
    hurdle_rate: float | None = None
    researched: dict[str, str] = field(default_factory=dict)   # CIK → ticker (studied)

    @property
    def today(self) -> str:
        return self.now.date().isoformat()


# --- Rules ------------------------------------------------------------------------------


def macro_release_soon(snap: Snapshot, params: Mapping[str, Any]) -> list[Alert]:
    """A high-impact release or FOMC decision inside the next ``within_hours``."""
    within = float(params.get("within_hours", 48))
    calendar = current_calendar(snap.events)
    out: list[Alert] = []
    for row in calendar.to_dict("records"):
        when = pd.Timestamp(row["ts"])
        hours = (when - snap.now).total_seconds() / 3600
        if not 0 <= hours <= within:
            continue
        local = when.tz_convert(NEW_YORK)
        out.append(Alert("macro_release_soon", f"macro_release_soon:{row['event_id']}", (
            f"📅 {row['label']}: {local:%Y-%m-%d} a las {local:%H:%M} hora de Nueva York "
            f"(en {hours:.0f} h).\n"
            "Dato de alto impacto: no ejecutes el aporte hasta después de publicarse "
            "(§2, nivel 1)."
        )))
    return out


def earnings_soon(snap: Snapshot, params: Mapping[str, Any]) -> list[Alert]:
    """Results of a held company within the window of section 2."""
    window = int(params.get("within_days") or snap.earnings_window_days)
    today = pd.Timestamp(snap.today)
    earnings = (snap.events[snap.events["category"] == "earnings"]
                if not snap.events.empty else pd.DataFrame())
    out: list[Alert] = []
    for ticker, cik in sorted(snap.held.items()):
        if cik is None:
            out.append(Alert("earnings_soon", f"earnings_soon:no_cik:{ticker}", (
                f"ℹ️ {ticker} está en cartera y no tiene CIK de empresa en la SEC, así que "
                "el panel no le conoce fecha de resultados. Si es un ETF es lo esperado; "
                "si es una empresa, revisa el mapa de tickers (§9.3). Aviso único."
            )))
            continue
        upcoming = earnings[(earnings["cik"] == cik)
                            & (pd.to_datetime(earnings["ts"].str[:10]) >= today)] \
            if not earnings.empty else earnings
        if upcoming.empty:
            out.append(Alert("earnings_soon", f"earnings_soon:no_date:{cik}", (
                f"ℹ️ {ticker} está en cartera y el panel no tiene ninguna fecha próxima de "
                "resultados: ni anunciada ni estimable desde sus 8-K. Revísala a mano "
                "antes de ampliar la posición (§2). Aviso único."
            )))
            continue
        row = upcoming.sort_values("ts").iloc[0]
        day = str(row["ts"])[:10]
        days = (pd.Timestamp(day) - today).days
        if days > window:
            continue
        kind = "estimada" if int(row.get("is_estimated") or 0) else "confirmada"
        out.append(Alert("earnings_soon", f"earnings_soon:{cik}:{day}", (
            f"📊 {ticker}: resultados el {day} (en {days} días, fecha {kind}).\n"
            f"No se abre ni se amplía la posición en los {window} días previos sin una "
            "decisión explícita (§2, nivel 3)."
        )))
    return out


def red_spell_start(verdicts: pd.Series, quiet_sessions: int) -> pd.Timestamp | None:
    """First session of the red spell the last verdict belongs to, or ``None`` if not red.

    A spell survives short excursions to amber: it only ends after ``quiet_sessions``
    sessions without red. The light flickers — 43 entries into red since 2012, many a few
    days apart — and one message per entry would be the drip-feed section 2 forbids. The
    verdict itself is untouched; this only decides when a red is *news*.
    """
    judged = verdicts[verdicts != rg.INSUFFICIENT]
    if judged.empty or judged.iloc[-1] != rg.RISK_OFF:
        return None
    red = (judged == rg.RISK_OFF).to_numpy()
    start = len(red) - 1
    gap = 0
    for i in range(len(red) - 2, -1, -1):
        if red[i]:
            start, gap = i, 0
        else:
            gap += 1
            if gap >= quiet_sessions:
                break
    return judged.index[start]


def regime_risk_off(snap: Snapshot, params: Mapping[str, Any]) -> list[Alert]:
    """The light turned red. Keyed on the spell's first day: one message per spell."""
    frame = snap.regime
    if frame is None or frame.empty:
        return []
    spell = red_spell_start(frame["verdict"], int(params.get("quiet_sessions", 20)))
    if spell is None:
        return []
    last = frame[frame["verdict"] != rg.INSUFFICIENT].iloc[-1]
    off = [label for key, label in snap.regime_labels.items()
           if key in frame.columns and last[key] == -1]
    return [Alert("regime_risk_off", f"regime_risk_off:{spell.date()}", (
        f"🔴 Semáforo de régimen en RISK-OFF (episodio desde el {spell.date()}; "
        f"hoy {int(last['off'])} de {int(last['available'])} componentes en contra: "
        f"{', '.join(off)}).\n"
        "Bloquea compras aunque el nivel 3 sea perfecto (§2). No es una señal de venta, y "
        "el semáforo confirma más que anticipa (RESEARCH.md §2.27)."
    ))]


def _held_cards(snap: Snapshot) -> list[tuple[str, Mapping[str, Any]]]:
    """Written cards of the companies actually held, with their normalised CIK."""
    held_ciks = {cik for cik in snap.held.values() if cik}
    out = []
    for card in snap.tracked:
        cik = str(card.get("cik", "")).zfill(10)
        if cik in held_ciks:
            out.append((cik, card))
    return out


def invalidation_crossed(snap: Snapshot, params: Mapping[str, Any]) -> list[Alert]:
    """A held company's structured invalidation rule is met.

    Keyed on the value that breached, so a new quarterly figure that still breaches is
    said again — once per filing, not once per day.
    """
    out: list[Alert] = []
    for cik, card in _held_cards(snap):
        rule = card.get("invalidation_rule")
        if not rule:
            continue
        metrics = th.evaluable_metrics(snap.fundamentals, cik, snap.today, snap.hurdle_rate)
        breached, detail = th.evaluate_rule(rule, metrics)
        if not breached:
            continue
        value = metrics.get(str(rule.get("metric")))
        out.append(Alert("invalidation_crossed", (
            f"invalidation_crossed:{cik}:{rule.get('metric')}{rule.get('operator')}"
            f"{rule.get('threshold')}:{value:.6g}"
        ), (
            f"⚠️ {card.get('ticker')}: criterio de invalidación CRUZADO.\n"
            f"{detail.replace('**', '')}\n"
            f"Lo que escribiste: «{card.get('invalidation')}»\n"
            "El panel no decide que la tesis ha muerto; te avisa de que tu regla se cumplió."
        )))
    return out


def review_overdue(snap: Snapshot, params: Mapping[str, Any]) -> list[Alert]:
    """A written thesis passed its review date. Once per date: rewriting it re-arms."""
    out: list[Alert] = []
    for card in snap.tracked:
        review = str(card.get("review_date") or "")
        if not review or review >= snap.today:
            continue
        cik = str(card.get("cik", "")).zfill(10)
        out.append(Alert("review_overdue", f"review_overdue:{cik}:{review}", (
            f"🗓️ {card.get('ticker')}: la revisión de la tesis venció el {review}.\n"
            "Una tesis sin revisar se sostiene por inercia. Reléela, confírmala o "
            "invalídala, y pon una fecha nueva (§5.2)."
        )))
    return out


def _position_metrics(snap: Snapshot, ticker: str) -> dict[str, float | None]:
    """``price``, ``weight`` and ``return_pct`` of one held position."""
    rows = snap.valued[snap.valued["ticker"] == ticker] if not snap.valued.empty else []
    if len(rows) == 0:
        return {"price": None, "weight": None, "return_pct": None}
    row = rows.iloc[0]

    def clean(value: Any) -> float | None:
        return None if pd.isna(value) else float(value)

    return {"price": clean(row["price"]), "weight": clean(row["weight"]),
            "return_pct": clean(row["unrealized_return"])}


def exit_ladder_triggered(snap: Snapshot, params: Mapping[str, Any]) -> list[Alert]:
    """A held company's structured exit rule is met. Editing the threshold re-arms it."""
    out: list[Alert] = []
    for cik, card in _held_cards(snap):
        ladder = [r for r in card.get("exit_ladder") or [] if r.get("rule")]
        if not ladder:
            continue
        metrics = {
            **th.evaluable_metrics(snap.fundamentals, cik, snap.today, snap.hurdle_rate),
            **_position_metrics(snap, str(card.get("ticker"))),
        }
        for exit_rule in ladder:
            rule = exit_rule["rule"]
            breached, detail = th.evaluate_rule(rule, metrics)
            if not breached:
                continue
            out.append(Alert("exit_ladder_triggered", (
                f"exit_ladder_triggered:{exit_rule['rule_id']}:{rule.get('operator')}"
                f"{rule.get('threshold')}"
            ), (
                f"🎯 {card.get('ticker')}: regla de salida {exit_rule['rule_id']} "
                f"({exit_rule['kind']}) alcanzada.\n"
                f"{detail.replace('**', '')}\n"
                f"Disparador escrito: «{exit_rule['trigger']}»\n"
                f"Acción escrita: «{exit_rule['action']}»\n"
                "La escribiste antes de comprar; el panel no ejecuta nada (§12)."
            )))
    return out


def amended_filing(snap: Snapshot, params: Mapping[str, Any]) -> list[Alert]:
    """A held company filed an amended financial statement within ``lookback_days``.

    The lookback keeps an old amendment — TMUS's 10-Q/A of 2020 — from arriving as news
    on the first run; the company page shows the whole history.
    """
    lookback = int(params.get("lookback_days", 30))
    if snap.filings.empty:
        return []
    since = (snap.now - pd.Timedelta(days=lookback)).date().isoformat()
    by_cik = {cik: ticker for ticker, cik in snap.held.items() if cik}
    amended = snap.filings[
        (snap.filings["is_amended"] == 1)
        & snap.filings["cik"].isin(by_cik)
        & (snap.filings["filed_date"] >= since)
        & snap.filings["form"].str[:-2].str.upper().isin(th.FINANCIAL_FORMS)
    ]
    return [
        Alert("amended_filing", f"amended_filing:{row['accession']}", (
            f"🚩 {by_cik[row['cik']]} presentó un {row['form']} el {row['filed_date']}: "
            "estados financieros enmendados, posible reexpresión.\n"
            "Es bandera de gobernanza, no prueba: léelo antes de cualquier decisión "
            f"(§9.6).\n{row.get('url') or ''}"
        ).rstrip())
        for row in amended.to_dict("records")
    ]


def filing_signal(snap: Snapshot, params: Mapping[str, Any]) -> list[Alert]:
    """An actionable filing by a held or studied company within ``lookback_days``.

    One alert per filing and signal (keyed on the accession). Cash burn — negative TTM free
    cash flow from the SEC facts — is what promotes a shelf or a sale to a dilution warning;
    unknown burn promotes nothing (section 12).
    """
    from transform import filing_signals as fs  # noqa: PLC0415
    from transform import fundamentals as fun  # noqa: PLC0415

    if snap.filings.empty:
        return []
    lookback = int(params.get("lookback_days", 14))
    since = (snap.now - pd.Timedelta(days=lookback)).date().isoformat()
    companies = {**snap.researched, **{cik: t for t, cik in snap.held.items() if cik}}
    alerts = []
    for cik, ticker in sorted(companies.items()):
        found = fs.signals(snap.filings, cik, since, snap.today)
        if found.empty:
            continue
        fcf = fun.free_cash_flow(snap.fundamentals, cik, snap.today) \
            if not snap.fundamentals.empty else None
        for w in fs.warnings(found, None if fcf is None else fcf < 0):
            accession = next((str(r["accession"]) for r in snap.filings[
                (snap.filings["cik"] == cik) & (snap.filings["form"] == w.form)
                & (snap.filings["filed_date"].astype(str).str[:10] == w.date)
            ].to_dict("records")), f"{w.date}:{w.form}")
            mark = "🚩" if w.severity == fs.RED else "⚠️"
            alerts.append(Alert("filing_signal", f"filing_signal:{accession}:{w.text[:20]}", (
                f"{mark} {ticker} presentó un {w.form} el {w.date}: {w.text}.\n"
                "Léelo antes de cualquier decisión; el panel no interpreta el documento."
                + (f"\n{w.url}" if w.url else "")
            )))
    return alerts


RULES: dict[str, Callable[[Snapshot, Mapping[str, Any]], list[Alert]]] = {
    "macro_release_soon": macro_release_soon,
    "earnings_soon": earnings_soon,
    "regime_risk_off": regime_risk_off,
    "invalidation_crossed": invalidation_crossed,
    "review_overdue": review_overdue,
    "exit_ladder_triggered": exit_ladder_triggered,
    "amended_filing": amended_filing,
    "filing_signal": filing_signal,
}


def evaluate(snap: Snapshot, rules: Sequence[Mapping[str, Any]]) -> tuple[list[Alert], list[str]]:
    """Run every configured rule. Returns ``(alerts, failures)``.

    An unknown kind or a rule that raises is a failure, reported — never a rule that
    quietly stopped existing. The others still run.
    """
    alerts: list[Alert] = []
    failures: list[str] = []
    for params in rules:
        kind = str(params.get("kind"))
        rule = RULES.get(kind)
        if rule is None:
            failures.append(f"unknown alert rule kind {kind!r}")
            continue
        try:
            alerts.extend(rule(snap, params))
        except Exception as exc:  # noqa: BLE001 — one broken rule must not sink the rest
            log.exception("Alert rule '%s' failed.", kind)
            failures.append(f"{kind}: {exc}")
    return alerts, failures


def ingest_failure_alerts(failures: Sequence[str], today: str) -> list[Alert]:
    """One alert per failed source per day, from ``run_ingest.py``'s own list.

    This is where an expired IBKR Flex token surfaces (section 4.1): the ingester fails
    loudly, the runner collects it, and it arrives here instead of only in a log nobody
    reads.
    """
    return [
        Alert("ingest_failure", f"ingest_failure:{failure.split(' ', 1)[0]}:{today}", (
            f"❌ Fallo de ingesta: {failure}\n"
            "Los datos de esa fuente no se actualizaron hoy; el panel sigue con los de "
            "la última ejecución buena. Revisa el log de run_ingest.py."
        ))
        for failure in failures
    ]


# --- Delivery and deduplication ---------------------------------------------------------


class Sender(Protocol):
    def send(self, text: str) -> bool: ...


def already_delivered(conn: Any, key: str) -> bool:
    """True only if this alert was **delivered**. A logged-only one is retried later —
    once Telegram is configured, what is still true gets sent."""
    row = conn.execute("SELECT payload FROM alerts_log WHERE alert_id = ?", (key,)).fetchone()
    if not row or not row[0]:
        return False
    try:
        return bool(json.loads(row[0]).get("delivered"))
    except ValueError:
        return False


def dispatch(conn: Any, alerts: Sequence[Alert], sender: Sender, *, record: bool = True) -> int:
    """Send what has not been delivered yet and log it in ``alerts_log``. Returns sent."""
    from db.loader import upsert_alerts  # noqa: PLC0415 — keeps the rules importable alone

    sent = 0
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for alert in alerts:
        if already_delivered(conn, alert.key):
            continue
        delivered = sender.send(alert.text)
        sent += int(delivered)
        if record:
            upsert_alerts(conn, [{
                "alert_id": alert.key, "rule_id": alert.rule, "fired_at": now,
                "payload": json.dumps({"delivered": delivered, "text": alert.text},
                                      ensure_ascii=False),
            }])
    return sent


# --- Reading the database into a snapshot -----------------------------------------------


def load_snapshot(conn: Any, settings: Settings, now: pd.Timestamp | None = None) -> Snapshot:
    """Read everything the rules need, once."""
    from db.database import read_observations, read_table  # noqa: PLC0415

    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    level2 = settings.raw["panel"]["level2"]
    level3 = settings.raw["panel"].get("level3", {})

    account = read_observations(conn, source="ibkr")
    positions = portfolio.latest_positions(account)
    tickers = list(positions["ticker"])
    prices = read_observations(conn, series_ids=[f"{t}:close_raw" for t in tickers]) \
        if tickers else pd.DataFrame(columns=["series_id", "ts", "value"])
    valued = portfolio.valuation(positions, portfolio.latest_prices(prices))

    companies = read_table(conn, "companies")
    by_ticker = dict(zip(companies["ticker"], companies["cik"])) if not companies.empty else {}
    held = {t: by_ticker.get(t) for t in tickers}

    tracked = settings.tracked_companies
    researched = {str(c["cik"]).zfill(10): str(c["ticker"])
                  for c in settings.researched_companies if c.get("cik") and c.get("ticker")}
    fundamentals = read_observations(conn, source="sec") if tracked or researched \
        else pd.DataFrame()

    etf_prices = read_observations(
        conn, series_ids=[f"{t}:close_raw" for t in rg.regime_tickers(level2)])
    built = rg.build(read_observations(conn, source="fred"), etf_prices,
                     read_table(conn, "corporate_actions"), level2)

    return Snapshot(
        now=now,
        events=read_table(conn, "events"),
        filings=read_table(conn, "filings"),
        held=held,
        valued=valued,
        tracked=tracked,
        fundamentals=fundamentals,
        regime=None if built is None else built.frame,
        regime_labels={} if built is None else {r.key: r.label for r in built.rules},
        earnings_window_days=int(level3.get("earnings_window_days", th.EARNINGS_WINDOW_DAYS)),
        hurdle_rate=level3.get("hurdle_rate"),
        researched=researched,
    )


def exit_ladder_rows(tracked: Sequence[Mapping[str, Any]], now: str) -> list[dict[str, Any]]:
    """The written exit rules as ``exit_ladder`` rows. ``created_at`` is kept from the
    first sync by the loader, so the table records when each rule was first written."""
    rows = []
    for card in tracked:
        for rule in card.get("exit_ladder") or []:
            rows.append({
                "rule_id": str(rule["rule_id"]), "cik": str(card["cik"]).zfill(10),
                "kind": rule["kind"], "trigger": rule["trigger"], "action": rule["action"],
                "created_at": now,
            })
    return rows
