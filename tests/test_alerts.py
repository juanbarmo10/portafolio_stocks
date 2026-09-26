"""Alert rules, deduplication and delivery (section 8, phase 4).

Every rule is a pure function of a snapshot, so each is tested on a hand-built one whose
answer is known in advance. Delivery runs against a fake sender; nothing reaches Telegram.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from alerts import rules as ar
from alerts.telegram import TelegramSender, redact
from core.config import Settings, load_settings
from db import loader

NOW = pd.Timestamp("2026-10-19T12:00:00Z")
TMUS, UBER = "0001283699", "0001543151"


def events(*rows):
    base = {"cik": None, "is_estimated": 0, "label": "x", "payload": "{}"}
    return pd.DataFrame([{**base, **row} for row in rows])


def macro(event_id, ts, as_of="2026-10-18", release_id=10):
    return {"event_id": event_id, "category": "macro", "ts": ts, "label": "CPI",
            "payload": json.dumps({"calendar_as_of": as_of, "release_id": release_id})}


def snap(**kw):
    return ar.Snapshot(now=NOW, **kw)


# --- macro_release_soon -----------------------------------------------------------------


def test_a_release_inside_48_hours_fires_and_says_what_not_to_do():
    alerts = ar.macro_release_soon(snap(events=events(
        macro("fred:10:2026-10-20", "2026-10-20T12:30:00+00:00"),
        macro("fred:10:2026-11-10", "2026-11-10T13:30:00+00:00"),
    )), {"within_hours": 48})
    assert [a.key for a in alerts] == ["macro_release_soon:fred:10:2026-10-20"]
    assert "08:30 hora de Nueva York" in alerts[0].text
    assert "no ejecutes el aporte" in alerts[0].text


def test_a_release_already_out_does_not_fire():
    alerts = ar.macro_release_soon(snap(events=events(
        macro("fred:10:2026-10-19", "2026-10-19T11:30:00+00:00"))), {})
    assert alerts == []


def test_a_rescheduled_release_does_not_fire_from_its_old_date():
    alerts = ar.macro_release_soon(snap(events=events(
        macro("fred:10:2026-10-20", "2026-10-20T12:30:00+00:00", as_of="2026-10-01"),
        macro("fred:10:2026-10-28", "2026-10-28T12:30:00+00:00", as_of="2026-10-18"),
    )), {})
    assert alerts == []


# --- earnings_soon ----------------------------------------------------------------------


def earnings(cik, ts, estimated=1):
    return {"event_id": f"{cik}:earnings:next", "category": "earnings", "cik": cik,
            "ts": ts, "is_estimated": estimated}


def test_results_of_a_position_inside_the_window_fire_with_their_kind():
    alerts = ar.earnings_soon(snap(
        held={"TMUS": TMUS, "UBER": UBER},
        events=events(earnings(TMUS, "2026-10-22"), earnings(UBER, "2026-11-03")),
    ), {})
    assert [a.key for a in alerts] == [f"earnings_soon:{TMUS}:2026-10-22"]
    assert "en 3 días, fecha estimada" in alerts[0].text


def test_results_of_something_not_held_do_not_fire():
    alerts = ar.earnings_soon(snap(held={}, events=events(earnings(TMUS, "2026-10-20"))), {})
    assert alerts == []


def test_a_held_position_the_rule_cannot_judge_is_said_not_skipped():
    """Silence would read as "no results coming"."""
    alerts = ar.earnings_soon(snap(held={"SPY": None, "TMUS": TMUS}, events=events()), {})
    assert sorted(a.key for a in alerts) == [
        "earnings_soon:no_cik:SPY", f"earnings_soon:no_date:{TMUS}",
    ]


# --- regime_risk_off --------------------------------------------------------------------


def regime(verdicts):
    index = pd.bdate_range("2026-01-01", periods=len(verdicts))
    frame = pd.DataFrame({"verdict": verdicts, "off": 5, "available": 8, "credit": -1},
                         index=index)
    return frame


def test_red_fires_keyed_on_the_first_day_of_the_spell():
    frame = regime(["risk_on"] * 5 + ["risk_off"] * 3)
    alerts = ar.regime_risk_off(snap(regime=frame, regime_labels={"credit": "Crédito"}), {})
    assert [a.key for a in alerts] == ["regime_risk_off:2026-01-08"]
    assert "Crédito" in alerts[0].text and "No es una señal de venta" in alerts[0].text


def test_flickering_through_amber_is_one_spell_not_many():
    """43 entries into red since 2012; one message each would be the drip section 2 forbids."""
    frame = regime(["risk_off"] * 3 + ["neutral"] * 4 + ["risk_off"] * 2)
    assert ar.red_spell_start(frame["verdict"], quiet_sessions=20) == frame.index[0]
    assert ar.red_spell_start(frame["verdict"], quiet_sessions=3) == frame.index[7]


def test_insufficient_sessions_neither_break_nor_make_a_spell():
    frame = regime(["risk_off"] * 2 + ["insufficient"] * 30 + ["risk_off"])
    assert ar.red_spell_start(frame["verdict"], quiet_sessions=20) == frame.index[0]
    assert ar.red_spell_start(regime(["risk_on", "insufficient"])["verdict"], 20) is None


def test_green_or_amber_today_is_silent():
    assert ar.regime_risk_off(snap(regime=regime(["risk_off", "neutral"])), {}) == []
    assert ar.regime_risk_off(snap(regime=None), {}) == []


# --- written discipline: theses and exit rules ------------------------------------------


def card(**extra):
    base = {"ticker": "TMUS", "cik": TMUS, "thesis": "t", "value_accrual": "v",
            "key_metric": "k", "invalidation": "El FCF cae por debajo de X.",
            "review_date": "2027-01-01"}
    return {**base, **extra}


def test_an_overdue_review_fires_once_per_date():
    alerts = ar.review_overdue(snap(tracked=[card(review_date="2026-10-01"),
                                             card(ticker="X", cik="1", review_date="2027-01-01")]), {})
    assert [a.key for a in alerts] == [f"review_overdue:{TMUS}:2026-10-01"]


def test_an_invalidation_rule_fires_only_for_a_position(monkeypatch):
    monkeypatch.setattr(ar.th, "evaluable_metrics", lambda *a, **k: {"fcf_margin": 0.01})
    rule = {"metric": "fcf_margin", "operator": "<", "threshold": 0.05}
    tracked = [card(invalidation_rule=rule)]

    fired = ar.invalidation_crossed(snap(tracked=tracked, held={"TMUS": TMUS}), {})
    assert len(fired) == 1 and "CRUZADO" in fired[0].text
    assert "El FCF cae por debajo de X." in fired[0].text, "the prose travels verbatim"
    assert ar.invalidation_crossed(snap(tracked=tracked, held={}), {}) == []


def test_an_exit_rule_reads_the_position(monkeypatch):
    monkeypatch.setattr(ar.th, "evaluable_metrics", lambda *a, **k: {})
    valued = pd.DataFrame([{"ticker": "TMUS", "price": 250.0, "weight": 0.15,
                            "unrealized_return": 0.3}])
    exit_rule = {"rule_id": "tmus-tp", "kind": "take_profit", "trigger": "Peso > 12 %",
                 "action": "Vender hasta el 8 %",
                 "rule": {"metric": "weight", "operator": ">", "threshold": 0.12}}
    alerts = ar.exit_ladder_triggered(snap(
        tracked=[card(exit_ladder=[exit_rule])], held={"TMUS": TMUS}, valued=valued), {})
    assert [a.key for a in alerts] == ["exit_ladder_triggered:tmus-tp:>0.12"]
    assert "Vender hasta el 8 %" in alerts[0].text


def test_an_exit_rule_on_an_unknown_metric_does_not_fire(monkeypatch):
    monkeypatch.setattr(ar.th, "evaluable_metrics", lambda *a, **k: {})
    exit_rule = {"rule_id": "r", "kind": "rebalance", "trigger": "t", "action": "a",
                 "rule": {"metric": "nope", "operator": ">", "threshold": 1}}
    assert ar.exit_ladder_triggered(snap(
        tracked=[card(exit_ladder=[exit_rule])], held={"TMUS": TMUS}), {}) == []


# --- amended_filing ---------------------------------------------------------------------


def filing(accession, form, filed, cik=TMUS):
    return {"accession": accession, "cik": cik, "form": form, "period_end": None,
            "filed_date": filed, "is_amended": int(form.endswith("/A")), "url": "u"}


def test_only_a_recent_financial_amendment_of_a_position_fires():
    filings = pd.DataFrame([
        filing("a1", "10-Q/A", "2026-10-10"),
        filing("a2", "10-Q/A", "2020-08-10"),           # TMUS's real one: old news
        filing("a3", "8-K/A", "2026-10-10"),            # not the accounts
        filing("a4", "10-K/A", "2026-10-10", cik="9"),  # not held
    ])
    alerts = ar.amended_filing(snap(filings=filings, held={"TMUS": TMUS}),
                               {"lookback_days": 30})
    assert [a.key for a in alerts] == ["amended_filing:a1"]


# --- evaluation, dedup and delivery -----------------------------------------------------


def test_an_unknown_rule_kind_is_a_failure_not_a_silence():
    alerts, failures = ar.evaluate(snap(), [{"kind": "price_drop"}])
    assert alerts == [] and failures == ["unknown alert rule kind 'price_drop'"]


def test_the_shipped_config_names_only_known_rules():
    for params in load_settings().raw["alerts"]["rules"]:
        assert params["kind"] in ar.RULES


class FakeSender:
    def __init__(self, delivers=True):
        self.delivers, self.sent = delivers, []

    def send(self, text):
        self.sent.append(text)
        return self.delivers


@pytest.fixture()
def conn(tmp_path):
    connection = loader.init_db(tmp_path / "alerts.db")
    yield connection
    connection.close()


def test_a_delivered_alert_is_not_sent_twice(conn):
    alert = ar.Alert("r", "r:1", "hola")
    sender = FakeSender()
    assert ar.dispatch(conn, [alert], sender) == 1
    assert ar.dispatch(conn, [alert], sender) == 0
    assert sender.sent == ["hola"]


def test_an_undelivered_alert_goes_out_once_the_bot_works(conn):
    """Logged while Telegram was not configured; sent on the first run that can."""
    alert = ar.Alert("r", "r:1", "hola")
    assert ar.dispatch(conn, [alert], FakeSender(delivers=False)) == 0
    assert ar.dispatch(conn, [alert], FakeSender(delivers=True)) == 1
    payload = json.loads(conn.execute("SELECT payload FROM alerts_log").fetchone()[0])
    assert payload == {"delivered": True, "text": "hola"}


def test_ingest_failures_are_one_alert_per_source_per_day():
    alerts = ar.ingest_failure_alerts(["ibkr", "prices (AAPL, XYZ)"], "2026-10-19")
    assert [a.key for a in alerts] == ["ingest_failure:ibkr:2026-10-19",
                                       "ingest_failure:prices:2026-10-19"]


def test_exit_rules_keep_the_date_they_were_first_written(conn):
    rule = {"rule_id": "r1", "kind": "rebalance", "trigger": "t", "action": "a"}
    tracked = [card(exit_ladder=[rule])]
    loader.upsert_exit_ladder(conn, ar.exit_ladder_rows(tracked, "2026-10-01"))
    loader.upsert_exit_ladder(conn, ar.exit_ladder_rows(tracked, "2026-10-19"))
    assert conn.execute("SELECT created_at FROM exit_ladder").fetchone()[0] == "2026-10-01"


# --- Telegram ---------------------------------------------------------------------------


def settings_with(secrets):
    base = load_settings()
    return Settings(raw=base.raw, db_path=base.db_path, log_level="INFO",
                    secrets=secrets, public_mode=False)


def test_without_a_bot_nothing_leaves_the_machine(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a request was attempted")

    monkeypatch.setattr("alerts.telegram.requests.post", refuse)
    assert TelegramSender(settings_with({})).send("hola") is False


def test_secrets_are_redacted_before_a_message_leaves(monkeypatch):
    """An ingest-failure alert quotes an exception, and an exception can quote a key."""
    key = "fredkey0123456789abcdef"
    captured = {}

    class Response:
        status_code = 200

    def post(url, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr("alerts.telegram.requests.post", post)
    sender = TelegramSender(settings_with({"TELEGRAM_TOKEN": "tok-0123456789",
                                           "TELEGRAM_CHAT_ID": "42", "FRED_API_KEY": key}))
    assert sender.send(f"❌ fallo: https://x/?api_key={key}") is True
    assert key not in captured["text"] and "[REDACTED]" in captured["text"]
    assert "parse_mode" not in captured, "plain text: a stray '_' must not reject the alert"


def test_redact_leaves_short_values_alone():
    assert redact("chat 42", ["42", ""]) == "chat 42"


# --- filing_signal (2026-09-25) --------------------------------------------------------------

BURNER = "0001808805"   # shaped like NAUT: burns cash


def _fcf(cik, ocf, capex):
    rows = []
    for end in ("2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"):
        rows += [{"source": "sec", "series_id": f"{cik}:operating_cash_flow:q", "ts": end,
                  "ts_release": end, "value": ocf},
                 {"source": "sec", "series_id": f"{cik}:capex:q", "ts": end,
                  "ts_release": end, "value": capex}]
    return rows


def test_a_sale_by_a_company_burning_cash_is_a_dilution_warning_and_a_bond_is_not():
    filings = pd.DataFrame([
        filing("s1", "424B5", "2026-10-15", cik=BURNER),
        filing("s2", "424B5", "2026-10-15", cik=TMUS),     # a cash generator: likely debt
        filing("s3", "NT 10-Q", "2026-10-16", cik=TMUS),   # late: red whatever the cash
        filing("s4", "424B5", "2026-08-01", cik=BURNER),   # old news
    ] + [filing("s5", "8-K", "2026-10-17", cik=TMUS) | {"items": "4.02,9.01"}])
    fundamentals = pd.DataFrame(_fcf(BURNER, -10.0, 2.0) + _fcf(TMUS, 50.0, 10.0))
    alerts = ar.filing_signal(snap(filings=filings, held={"TMUS": TMUS},
                                   researched={BURNER: "NAUT"}, fundamentals=fundamentals),
                              {"lookback_days": 14})
    texts = sorted(a.text for a in alerts)
    assert len(alerts) == 3
    assert any("NAUT" in t and "acciones nuevas" in t for t in texts)
    assert any("TMUS" in t and "tardía" in t for t in texts)
    assert any("TMUS" in t and "4.02" in t for t in texts)
    assert not any("TMUS" in t and "424B5" in t for t in texts), "a bond is not dilution"
