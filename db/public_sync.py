"""Copy the PUBLIC part of the local database to the public deployment's (section 8 phase 5).

Decided by the user on 2026-09-24: the public panel shows market and companies only, **no
account data reaches the cloud**, and the local timer uploads it after the daily run. The
local database stays SQLite and complete; the public one is a filtered copy.

**An allow-list, like PUBLIC_PAGES.** What is copied is named here; anything new is private
until someone adds it. Never copied, whatever happens: the account (``ibkr`` observations,
trades, cash, securities), the local exchange rate (it names the jurisdiction), the short
interest of anything **not** researched (FINRA is also asked about held tickers; the
researched companies' is copied since 2026-09-25, by the user's decision), alerts, exit
rules, theses and the raw index membership. :func:`assert_public` re-checks the selection before any write.

**Which companies.** Only the ones written down for research (``tracked`` + ``watchlist``).
A position held without a card is known to the local base — its filings and calendar feed
the alerts — and must not surface publicly through the company list.

**The constituent breadth** travels as its computed series (``transform.breadth``), not as
the ~1.4 M member closes it comes from: the free cloud tier holds ~0.5 GB.

**Incremental.** The first run copies everything (``full=True``); after that, only rows whose
reference *or* publication date falls within ``lookback_days`` — the second condition is what
carries a restatement of an old quarter. Small tables are copied whole every time. Nothing is
ever deleted from the copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from core.config import Settings
from core.logging_setup import get_logger
from db import loader
from db.database import read_observations, read_table
from transform import breadth as br
from transform import regime as rg

log = get_logger(__name__)

PRIVATE_SOURCES = frozenset({"ibkr", "local_fx"})
# FINRA is public data, but its ingester also asks about what is held: only the researched
# companies' rows travel, and the guard checks every one.
FINRA_SOURCE = "finra"
FINRA_SERIES = ("short_interest", "days_to_cover", "short_interest:revised")
PRIVATE_TABLES = frozenset({"trades", "cash_transactions", "securities", "alerts_log",
                            "exit_ladder", "thesis_log", "universe_membership"})
TABLES = ("companies", "filings", "events", "corporate_actions")


@dataclass
class Selection:
    observations: pd.DataFrame
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def public_tickers(settings: Settings) -> list[str]:
    """Market references, the regime's ETFs and the researched companies — never holdings."""
    level2 = settings.raw["panel"]["level2"]
    researched = [str(c["ticker"]) for c in settings.researched_companies if c.get("ticker")]
    return list(dict.fromkeys([*settings.market_references, *rg.regime_tickers(level2),
                               *researched]))


def researched_tickers(settings: Settings) -> list[str]:
    return [str(c["ticker"]) for c in settings.researched_companies if c.get("ticker")]


def researched_ciks(settings: Settings) -> set[str]:
    return {str(c["cik"]).zfill(10) for c in settings.researched_companies if c.get("cik")}


def _recent(frame: pd.DataFrame, since: str | None) -> pd.DataFrame:
    if since is None or frame.empty:
        return frame
    return frame[(frame["ts"].str[:10] >= since) | (frame["ts_release"].str[:10] >= since)]


def select(conn: Any, settings: Settings, *, since: str | None = None,
           breadth_readings: list | None = None,
           newcomers: set[str] | frozenset[str] = frozenset()) -> Selection:
    """Everything the public panel reads, from the local database.

    ``newcomers``: researched CIKs the public copy does not have yet. Their rows travel
    whole whatever ``since`` says — an incremental run would otherwise publish a company
    added to the watchlist with only its last ten days (found 2026-09-25, adding four).
    """
    ciks = researched_ciks(settings)
    tickers = public_tickers(settings)

    fred = read_observations(conn, source="fred")
    sec = read_observations(conn, source="sec")
    sec = sec[sec["series_id"].str[:10].isin(ciks)] if not sec.empty else sec
    prices = read_observations(conn, series_ids=[f"{t}:close_raw" for t in tickers])
    short = read_observations(conn, source=FINRA_SOURCE,
                              series_ids=[f"{t}:{s}" for t in researched_tickers(settings)
                                          for s in FINRA_SERIES])
    frames = [fred, sec, prices, short]
    if breadth_readings:
        frames.append(br.readings_to_observations(breadth_readings))
    everything = (pd.concat([f for f in frames if not f.empty], ignore_index=True)
                  if any(not f.empty for f in frames)
                  else pd.DataFrame(columns=loader.OBSERVATION_COLUMNS))
    new_keys = set(newcomers) | {str(c["ticker"]) for c in settings.researched_companies
                                 if str(c.get("cik", "")).zfill(10) in newcomers}
    whole = everything["series_id"].str.split(":").str[0].isin(new_keys) \
        if not everything.empty else pd.Series(dtype=bool)
    observations = pd.concat([_recent(everything[~whole], since), everything[whole]],
                             ignore_index=True) if not everything.empty else everything

    companies = read_table(conn, "companies")
    filings = read_table(conn, "filings")
    events = read_table(conn, "events")
    actions = read_table(conn, "corporate_actions")
    tables = {
        "companies": companies[companies["cik"].isin(ciks)],
        "filings": filings[filings["cik"].isin(ciks)],
        "events": events[events["category"].isin(["macro", "fomc"])
                         | ((events["category"] == "earnings") & events["cik"].isin(ciks))],
        "corporate_actions": actions[actions["ticker"].isin(tickers)],
    }
    return Selection(observations=observations,
                     tables={k: _records(v) for k, v in tables.items()})


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows with SQL NULLs as ``None``. Through pandas a numeric NULL becomes ``NaN``, and in
    PostgreSQL ``NaN`` is a number, not an absence (section 12)."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def assert_public(selection: Selection, settings: Settings, held: set[str]) -> None:
    """Refuse to write if anything private slipped into the selection.

    A second, independent check on :func:`select`: a filter that one day starts letting
    through a held ticker or an account row must fail here, before the cloud sees it.
    """
    obs = selection.observations
    leaks = []
    if not obs.empty:
        bad_sources = set(obs["source"]) & PRIVATE_SOURCES
        if bad_sources:
            leaks.append(f"private sources {sorted(bad_sources)}")
        tickers = {sid.split(":", 1)[0] for sid in obs["series_id"]
                   if sid.endswith(":close_raw")}
        allowed = set(public_tickers(settings))
        if tickers - allowed:
            leaks.append(f"prices outside the public list {sorted(tickers - allowed)}")
        shorted = {sid.split(":", 1)[0] for sid in obs.loc[obs["source"] == FINRA_SOURCE,
                                                           "series_id"]}
        if shorted - set(researched_tickers(settings)):
            leaks.append("short interest of non-researched tickers "
                         f"{sorted(shorted - set(researched_tickers(settings)))}")
    ciks = researched_ciks(settings)
    for table in ("companies", "filings"):
        stray = {r["cik"] for r in selection.tables.get(table, [])} - ciks
        if stray:
            leaks.append(f"{table} for non-researched CIKs {sorted(stray)}")
    held_only = held - set(public_tickers(settings))
    listed = {r["ticker"] for r in selection.tables.get("companies", [])}
    if listed & held_only:
        leaks.append(f"held-only positions in companies {sorted(listed & held_only)}")
    if set(selection.tables) & PRIVATE_TABLES:
        leaks.append(f"private tables {sorted(set(selection.tables) & PRIVATE_TABLES)}")
    if leaks:
        raise RuntimeError("Public sync refused: " + "; ".join(leaks))


def write(target: Any, selection: Selection) -> dict[str, int]:
    """Upsert the selection into the public database (schema applied first). Idempotent."""
    loader.apply_schema(target)
    counts = {"observations": loader.upsert_observations(target, selection.observations)}
    writers = {"companies": loader.upsert_companies, "filings": loader.upsert_filings,
               "events": loader.upsert_events,
               "corporate_actions": loader.upsert_corporate_actions}
    for table in TABLES:
        counts[table] = writers[table](target, selection.tables.get(table, []))
    return counts


def constituent_readings(conn: Any, level2: Mapping[str, Any]) -> list:
    """The constituent breadth, computed from the local universe (same config as the page)."""
    intervals = read_table(conn, "universe_membership")
    if intervals.empty:
        return []
    tickers = sorted(set(intervals["ticker"]))
    closes = read_observations(conn, series_ids=[f"{t}:close_raw" for t in tickers])
    if closes.empty:
        return []
    cfg = level2["breadth"]
    return br.breadth_series(
        intervals, br.wide_closes(closes), br.splits_by_ticker(read_table(conn, "corporate_actions")),
        window=cfg["window"], threshold=cfg["coverage_threshold"],
        min_window_fraction=cfg["min_window_fraction"],
    )
