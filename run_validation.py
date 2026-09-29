"""Validation report entry point (CLAUDE.md section 8, phase 4).

Runs the pre-registered battery of :mod:`validation.backtest` once and prints the report —
including, and first, what does not work. Re-running gives the same numbers: the shuffles
are seeded and the data is point-in-time.

Usage:
    python run_validation.py                    # print the report
    python run_validation.py --out informe.md   # also write it to a file
    python run_validation.py --insiders         # the insider-purchase study
    python run_validation.py --factors          # quality, accruals and momentum
    python run_validation.py --pead             # drift after earnings (§15.5 point 6)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from core.config import load_settings
from core.logging_setup import configure_logging, get_logger
from db.database import open_connection, read_observations, read_table
from transform import regime as rg
from validation.backtest import discrimination, report, run_battery

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="equitydash signal validation")
    parser.add_argument("--out", help="Also write the Markdown report to this file.")
    parser.add_argument("--brake", action="store_true",
                        help="Run the brake study (RESEARCH.md §2.30) instead of the battery.")
    parser.add_argument("--insiders", action="store_true",
                        help="Run the insider-purchase study (settings.yaml insider_study).")
    parser.add_argument("--factors", action="store_true",
                        help="Run the factor study (settings.yaml factor_study).")
    parser.add_argument("--pead", action="store_true",
                        help="Run the post-earnings drift study (settings.yaml pead_study).")
    parser.add_argument("--refresh", action="store_true",
                        help="With --factors: download the members' companyfacts again.")
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level, settings.secrets.values())
    if args.insiders:
        return insider_study(settings, args.out)
    if args.factors:
        return factor_study(settings, args.out, refresh=args.refresh)
    if args.pead:
        return pead_study(settings, args.out)
    level2 = settings.raw["panel"]["level2"]
    cfg = settings.raw.get("validation", {})

    conn = open_connection(settings.db_path)
    try:
        built = rg.build(
            read_observations(conn, source="fred"),
            read_observations(conn, series_ids=[f"{t}:close_raw"
                                                for t in rg.regime_tickers(level2)]),
            read_table(conn, "corporate_actions"), level2,
        )
    finally:
        conn.close()
    if built is None:
        log.error("No data for the regime light: run `python run_ingest.py --only fred prices`.")
        return 1

    if args.brake:
        from transform.regime import asof  # noqa: PLC0415
        from validation.brake import report_study, run_study  # noqa: PLC0415

        conn = open_connection(settings.db_path)
        try:
            dff = read_observations(conn, series_ids=["DFF"])
        finally:
            conn.close()
        cash_rate = asof(dff, "DFF", built.benchmark_total.dropna().index)["value"]
        text = report_study(run_study(built, cash_rate, settings.raw["brake_study"]),
                            date.today().isoformat())
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        return 0

    battery = run_battery(built, cfg)
    judged = built.frame.index[built.frame["verdict"] != rg.INSUFFICIENT]
    verdict_start = judged[0].date().isoformat() if len(judged) else None
    key = str(cfg.get("out_of_sample_component", "equal_weight"))
    in_sample = discrimination(built, key, start=verdict_start)
    oos = [discrimination(built, key, end=verdict_start)]
    battery.notes += [
        f"Dentro de muestra = la ventana con veredicto (desde {verdict_start}), la que miró "
        "la fase 3. Fuera de muestra = todo lo anterior en que el componente existe.",
        "La hipótesis de NFCI no se puede probar fuera de muestra: su historia point-in-time "
        "empieza en 2011 y lo anterior son valores revisados (look-ahead, §9.4).",
    ]
    text = report(battery, oos, in_sample, date.today().isoformat())
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


def insider_study(settings, out: str | None) -> int:
    """Download what is missing (cached per quarter), run the pre-registered study, report."""
    import pandas as pd  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from ingest import insiders as ingest_insiders  # noqa: PLC0415
    from validation import insiders as study  # noqa: PLC0415

    cfg = settings.raw["insider_study"]
    sec = settings.source("sec")
    agent = settings.secret(str(sec.get("user_agent_env", "SEC_USER_AGENT")))
    cache = Path(str(sec.get("cache_dir", ".cache/sec"))) / "form345"
    frames = []
    for quarter in ingest_insiders.quarters(cfg["first_quarter"], date.today()):
        frame = ingest_insiders.load_quarter(quarter, cfg["source_url"], agent, cache)
        if frame is None:
            log.info("Form 4 %s not published yet by the SEC.", quarter)
            continue
        frames.append(frame)
    purchases = pd.concat(frames, ignore_index=True)

    conn = open_connection(settings.db_path)
    try:
        intervals = read_table(conn, "universe_membership")
        tickers = sorted(set(intervals["ticker"])) + [str(cfg["benchmark"])]
        closes = read_observations(conn, series_ids=[f"{t}:close_raw" for t in tickers])
        actions = read_table(conn, "corporate_actions")
    finally:
        conn.close()
    outcome = study.study(purchases, closes, actions, intervals, cfg)
    text = study.report(outcome, cfg, date.today().isoformat())
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


def factor_study(settings, out: str | None, *, refresh: bool = False) -> int:
    """Download what is missing (companyfacts subsets, SIC codes), run the pre-registered
    study once, report."""
    import datetime as dt  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415
    import requests  # noqa: PLC0415

    from ingest import member_facts  # noqa: PLC0415
    from ingest.screen import sic_map  # noqa: PLC0415
    from ingest.sec_filings import normalize_ticker, ticker_map  # noqa: PLC0415
    from validation import factors as study  # noqa: PLC0415

    cfg = settings.raw["factor_study"]
    sec = settings.source("sec")
    agent = settings.secret(str(sec.get("user_agent_env", "SEC_USER_AGENT")))
    if not agent:
        log.error("SEC_USER_AGENT is not set: the SEC refuses anonymous requests.")
        return 1
    base_url = str(sec.get("base_url", "https://data.sec.gov")).rstrip("/")
    delay_s = 1.0 / float(sec.get("rate_limit_rps", 8) or 8)
    cache = Path(str(sec.get("cache_dir", ".cache/sec")))
    concepts = {m: sec["concepts"][m] for m in study.METRICS}
    tags = sorted({t for spec in concepts.values() for t in spec["tags"]})

    lookback = int(cfg["signals"]["momentum"]["lookback_days"])
    since = (pd.Timestamp(cfg["first_date"]) - pd.Timedelta(days=lookback)).date().isoformat()
    conn = open_connection(settings.db_path)
    try:
        intervals = read_table(conn, "universe_membership")
        touching = intervals[intervals["end_date"].isna()
                             | (intervals["end_date"].astype(str) > since)]
        tickers = sorted(set(touching["ticker"]))
        closes = read_observations(conn, series_ids=[f"{t}:close_raw" for t in tickers]
                                   + [f"{cfg['calendar_ticker']}:close_raw"])
        actions = read_table(conn, "corporate_actions")
    finally:
        conn.close()
    priced = sorted({s.rsplit(":", 1)[0] for s in closes["series_id"].unique()}
                    - {str(cfg["calendar_ticker"])})

    response = requests.get(str(sec.get("ticker_map_url")), headers={"User-Agent": agent},
                            timeout=60)
    response.raise_for_status()
    mapping = ticker_map(response.json())
    cik_of = {t: mapping[normalize_ticker(t)] for t in priced
              if normalize_ticker(t) in mapping}
    ciks = sorted(set(cik_of.values()))
    log.info("Factor study: %d priced members, %d with a CIK today.", len(priced), len(cik_of))

    failures: list[str] = []
    sics = {cik: row[0] for cik, row in sic_map(
        ciks, cache / "sic_map.json", base_url=base_url, user_agent=agent, delay_s=delay_s,
        failures=failures).items() if cik in set(ciks)}
    facts, fact_failures = member_facts.load_all(
        ciks, tags, cache / "member_facts", base_url=base_url, user_agent=agent,
        delay_s=delay_s, refresh=refresh)
    failures += fact_failures
    for failure in failures:
        log.warning("Left out of the study: %s", failure)

    outcome = study.study(facts, cik_of, sics, closes, actions, intervals, concepts,
                          sec["duration_windows"], cfg, dt.date.today().isoformat())
    text = study.report(outcome, cfg, dt.date.today().isoformat())
    if failures:
        text += f"\n- Empresas que no se pudieron descargar: {len(failures)} (fuera del estudio)."
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


def pead_study(settings, out: str | None) -> int:
    """The earnings history of every priced S&P 500 member (cached per company), then the
    pre-registered drift study, once."""
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import requests  # noqa: PLC0415

    from ingest.sec_filings import normalize_ticker, ticker_map  # noqa: PLC0415
    from validation import pead as study  # noqa: PLC0415

    cfg = settings.raw["pead_study"]
    sec = settings.source("sec")
    agent = settings.secret(str(sec.get("user_agent_env", "SEC_USER_AGENT")))
    if not agent:
        log.error("SEC_USER_AGENT is not set: the SEC refuses anonymous requests.")
        return 1
    base_url = str(sec.get("base_url", "https://data.sec.gov")).rstrip("/")
    delay_s = 1.0 / float(sec.get("rate_limit_rps", 8) or 8)
    cache = Path(str(sec.get("cache_dir", ".cache/sec"))) / "earnings_history"
    cache.mkdir(parents=True, exist_ok=True)

    conn = open_connection(settings.db_path)
    try:
        intervals = read_table(conn, "universe_membership")
        tickers = sorted(set(intervals["ticker"]))
        closes = read_observations(conn, series_ids=[f"{t}:close_raw" for t in tickers]
                                   + [f"{cfg['benchmark']}:close_raw"])
        actions = read_table(conn, "corporate_actions")
    finally:
        conn.close()
    priced = sorted({sid.rsplit(":", 1)[0] for sid in closes["series_id"].unique()}
                    - {str(cfg["benchmark"])})

    def get(url: str) -> dict:
        response = requests.get(url, headers={"User-Agent": agent}, timeout=90)
        response.raise_for_status()
        time.sleep(delay_s)
        return response.json()

    mapping = ticker_map(get(str(sec.get("ticker_map_url"))))
    # One ticker per company (GOOG and GOOGL would double every event).
    cik_of: dict[str, str] = {}
    for ticker in priced:
        cik = mapping.get(normalize_ticker(ticker))
        if cik and cik not in cik_of.values():
            cik_of[ticker] = cik
    log.info("PEAD study: %d priced members, %d with a CIK today.", len(priced), len(cik_of))

    dates_by_symbol, failures = {}, []
    for i, (ticker, cik) in enumerate(sorted(cik_of.items())):
        path = cache / f"{cik}.json"
        if path.exists():
            dates_by_symbol[ticker] = json.loads(path.read_text(encoding="utf-8"))["dates"]
            continue
        try:
            main = get(f"{base_url}/submissions/CIK{cik}.json")["filings"]
            pages = [main["recent"]]
            for extra in main.get("files", []):
                # Older pages only while they reach back into the study's window.
                if str(extra.get("filingTo", "")) >= str(cfg["first_date"]):
                    pages.append(get(f"{base_url}/submissions/{extra['name']}"))
            dates = study.earnings_dates_from(pages, str(cfg["event_item"]))
        except Exception as exc:  # noqa: BLE001 — one company must not sink the study
            failures.append(f"{ticker}: {exc}")
            continue
        path.write_text(json.dumps({"cik": cik, "dates": dates}), encoding="utf-8")
        dates_by_symbol[ticker] = dates
        if i % 100 == 99:
            log.info("PEAD study: earnings history of %d companies read.", i + 1)
    for failure in failures:
        log.warning("Left out of the study: %s", failure)

    outcome = study.study(dates_by_symbol, closes, actions, intervals, cfg)
    text = study.report(outcome, cfg, date.today().isoformat())
    if failures:
        text += f"\n- Empresas que no se pudieron descargar: {len(failures)} (fuera del estudio)."
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
