"""Upload the public part of the local database to the public deployment (section 8, phase 5).

Usage:
    python run_public_sync.py           # incremental: rows since the last upload (+ margin)
    python run_public_sync.py --full    # everything (first run)
    python run_public_sync.py --dry-run # count what would be sent, send nothing

Needs ``PUBLIC_DATABASE_URL`` in config/.env. Without it: skipped, exit 0 — the public copy
is optional. What is and is not copied: ``db/public_sync.py``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from core.config import load_settings
from core.logging_setup import configure_logging, get_logger
from db import public_sync as ps
from db.database import connect_url, open_connection, read_table
from ingest.base import held_tickers

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="equitydash public sync")
    parser.add_argument("--full", action="store_true", help="Copy everything, not only recent rows.")
    parser.add_argument("--dry-run", action="store_true", help="Count, do not write.")
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level, settings.secrets.values())
    url = settings.secret("PUBLIC_DATABASE_URL")
    if not url and not args.dry_run:
        log.warning("PUBLIC_DATABASE_URL is not set; public sync skipped.")
        return 0
    margin = int(settings.raw.get("public_sync", {}).get("lookback_days", 10))
    since: str | None = None
    if not args.full and not url:
        # A dry run with no target to ask: the margin before today, only to count.
        since = ps.window_start(pd.Timestamp.now(tz="UTC").isoformat(), margin)

    # Researched companies the cloud does not list yet go up whole (ps.select).
    newcomers: set[str] = set()
    published_series: set[str] | None = None
    if not args.full and url:
        # One connection per question: on PostgreSQL a failed query (a first run has no
        # tables) aborts the transaction, and the next question would fail with it.
        last = _ask(url, ps.last_upload, None)
        since = ps.window_start(last, margin)
        if since is None:
            log.info("Public sync: the copy has no observations yet — sending everything.")
        else:
            log.info("Public sync: last complete upload %s; sending rows since %s.",
                     last, since)
        published = _ask(url, lambda t: set(read_table(t, "companies")["cik"]), set())
        newcomers = ps.researched_ciks(settings) - published
        published_series = _ask(url, lambda t: {str(r[0]) for r in t.execute(
            "SELECT DISTINCT series_id FROM observations").fetchall()}, set())
        if newcomers:
            log.info("Public sync: %d new researched compan(ies) sent whole.", len(newcomers))

    local = open_connection(settings.db_path)
    try:
        readings = ps.constituent_readings(local, settings.raw["panel"]["level2"])
        selection = ps.select(local, settings, since=since, breadth_readings=readings,
                              newcomers=newcomers, published_series=published_series)
        ps.assert_public(selection, settings, held=set(held_tickers(local)))
    finally:
        local.close()

    sizes = {"observations": len(selection.observations),
             **{k: len(v) for k, v in selection.tables.items()}}
    scope = "full" if since is None else f"since {since}"
    if args.dry_run:
        log.info("Dry run (%s): would send %s", scope, sizes)
        return 0

    target = connect_url(url)
    try:
        counts = ps.write(target, selection)
    finally:
        target.close()
    log.info("Public sync (%s) done: %s", scope, counts)
    return 0


def _ask(url: str, question, default):
    """Run one read against the public copy; ``default`` when it fails (a first run has no
    tables). Every default errs toward sending more — never less."""
    target = connect_url(url)
    try:
        return question(target)
    except Exception:  # noqa: BLE001 — a first run has no tables: nothing is published yet
        return default
    finally:
        target.close()


if __name__ == "__main__":
    sys.exit(main())
