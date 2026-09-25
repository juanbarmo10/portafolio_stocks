"""Upload the public part of the local database to the public deployment (section 8, phase 5).

Usage:
    python run_public_sync.py           # incremental: rows touched in the last days
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
    cfg = settings.raw.get("public_sync", {})
    since = None if args.full else (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(cfg.get("lookback_days", 10)))
    ).date().isoformat()

    # Researched companies the cloud does not list yet go up whole (ps.select).
    newcomers: set[str] = set()
    if since is not None and url:
        target = connect_url(url)
        try:
            published = set(read_table(target, "companies")["cik"])
        except Exception:  # noqa: BLE001 — a first run has no table: everything is new
            published = set()
        finally:
            target.close()
        newcomers = ps.researched_ciks(settings) - published
        if newcomers:
            log.info("Public sync: %d new researched compan(ies) sent whole.", len(newcomers))

    local = open_connection(settings.db_path)
    try:
        readings = ps.constituent_readings(local, settings.raw["panel"]["level2"])
        selection = ps.select(local, settings, since=since, breadth_readings=readings,
                              newcomers=newcomers)
        ps.assert_public(selection, settings, held=set(held_tickers(local)))
    finally:
        local.close()

    sizes = {"observations": len(selection.observations),
             **{k: len(v) for k, v in selection.tables.items()}}
    if args.dry_run:
        log.info("Dry run (%s): would send %s", "full" if args.full else f"since {since}", sizes)
        return 0

    target = connect_url(url)
    try:
        counts = ps.write(target, selection)
    finally:
        target.close()
    log.info("Public sync (%s) done: %s", "full" if args.full else f"since {since}", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
