"""Alerts entry point (CLAUDE.md section 8, phase 4). Run after ``run_ingest.py``.

Evaluates the rules in ``alerts.rules`` against the database, sends to Telegram what has
not been delivered yet, and records every alert in ``alerts_log``. Without Telegram
configured the alerts are logged and recorded as not delivered, so they go out — if still
true — on the first run after the bot is set up.

Usage:
    python run_alerts.py                 # evaluate, send, record
    python run_alerts.py --dry-run       # evaluate and print; send nothing, record nothing
    python run_alerts.py --test-message  # send one test message to check the bot setup

Reads: the database and config. Writes: ``alerts_log``, ``exit_ladder``; Telegram.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from alerts.rules import dispatch, evaluate, exit_ladder_rows, load_snapshot
from alerts.telegram import TelegramSender
from core.config import load_settings
from core.logging_setup import configure_logging, get_logger
from db import loader

log = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="equitydash alerts")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Evaluate and print the alerts; send and record nothing.")
    mode.add_argument("--test-message", action="store_true",
                      help="Send one test message and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level, settings.secrets.values())
    sender = TelegramSender(settings)

    if args.test_message:
        if not sender.enabled:
            log.error("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID are not set in config/.env.")
            return 1
        ok = sender.send("✅ equitydash: mensaje de prueba. Las alertas llegarán a este chat.")
        return 0 if ok else 1

    if not args.dry_run and not sender.enabled:
        log.warning("Telegram not configured: alerts are logged and recorded as undelivered.")

    conn = loader.init_db(settings.db_path)
    try:
        now = pd.Timestamp.now(tz="UTC")
        if not args.dry_run:
            loader.upsert_exit_ladder(conn, exit_ladder_rows(settings.tracked_companies,
                                                             now.isoformat()))
        snapshot = load_snapshot(conn, settings, now)
        alerts, failures = evaluate(snapshot, settings.raw.get("alerts", {}).get("rules", []))

        if args.dry_run:
            for alert in alerts:
                print(f"--- {alert.key}\n{alert.text}\n")
            print(f"{len(alerts)} alerta(s) activa(s); nada enviado ni registrado (--dry-run).")
        else:
            sent = dispatch(conn, alerts, sender)
            log.info("Alerts: %d active, %d newly sent.", len(alerts), sent)

        if failures:
            log.error("Alert rules failed: %s", failures)
            return 1
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
