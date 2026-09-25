"""Daily orchestration: ingest, alerts, then the public copy (CLAUDE.md sections 3, 8).

What the systemd timer runs (``deploy/``). Two rules:

- **Alerts run even when the ingest fails.** A failed source already raises its own alert
  from ``run_ingest.py``, and the other rules (a macro release tomorrow, results of a
  position) still hold with yesterday's data. One broken source must not silence them.
- **Wait for the network first.** With ``Persistent=true`` a missed run starts as soon as
  the machine wakes, often before the network is up. Starting anyway would fail every
  source *and* the Telegram message that should report it — the one failure mode that
  reaches nobody. So it waits, and gives up loudly after ``orchestration.network_wait_s``.

The exit code is non-zero if either step failed, so ``systemctl --user status`` and the
journal show it. Output goes to the journal: ``journalctl --user -u equitydash-daily``.
"""

from __future__ import annotations

import socket
import sys
import time
from typing import Callable

import run_alerts
import run_ingest
import run_public_sync
from core.config import load_settings
from core.logging_setup import configure_logging, get_logger

log = get_logger(__name__)

# sysexits.h: a temporary failure, worth retrying later — not a bug.
EX_TEMPFAIL = 75


def wait_for_network(
    host: str,
    timeout_s: float,
    *,
    resolve: Callable[..., object] = socket.getaddrinfo,
    sleep: Callable[[float], None] = time.sleep,
    step_s: float = 10.0,
) -> bool:
    """True once ``host`` resolves; False after ``timeout_s`` without it."""
    waited = 0.0
    while True:
        try:
            resolve(host, 443)
            return True
        except OSError:
            if waited >= timeout_s:
                return False
            sleep(step_s)
            waited += step_s


def main(
    argv: list[str] | None = None,
    *,
    ingest: Callable[[list[str]], int] = run_ingest.main,
    alerts: Callable[[list[str]], int] = run_alerts.main,
    public_sync: Callable[[list[str]], int] = run_public_sync.main,
    network: Callable[[str, float], bool] = wait_for_network,
) -> int:
    settings = load_settings()
    configure_logging(settings.log_level, settings.secrets.values())
    cfg = settings.raw.get("orchestration", {})
    host = str(cfg.get("network_check_host", "api.stlouisfed.org"))
    wait = float(cfg.get("network_wait_s", 600))

    if not network(host, wait):
        log.error("No network after %.0f s (could not resolve %s); nothing was run.", wait, host)
        return EX_TEMPFAIL

    ingest_exit = ingest([])
    alerts_exit = alerts([])
    # Last, and only with its URL configured (it skips itself otherwise): the public copy
    # must never delay the alerts, and a failed upload must not look like a failed ingest.
    try:
        sync_exit = public_sync([])
    except Exception:  # noqa: BLE001 — the cloud being down is not the day's result
        log.exception("Public sync failed.")
        sync_exit = 1
    log.info("Daily run finished: ingest=%d alerts=%d public_sync=%d",
             ingest_exit, alerts_exit, sync_exit)
    return 0 if ingest_exit == alerts_exit == sync_exit == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
