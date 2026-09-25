"""Daily orchestration (section 8, phase 5): the order, the exit code, and the network wait."""

from __future__ import annotations

import pathlib

import run_daily

REPO = pathlib.Path(__file__).resolve().parents[1]


def run(ingest_exit, alerts_exit, network=True, sync_exit=0):
    calls = []

    def ingest(argv):
        calls.append("ingest")
        return ingest_exit

    def alerts(argv):
        calls.append("alerts")
        return alerts_exit

    def sync(argv):
        calls.append("sync")
        if isinstance(sync_exit, Exception):
            raise sync_exit
        return sync_exit

    code = run_daily.main([], ingest=ingest, alerts=alerts, public_sync=sync,
                          network=lambda host, wait: network)
    return code, calls


def test_ingest_then_alerts_and_zero_when_both_succeed():
    assert run(0, 0) == (0, ["ingest", "alerts", "sync"])


def test_alerts_still_run_when_the_ingest_fails():
    """A failed source must not silence tomorrow's CPI warning."""
    assert run(1, 0) == (1, ["ingest", "alerts", "sync"])


def test_a_failed_alert_run_reaches_the_exit_code():
    assert run(0, 1)[0] == 1


def test_without_network_nothing_runs_and_the_code_says_try_later():
    assert run(0, 0, network=False) == (run_daily.EX_TEMPFAIL, [])


def test_the_wait_gives_up_after_its_timeout():
    slept = []

    def never(host, port):
        raise OSError("no route")

    ok = run_daily.wait_for_network("x", 30, resolve=never, sleep=slept.append, step_s=10)
    assert ok is False and sum(slept) == 30


def test_the_wait_returns_as_soon_as_the_host_resolves():
    attempts = iter([OSError("down"), None])

    def flaky(host, port):
        outcome = next(attempts)
        if outcome:
            raise outcome

    slept = []
    assert run_daily.wait_for_network("x", 600, resolve=flaky, sleep=slept.append, step_s=10)
    assert slept == [10]


def test_the_unit_template_runs_this_script_from_the_repo_venv():
    """The installer only substitutes @REPO@; the rest must already point at run_daily.py."""
    unit = (REPO / "deploy" / "equitydash-daily.service.in").read_text(encoding="utf-8")
    assert "ExecStart=@REPO@/.venv/bin/python @REPO@/run_daily.py" in unit
    assert "WorkingDirectory=@REPO@" in unit
    assert "/home/" not in unit, "no personal path in a tracked file"
    timer = (REPO / "deploy" / "equitydash-daily.timer").read_text(encoding="utf-8")
    assert "Persistent=true" in timer


def test_the_public_copy_goes_last_and_its_failure_is_reported():
    """The cloud being down must neither delay the alerts nor crash the run."""
    assert run(0, 0, sync_exit=RuntimeError("neon down")) == (1, ["ingest", "alerts", "sync"])
