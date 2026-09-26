"""Signals in a company's filings (transform/filing_signals.py), on synthetic filings."""

from __future__ import annotations

import pandas as pd

from transform import filing_signals as fs


def row(form, filed, items=None, cik="1"):
    return {"accession": f"{form}-{filed}", "cik": cik, "form": form, "filed_date": filed,
            "items": items, "url": "u"}


def test_each_form_is_labelled_for_what_it_certainly_is():
    assert fs.classify("NT 10-K", None)[0][0] == fs.RED
    assert fs.classify("S-3ASR", None)[0][1] == "shelf"
    assert fs.classify("424B5", None)[0][1] == "offering"
    assert fs.classify("SCHEDULE 13D/A", None)[0][1] == "stake"
    assert fs.classify("8-K", "2.02,9.01") == [], "results and exhibits signal nothing"
    assert [k for _, k, _ in fs.classify("8-K", "4.02,5.02")] == ["item_4.02", "item_5.02"]
    assert fs.classify("10-Q", None) == []


def test_a_shelf_is_a_warning_only_for_a_company_burning_cash():
    found = fs.signals(pd.DataFrame([row("S-3", "2026-09-11"), row("424B5", "2026-09-18")]),
                       "1", "2026-01-01")
    assert len(fs.warnings(found, burns_cash=True)) == 2
    assert fs.warnings(found, burns_cash=False) == []
    assert fs.warnings(found, burns_cash=None) == [], "unknown burn promotes nothing"


def test_the_window_is_respected():
    found = fs.signals(pd.DataFrame([row("NT 10-Q", "2025-01-01"), row("NT 10-Q", "2026-09-01")]),
                       "1", "2026-01-01", "2026-12-31")
    assert list(found["date"]) == ["2026-09-01"]
