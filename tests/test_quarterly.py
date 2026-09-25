"""Quarter by quarter and the balance sheet (CLAUDE.md §15.1.3), on synthetic filings whose
answers are known in advance."""

from __future__ import annotations

import pandas as pd
import pytest

from transform import fundamentals as fun
from transform import quarterly as qt

CIK = "0000000001"


def fact(metric: str, end: str, value: float, *, period: str = "q",
         filed: str | None = None) -> dict:
    series = f"{CIK}:{metric}" + (f":{period}" if period else "")
    return {"source": "sec", "series_id": series, "ts": end,
            "ts_release": filed or end, "value": value}


QUARTERS = ["2024-03-31", "2024-06-30", "2024-09-30", "2025-03-31", "2025-06-30",
            "2025-09-30"]


def filings() -> pd.DataFrame:
    rows = []
    for i, end in enumerate(QUARTERS):
        revenue = 100.0 + 10 * i
        rows += [fact("revenue", end, revenue), fact("gross_profit", end, revenue * 0.6),
                 fact("operating_income", end, revenue * 0.2)]
    # Q4s come from the 10-K, as nobody files a Q4 10-Q (§9.12).
    rows += [fact("revenue", "2024-12-31", 100 + 110 + 120 + 150.0, period="fy"),
             fact("gross_profit", "2024-12-31", 0.6 * 480, period="fy")]
    return pd.DataFrame(rows)


def test_quarters_include_the_derived_fourth_and_grow_year_over_year():
    table = qt.quarter_table(filings(), CIK, "2025-12-31")
    assert "2024-12-31" in set(table["quarter_end"]), "the Q4 derived from the 10-K"
    q4 = table[table["quarter_end"] == "2024-12-31"].iloc[0]
    assert q4["revenue"] == pytest.approx(150.0)
    last = table.iloc[-1]
    assert last["quarter_end"] == "2025-09-30"
    assert last["revenue_growth"] == pytest.approx(150 / 120 - 1), "against Q3 a year before"
    assert last["gross_margin"] == pytest.approx(0.6)
    assert pd.isna(last["rd_share"]), "not filed is None, never zero"


def test_a_quarter_filed_later_is_not_in_an_earlier_table():
    rows = filings()
    rows.loc[rows["ts"] == "2025-09-30", "ts_release"] = "2025-11-05"
    table = qt.quarter_table(rows, CIK, "2025-10-31")
    assert "2025-09-30" not in set(table["quarter_end"])


def test_a_balance_line_missing_from_the_latest_sheet_is_not_carried_forward():
    """HIMS last tagged DebtCurrent in 2020: a stale figure must not sit next to today's."""
    rows = pd.DataFrame([
        fact("assets", "2020-12-31", 100.0, period=""), fact("assets", "2026-06-30", 900.0, period=""),
        fact("debt_current", "2020-12-31", 50.0, period=""),
        fact("long_term_debt", "2026-06-30", 300.0, period=""),
        fact("cash", "2026-06-30", 120.0, period=""),
        fact("equity", "2026-06-30", 200.0, period=""),
    ])
    assert fun.latest_instant(rows, CIK, "debt_current", "2026-09-01") is None
    assert fun.latest_instant(rows, CIK, "debt_current", "2021-01-01") == 50.0, \
        "on the 2020 sheet it was there"
    sheet = qt.balance(rows, CIK, "2026-09-01")
    assert sheet.balance_date == "2026-06-30"
    assert sheet.debt == 300.0 and sheet.net_cash == pytest.approx(120.0 - 300.0)
    assert sheet.interest_coverage is None, "no interest filed"
