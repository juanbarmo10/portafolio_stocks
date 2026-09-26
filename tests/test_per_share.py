"""Growth of the business against growth per share (transform/per_share.py), synthetic."""

from __future__ import annotations

import pandas as pd
import pytest

from transform import per_share as ps

CIK = "0000000001"
QUARTERS = pd.date_range("2019-03-31", "2026-06-30", freq="QE")


def fact(metric, end, value, period):
    return {"source": "sec", "series_id": f"{CIK}:{metric}:{period}", "ts": end,
            "ts_release": end, "value": value}


def company(shares_of, revenue_of=lambda i: 100.0 * 1.2 ** (i / 4), income_of=lambda i: 5.0,
            basic=True, diluted_of=None):
    """Quarterly revenue and income, and the share counts per quarter (Q4 as a fiscal year:
    no company files a Q4 10-Q, §9.12)."""
    rows = []
    for i, day in enumerate(QUARTERS):
        end = day.date().isoformat()
        period = "fy" if day.month == 12 else "q"
        rows.append(fact("revenue", end, revenue_of(i), "q"))
        rows.append(fact("net_income", end, income_of(i), "q"))
        if period == "fy":
            rows.append(fact("revenue", end, sum(revenue_of(j) for j in range(i - 3, i + 1)),
                             "fy"))
            rows.append(fact("net_income", end, sum(income_of(j) for j in range(i - 3, i + 1)),
                             "fy"))
        if basic:
            rows.append(fact("basic_shares", end, shares_of(i), period))
        if diluted_of is not None:
            rows.append(fact("diluted_shares", end, diluted_of(i), period))
    return pd.DataFrame(rows)


def test_per_share_growth_is_total_growth_net_of_dilution():
    """Revenue +20 %/yr compounding quarterly, shares +5 %/yr: per share ≈ 1,2/1,05 − 1."""
    obs = company(lambda i: 100.0 * 1.05 ** (i / 4))
    result = ps.assess(obs, CIK, "2026-09-30").table.set_index(["metric", "years"])
    row = result.loc[("revenue", 3)]
    assert row["total"] == pytest.approx(0.20, abs=0.01)
    assert row["per_share"] == pytest.approx(1.20 / 1.05 - 1, abs=0.01)
    assert row["gap"] == pytest.approx(row["total"] - row["per_share"])


def test_an_ipo_jump_is_not_dilution():
    """The count doubles in one quarter (preferred shares converting at the listing)."""
    obs = company(lambda i: 100.0 if i < 14 else 250.0)          # jump in 2022-Q3
    result = ps.assess(obs, CIK, "2026-09-30")
    assert result.jumps[5] and pd.isna(result.table.set_index(["metric", "years"])
                                       .loc[("revenue", 5), "per_share"])
    assert result.shares[5] is None
    assert not result.jumps[3] and result.shares[3] == pytest.approx(0.0, abs=1e-9)


def test_a_stale_basic_series_falls_back_to_the_diluted_one():
    """A company that stopped filing the basic count (by share class since 2023)."""
    obs = company(lambda i: 100.0, diluted_of=lambda i: 110.0)
    obs = obs[~((obs["series_id"].str.contains("basic_shares")) & (obs["ts"] > "2023-01-01"))]
    assert ps.share_basis(obs, CIK, "2026-09-30")[0] == "diluted_shares"


def test_on_the_diluted_count_a_loss_to_profit_window_is_not_compared():
    """ASC 260: in a loss period diluted = basic, so crossing into profit adds the options
    to the count without a single new share."""
    obs = company(lambda i: 100.0, basic=False,
                  income_of=lambda i: -5.0 if i < 14 else 5.0,        # profit from 2022-Q3
                  diluted_of=lambda i: 100.0 if i < 14 else 115.0)
    result = ps.assess(obs, CIK, "2026-09-30")
    assert result.basis == "diluted_shares"
    assert result.flips[5] and result.shares[5] is None
    assert not result.flips[3], "both ends of the 3-year window are profitable"


def test_growth_pairs_by_date_and_refuses_a_negative_start():
    series = pd.Series([-1.0, 2.0, 4.0],
                       index=pd.to_datetime(["2023-06-30", "2024-06-30", "2026-06-30"]))
    assert ps.growth(series, 3) is None, "a negative start leaves the rate undefined"
    assert ps.growth(series, 2) == pytest.approx(2 ** 0.5 - 1, abs=0.01)
    assert ps.growth(series, 5) is None, "no point five years back"
