"""Price-ingester tests — CI-BLOCKING (CLAUDE.md sections 9.1, 10).

This file is the guard for the adjusted-price trap. It runs against a frozen capture of
yfinance's response around AAPL's 4:1 split (ex-date 2020-08-31), needs no network and no
yfinance install, and asserts that the series equitydash stores is the raw one — the one
that does not change when the future happens.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from db import loader
from db.loader import OBSERVATION_COLUMNS
from ingest.prices import (
    PricesIngester,
    build_corporate_actions,
    build_observations,
    split_factor,
    unadjust_history,
)

FIXTURES = Path(__file__).parent / "fixtures"

# AAPL's only split inside the captured window.
AAPL_SPLITS = {dt.date(2020, 8, 31): 4.0}

# Independently documented facts, not values derived from this code:
#   AAPL closed at 499.23 on 2020-08-28, the last session before its 4:1 split,
#   and at 129.04 on 2020-08-31, the first session quoted post-split.
REAL_CLOSE_BEFORE_SPLIT = 499.23
REAL_CLOSE_ON_EX_DATE = 129.04


@pytest.fixture()
def history() -> pd.DataFrame:
    """Frozen yfinance response: split-adjusted, as the library actually serves it."""
    df = pd.read_csv(FIXTURES / "yfinance_aapl_split_2020.csv", index_col="date")
    df.index = [dt.date.fromisoformat(d) for d in df.index]
    return df


# --- The core guarantee -------------------------------------------------------


def test_reconstructed_close_matches_the_real_historical_close(history):
    """The headline assertion of section 9.1.

    yfinance reports 124.81 for 2020-08-28 even with auto_adjust=False. The price the
    market actually printed that day was 499.23. Storing 124.81 as `close_raw` would be
    a plausible, well-formed, wrong number.
    """
    raw = unadjust_history(history, AAPL_SPLITS)
    assert history.loc[dt.date(2020, 8, 28), "Close"] == pytest.approx(124.81, abs=0.01)
    assert raw.loc[dt.date(2020, 8, 28), "Close"] == pytest.approx(
        REAL_CLOSE_BEFORE_SPLIT, abs=0.01
    )


def test_ex_date_bar_is_already_post_split_and_must_not_be_rescaled(history):
    """The off-by-one that silently multiplies one bar by the split ratio.

    Comparing raw timestamps instead of calendar dates puts the split of the ex-date into
    its own factor, turning 129.04 into 516.16 — a 4x spike on a single day.
    """
    raw = unadjust_history(history, AAPL_SPLITS)
    assert raw.loc[dt.date(2020, 8, 31), "Close"] == pytest.approx(
        REAL_CLOSE_ON_EX_DATE, abs=0.01
    )
    assert split_factor(dt.date(2020, 8, 31), AAPL_SPLITS) == 1.0
    assert split_factor(dt.date(2020, 8, 30), AAPL_SPLITS) == 4.0


def test_stored_series_does_not_change_when_a_future_split_happens(history):
    """The property that makes the stored history trustworthy.

    A split-adjusted series is retroactively mutable: the day AAPL splits again, every
    past value Yahoo serves is rescaled and any backtest run before that day stops
    reproducing. The raw series must be invariant. Simulated here by rescaling the whole
    frame as a future 2:1 split would, then re-running the reconstruction.
    """
    raw_today = unadjust_history(history, AAPL_SPLITS)

    future_ex = dt.date(2027, 1, 15)
    rescaled = history.copy()
    for column in ("Open", "High", "Low", "Close"):
        rescaled[column] = rescaled[column] / 2.0
    rescaled["Volume"] = rescaled["Volume"] * 2.0
    raw_later = unadjust_history(rescaled, {**AAPL_SPLITS, future_ex: 2.0})

    for column in ("Open", "High", "Low", "Close", "Volume"):
        pd.testing.assert_series_equal(
            raw_today[column], raw_later[column], check_names=False, rtol=1e-9
        )


# --- Volume and dividends follow different conventions ------------------------


def test_dollar_volume_is_invariant_under_reconstruction(history):
    """Non-circular check that volume is scaled the *opposite* way to price.

    Yahoo preserves price x volume across its adjustment, so the reconstruction must too.
    Multiplying volume by the factor instead of dividing would quadruple dollar volume on
    every pre-split bar without breaking any other assertion here.
    """
    raw = unadjust_history(history, AAPL_SPLITS)
    adjusted_dollars = history["Close"] * history["Volume"]
    raw_dollars = raw["Close"] * raw["Volume"]
    pd.testing.assert_series_equal(
        adjusted_dollars, raw_dollars, check_names=False, rtol=1e-9
    )


def test_pre_split_raw_volume_is_smaller_than_reported(history):
    """Fewer shares changed hands pre-split than Yahoo's comparable figure suggests."""
    raw = unadjust_history(history, AAPL_SPLITS)
    day = dt.date(2020, 8, 28)
    assert raw.loc[day, "Volume"] == pytest.approx(history.loc[day, "Volume"] / 4.0)
    # On and after the ex-date nothing is rescaled.
    ex = dt.date(2020, 8, 31)
    assert raw.loc[ex, "Volume"] == pytest.approx(history.loc[ex, "Volume"])


def test_dividend_is_restored_to_the_amount_actually_paid():
    """Yahoo divides historical dividends by later splits too.

    AAPL's payout with ex-date 2020-08-07 is reported as 0.205; shareholders were paid
    0.82 per share. Storing the reported figure would understate every pre-split payout
    and quietly corrupt total-return and dividend-yield calculations.
    """
    div = pd.read_csv(FIXTURES / "yfinance_aapl_dividend_presplit_2020.csv", index_col="date")
    ex_date = dt.date.fromisoformat(div.index[0])
    reported = float(div.iloc[0]["Dividends"])
    assert reported == pytest.approx(0.205, abs=1e-6)

    rows = build_corporate_actions("AAPL", AAPL_SPLITS, {ex_date: reported})
    dividend = next(r for r in rows if r["kind"] == "dividend")
    assert dividend["amount"] == pytest.approx(0.82, abs=1e-6)


# --- Contract and idempotency -------------------------------------------------


def test_observations_honor_the_loader_contract(history):
    raw = unadjust_history(history, AAPL_SPLITS)
    df = build_observations("AAPL", raw)
    assert list(df.columns) == OBSERVATION_COLUMNS
    assert (df["source"] == "yfinance").all()
    assert set(df["series_id"].unique()) == {
        "AAPL:open_raw", "AAPL:high_raw", "AAPL:low_raw", "AAPL:close_raw", "AAPL:volume_raw",
    }


def test_release_date_equals_reference_date_for_daily_bars(history):
    """A daily close is public the day it happens: no publication lag to model."""
    df = build_observations("AAPL", unadjust_history(history, AAPL_SPLITS))
    assert (df["ts"] == df["ts_release"]).all()
    assert df["ts"].str.endswith("+00:00").all()


def test_split_row_records_the_ratio_and_leaves_cik_null(history):
    rows = build_corporate_actions("AAPL", AAPL_SPLITS, {})
    assert len(rows) == 1
    split = rows[0]
    assert split["kind"] == "split"
    assert split["ratio"] == 4.0
    assert split["ticker"] == "AAPL"
    # Never guessed from the ticker (section 9.3); resolved later or left visible as NULL.
    assert split["cik"] is None
    assert split["ex_date"].startswith("2020-08-31")


def test_corporate_actions_upsert_idempotently(tmp_path, history):
    """Deterministic action_id: re-running the pipeline must not duplicate."""
    rows = build_corporate_actions("AAPL", AAPL_SPLITS, {dt.date(2020, 8, 7): 0.205})
    conn = loader.init_db(tmp_path / "t.db")
    try:
        for _ in range(3):
            loader.upsert_corporate_actions(conn, rows)
        assert conn.execute("SELECT COUNT(*) FROM corporate_actions").fetchone()[0] == 2
    finally:
        conn.close()


def test_observations_upsert_idempotently(tmp_path, history):
    df = build_observations("AAPL", unadjust_history(history, AAPL_SPLITS))
    conn = loader.init_db(tmp_path / "t.db")
    try:
        loader.upsert_observations(conn, df)
        first = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        loader.upsert_observations(conn, df)
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == first
        assert first == len(df)
    finally:
        conn.close()


# --- Unit-level edge cases ----------------------------------------------------


def test_split_factor_with_no_splits():
    assert split_factor(dt.date(2020, 1, 1), {}) == 1.0


def test_split_factor_compounds_multiple_splits():
    """AAPL pre-1987 needs every later split applied: 2 x 2 x 2 x 7 x 4 = 224."""
    splits = {
        dt.date(1987, 6, 16): 2.0, dt.date(2000, 6, 21): 2.0, dt.date(2005, 2, 28): 2.0,
        dt.date(2014, 6, 9): 7.0, dt.date(2020, 8, 31): 4.0,
    }
    assert split_factor(dt.date(1987, 1, 1), splits) == pytest.approx(224.0)
    assert split_factor(dt.date(2014, 6, 9), splits) == pytest.approx(4.0)
    assert split_factor(dt.date(2020, 8, 31), splits) == 1.0


def test_empty_history_returns_empty_without_raising():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    assert unadjust_history(empty, AAPL_SPLITS).empty
    assert build_observations("AAPL", unadjust_history(empty, AAPL_SPLITS)).empty


def test_zero_ratio_split_is_ignored():
    """yfinance uses 0.0 for "no action on this bar"; it must never zero a price."""
    assert split_factor(dt.date(2020, 1, 1), {dt.date(2020, 6, 1): 0.0}) == 1.0


def _settings():
    """Real settings.yaml, so the configured ticker list is the one the project ships."""
    from core.config import load_settings

    return load_settings()


def test_held_tickers_join_the_download_list(tmp_path):
    """A position outside the written universe must still get a price (section 12).

    Without this the reconciliation against IBKR's NAV cannot run at all for a holding
    whose thesis card has not been written yet — and refusing to value what you own is not
    the same discipline as refusing to invent a number.
    """
    import pandas as pd

    from db import loader

    ingester = PricesIngester(_settings())
    before = list(ingester._tickers)
    assert "TMUS" not in before

    conn = loader.init_db(tmp_path / "held.db")
    try:
        loader.upsert_observations(conn, pd.DataFrame([
            {"source": "ibkr", "series_id": "TMUS:position_qty", "ts": "2026-09-15",
             "ts_release": "2026-09-15", "value": 1.0},
        ]))
        ingester.attach_database(conn)
    finally:
        conn.close()

    assert "TMUS" in ingester._tickers
    assert before == ingester._tickers[:len(before)], "the configured list was reordered"


def test_attaching_an_empty_database_changes_nothing(tmp_path):
    from db import loader

    ingester = PricesIngester(_settings())
    before = list(ingester._tickers)
    conn = loader.init_db(tmp_path / "empty.db")
    try:
        ingester.attach_database(conn)
    finally:
        conn.close()
    assert ingester._tickers == before
