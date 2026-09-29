"""Thematic exposure (CLAUDE.md §15.5, point 10).

What is defended: the split of a return against SPY into the theme's part and the company's
own is exact (a product, never a difference); a horizon without a close near both ends is
unknown, not measured over a shorter span; a card joins its theme by ``thesis_category``;
a malformed config is rejected at load; and the ETFs the pages compare against are
downloaded.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core import config
from ingest.prices import PricesIngester
from transform import themes as tf

DAYS = pd.date_range("2025-01-01", "2026-01-01", freq="D")


def series(start: float, end: float) -> pd.Series:
    """A total-return index moving linearly from ``start`` to ``end`` over DAYS."""
    return pd.Series([start + (end - start) * i / (len(DAYS) - 1) for i in range(len(DAYS))],
                     index=DAYS)


def test_the_split_is_exact_and_compounded():
    company, theme, spy = series(100, 150), series(100, 120), series(100, 110)
    one = tf.split(company, theme, spy, "2026-01-01", {"12m": 365})[0]
    assert one.company == pytest.approx(0.5) and one.theme == pytest.approx(0.2)
    assert one.theme_vs_spy == pytest.approx(1.2 / 1.1 - 1)
    assert one.company_vs_theme == pytest.approx(1.5 / 1.2 - 1)
    assert (1 + one.theme_vs_spy) * (1 + one.company_vs_theme) == pytest.approx(
        1 + one.company_vs_spy), "the two parts multiply back to the whole"
    assert one.company_vs_spy != pytest.approx(0.5 - 0.1), "compounded, not a difference"


def test_a_horizon_without_its_anchor_is_unknown():
    young = series(100, 150)[DAYS >= "2025-09-01"]
    assert tf.horizon_return(young, "2026-01-01", 365) is None, "listed 4 months ago"
    assert tf.horizon_return(young, "2026-01-01", 91) == pytest.approx(
        young.iloc[-1] / young[young.index <= "2025-10-02"].iloc[-1] - 1)
    # One close a month before the 12-month target, then a hole until September: the only
    # anchor is 31 days off, so the 12-month return is unknown, not a 13-month one.
    holed = pd.concat([pd.Series([90.0], index=[pd.Timestamp("2024-12-01")]), young])
    assert tf.horizon_return(holed, "2026-01-01", 365) is None
    stale = series(100, 150)[DAYS <= "2025-11-01"]
    assert tf.horizon_return(stale, "2026-01-01", 91) is None, "no close near today"


def test_a_card_joins_its_theme_by_category_and_tickers_may_repeat():
    raw = [{"name": "Biotecnología", "etf": "xbi", "tickers": ["aaa"]},
           {"name": "Brasil", "etf": "ETF1", "tickers": ["DDD"]},
           {"name": "Fintech", "etf": "ETF2", "tickers": ["DDD"]}]
    cards = [{"ticker": "BBB", "thesis_category": "biotecnología"}]
    themes = tf.parse(raw, cards)
    assert themes[0].etf == "XBI" and themes[0].tickers == ("AAA", "BBB")
    assert [t.name for t in tf.themes_of("DDD", themes)] == ["Brasil", "Fintech"]


def test_malformed_themes_are_named():
    assert tf.problems(None) == []
    assert any("tickers must be a list" in p
               for p in tf.problems([{"name": "B", "etf": "XBI", "tickers": "AAA"}]))
    assert any("repeated" in p for p in tf.problems([{"name": "B", "etf": "XBI"},
                                                     {"name": "b", "etf": "IBB"}]))
    assert any("not a member" in p
               for p in tf.problems([{"name": "B", "etf": "XBI", "tickers": ["XBI"]}]))
    assert any("etf is required" in p for p in tf.problems([{"name": "B"}]))


def test_the_theme_table_weighs_only_what_is_held():
    themes = tf.parse([{"name": "Bio", "etf": "XBI", "tickers": ["AAA", "BBB"]}])
    table = tf.theme_table(themes, {"XBI": series(100, 120), "SPY": series(100, 110)},
                           "2026-01-01", {"AAA": 0.1, "CCC": 0.2})
    row = table.iloc[0]
    assert row["held"] == "AAA" and row["weight"] == pytest.approx(0.1)
    assert row["vs_spy_12m"] == pytest.approx(1.2 / 1.1 - 1)
    assert pd.notna(row["vs_spy_3m"]), "a linear index has a 3-month return"


def test_loading_rejects_bad_themes_and_prices_download_their_etfs(tmp_path, monkeypatch):
    local = tmp_path / "settings.local.yaml"
    local.write_text("universe:\n  themes:\n    - {name: Bio, etf: XBI, tickers: AAA}\n",
                     encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()
    with pytest.raises(ValueError, match="tickers must be a list"):
        config.load_settings()
    local.write_text("universe:\n  themes:\n    - {name: Bio, etf: XBI, tickers: [AAA]}\n"
                     "  watchlist:\n    - {ticker: AAA, cik: \"0000000001\", benchmark: IBB}\n",
                     encoding="utf-8")
    config.load_settings.cache_clear()
    tickers = PricesIngester.tickers_for(config.load_settings())
    assert "XBI" in tickers and "IBB" in tickers, "theme ETF and card benchmark"
    config.load_settings.cache_clear()
