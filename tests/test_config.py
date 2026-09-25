"""Configuration tests (CLAUDE.md sections 5.1, 5.2, 10)."""

from __future__ import annotations

import pytest
import yaml

from core import config


def test_settings_yaml_parses_and_declares_market_references():
    """The tracked settings.yaml must load and carry the level-1/2 references (section 5.1)."""
    config.load_settings.cache_clear()
    settings = config.load_settings()
    refs = settings.market_references
    # RSP and SPY together are what expose a narrow rally; the sector SPDRs drive
    # the defensive-rotation read. Their absence would quietly gut level 2.
    assert {"SPY", "RSP", "QQQ", "IWM"} <= set(refs)
    assert {"XLK", "XLU", "XLP", "XLF", "XLE", "XLV", "XLY", "XLI"} <= set(refs)
    assert len(refs) == len(set(refs)), "market references must not repeat"
    assert settings.db_path.is_absolute()


def test_settings_yaml_has_no_secrets():
    """Secrets live in .env, never in the tracked YAML (section 10)."""
    raw = yaml.safe_load(config.SETTINGS_PATH.read_text(encoding="utf-8"))
    text = yaml.safe_dump(raw).lower()
    for forbidden in ("api_key:", "token:", "secret:", "chat_id:"):
        assert forbidden not in text, f"settings.yaml appears to carry a secret: {forbidden}"


def test_fred_series_cover_the_level_1_checklist():
    """The macro series behind "is there risk appetite?" must be configured (section 2)."""
    config.load_settings.cache_clear()
    series = set(config.load_settings().source("fred").get("series", []))
    # BAMLH0A0HYM2 is the high-yield spread: the earliest risk-off detector in the
    # checklist. T10Y2Y and NFCI complete the curve/conditions read.
    assert {"BAMLH0A0HYM2", "T10Y2Y", "NFCI", "CPIAUCSL", "PCEPILFE"} <= series


def test_deep_merge_overrides_only_named_keys():
    """settings.local.yaml sets what it names and leaves the rest untouched."""
    base = {"database": {"path": "equitydash.db"}, "universe": {"tracked": [], "x": 1}}
    override = {"universe": {"tracked": [{"ticker": "AAPL"}]}}
    merged = config._deep_merge(base, override)
    assert merged["database"] == {"path": "equitydash.db"}
    assert merged["universe"]["x"] == 1
    assert merged["universe"]["tracked"] == [{"ticker": "AAPL"}]


def _thesis(**overrides) -> dict:
    card = {
        "ticker": "AAPL",
        "cik": "0000320193",
        "thesis": "Installed base converts hardware share into recurring services revenue.",
        "value_accrual": "Buybacks shrink the share count faster than SBC dilutes it.",
        "key_metric": "WeightedAverageNumberOfDilutedSharesOutstanding",
        "invalidation": "Diluted share count rises year over year for two quarters.",
        "review_date": "2027-01-15",
    }
    card.update(overrides)
    return card


def test_thesis_without_invalidation_is_rejected():
    """A company with no falsification criterion does not enter the universe (section 5.2)."""
    with pytest.raises(ValueError, match="invalidation"):
        config.validate_theses([_thesis(invalidation="")])


def test_thesis_missing_cik_is_rejected():
    """The CIK is the real key; a card without one cannot be joined to SEC data (section 9.3)."""
    with pytest.raises(ValueError, match="AAPL"):
        config.validate_theses([_thesis(cik=None)])


def test_complete_thesis_is_accepted():
    config.validate_theses([_thesis()])


def test_public_mode_env_var_wins(monkeypatch):
    """A deployed instance flips PUBLIC_MODE without editing config (sections 5.1, 11)."""
    monkeypatch.setenv("PUBLIC_MODE", "1")
    config.load_settings.cache_clear()
    assert config.load_settings().public_mode is True
    config.load_settings.cache_clear()


# --- The CIK must survive YAML (section 9.3) ----------------------------------


def test_an_unquoted_cik_is_refused_instead_of_silently_becoming_another_company():
    """The bug this guards is invisible after the fact, so it is caught at load time.

    PyYAML resolves a bare number with leading zeros as **octal**. Alphabet's real CIK,
    written ``cik: 0001652044``, parses to the integer 480292 — which zero-fills back to
    ``0000480292``, a syntactically perfect CIK belonging to somebody else. Nothing
    downstream can notice: by then the original digits no longer exist.
    """
    parsed = yaml.safe_load('cik: 0001652044')["cik"]
    assert parsed == 480292, "PyYAML stopped reading leading zeros as octal"

    with pytest.raises(ValueError, match="Quote it"):
        config.validate_theses([_card(cik=parsed)])


def test_a_quoted_cik_is_accepted():
    config.validate_theses([_card(cik="0001652044")])


def test_a_cik_that_is_not_ten_digits_is_refused():
    with pytest.raises(ValueError, match="not a CIK"):
        config.validate_theses([_card(cik="00016520440000")])
    with pytest.raises(ValueError, match="not a CIK"):
        config.validate_theses([_card(cik="GOOGL")])


def _card(**overrides) -> dict:
    card = {
        "ticker": "GOOGL", "cik": "0001652044", "thesis": "t",
        "value_accrual": "v", "key_metric": "k", "invalidation": "i",
        "review_date": "2027-01-01",
    }
    card.update(overrides)
    return card


# --- The fourth circle: candidates under study (section 5.1) ------------------


def _candidate(**overrides) -> dict:
    entry = {"ticker": "HIMS", "cik": "0001773751"}
    entry.update(overrides)
    return entry


def test_a_candidate_needs_only_a_ticker_and_a_cik():
    """The deadlock this circle resolves.

    Without it the panel refused to fetch a single figure about a company until its thesis
    was complete — so the tool built to support the research could not be used *during* it,
    and the only way out was to invent an invalidation criterion.
    """
    config.validate_watchlist([_candidate()], [])


def test_a_candidate_without_a_cik_is_refused():
    """The CIK is the key (section 9.3); without it nothing can be fetched anyway."""
    with pytest.raises(ValueError, match="missing"):
        config.validate_watchlist([{"ticker": "HIMS"}], [])


def test_a_candidate_cik_gets_the_same_octal_guard():
    """HIMS is one of the vulnerable ones: 0001773751 reads as octal 522217."""
    parsed = yaml.safe_load("cik: 0001773751")["cik"]
    assert parsed == 522217, "PyYAML stopped reading leading zeros as octal"

    with pytest.raises(ValueError, match="Quote it"):
        config.validate_watchlist([_candidate(cik=parsed)], [])


def test_a_draft_thesis_on_a_candidate_is_carried_not_rejected():
    """Notes taken mid-research are useful. What a draft never does is promote itself."""
    config.validate_watchlist([_candidate(thesis="telesalud con marca propia")], [])


def test_the_same_ticker_cannot_be_in_both_circles():
    """Ambiguous rather than harmless: the two answer "has a thesis?" differently."""
    with pytest.raises(ValueError, match="at once"):
        config.validate_watchlist([_candidate()], [_card(ticker="HIMS")])


def test_promoting_means_removing_it_from_the_watchlist():
    """The error has to say what to do, not just that something is wrong."""
    with pytest.raises(ValueError, match="REMOVING it from watchlist"):
        config.validate_watchlist([_candidate()], [_card(ticker="HIMS")])


def test_the_two_circles_are_separate_but_fetched_together(tmp_path, monkeypatch):
    """Fetching is not judging: the union decides downloads, tracked decides opinions."""
    local = tmp_path / "settings.local.yaml"
    local.write_text(
        'universe:\n'
        '  watchlist:\n'
        '    - ticker: HIMS\n'
        '      cik: "0001773751"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()
    settings = config.load_settings()

    assert settings.tracked_companies == [], "a candidate must never count as universe"
    assert [c["ticker"] for c in settings.watchlist_companies] == ["HIMS"]
    assert [c["ticker"] for c in settings.researched_companies] == ["HIMS"]
    config.load_settings.cache_clear()


def test_a_candidate_is_enough_to_make_the_sec_ingester_run(tmp_path, monkeypatch):
    """The whole point: data flows before the thesis exists."""
    from ingest.sec_xbrl import SecXbrlIngester

    local = tmp_path / "settings.local.yaml"
    local.write_text(
        'universe:\n  watchlist:\n    - ticker: HIMS\n      cik: "0001773751"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    monkeypatch.setenv("SEC_USER_AGENT", "Nombre correo@x.com")
    config.load_settings.cache_clear()
    settings = config.load_settings()

    assert SecXbrlIngester.companies_for(settings) == [("0001773751", "HIMS")]
    assert SecXbrlIngester.is_available(settings) is True
    config.load_settings.cache_clear()


def test_a_candidate_gets_its_prices_downloaded_too(tmp_path, monkeypatch):
    from ingest.prices import PricesIngester

    local = tmp_path / "settings.local.yaml"
    local.write_text(
        'universe:\n  watchlist:\n    - ticker: HIMS\n      cik: "0001773751"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", local)
    config.load_settings.cache_clear()

    assert "HIMS" in PricesIngester.tickers_for(config.load_settings())
    config.load_settings.cache_clear()


# --- Exit rules (section 2, level 4; phase 4) --------------------------------------------


def _card(**extra):
    card = {"ticker": "AAPL", "cik": "0000320193", "thesis": "t", "value_accrual": "v",
            "key_metric": "k", "invalidation": "i", "review_date": "2027-01-01"}
    return {**card, **extra}


def test_an_exit_rule_needs_its_prose():
    rule = {"rule_id": "a1", "kind": "take_profit", "trigger": "", "action": "vender"}
    with pytest.raises(ValueError, match="exit rule missing"):
        config.validate_theses([_card(exit_ladder=[rule])])


def test_an_exit_rule_kind_is_one_of_three():
    rule = {"rule_id": "a1", "kind": "stop_loss", "trigger": "t", "action": "a"}
    with pytest.raises(ValueError, match="stop_loss"):
        config.validate_theses([_card(exit_ladder=[rule])])


def test_exit_rule_ids_are_unique_across_cards():
    """The id keys the table and the alert dedup: two rules sharing it silence each other."""
    rule = {"rule_id": "dup", "kind": "rebalance", "trigger": "t", "action": "a"}
    other = _card(ticker="MSFT", cik="0000789019", exit_ladder=[rule])
    with pytest.raises(ValueError, match="duplicated"):
        config.validate_theses([_card(exit_ladder=[rule]), other])


def test_a_complete_exit_rule_loads():
    rule = {"rule_id": "a1", "kind": "take_profit", "trigger": "t", "action": "a",
            "rule": {"metric": "weight", "operator": ">", "threshold": 0.12}}
    config.validate_theses([_card(exit_ladder=[rule])])


def test_the_panel_sections_keep_their_keys():
    """A block pasted at a shallower indentation in the middle of a mapping re-parents every
    key below it, with no error: on 2026-09-25 `panel.level1.readings` and `charts` ended up
    under `panel.risk`, and the landing page silently lost its tiles and charts."""
    panel = config.load_settings().raw["panel"]
    assert {"readings", "charts", "lookback_days", "key_readings"} <= set(panel["level1"])
    assert set(panel["risk"]) == {"window_days", "factor_years", "t_threshold"}
    assert set(panel["journal"]) == {"tolerance_days"}
