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
