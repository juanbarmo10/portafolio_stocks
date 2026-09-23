"""Configuration loading for equitydash (CLAUDE.md sections 3, 10).

Loads externalized settings from ``config/settings.yaml``, deep-merges the gitignored
``config/settings.local.yaml`` over it, and reads secrets from ``config/.env``. Nothing
is hardcoded: tickers, thresholds, windows and CIKs live in YAML; secrets live in the
environment and never enter the repo or the logs.

Inputs (read):
    - config/settings.yaml       : non-secret configuration (tracked).
    - config/settings.local.yaml : personal data — theses, target weights, fiscal (gitignored).
    - config/.env (optional)     : secrets (via python-dotenv).
    - environment variables      : take precedence over .env, which takes precedence
                                   over settings.yaml where they overlap.

Outputs:
    - Settings object exposing the parsed config plus resolved paths and secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repository root = two levels up from this file (core/config.py -> repo root).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_DIR: Path = REPO_ROOT / "config"
SETTINGS_PATH: Path = CONFIG_DIR / "settings.yaml"
# Gitignored local override deep-merged over settings.yaml. Holds PERSONAL config
# (theses per section 5.2, target weights, fiscal parameters) that must never be tracked.
SETTINGS_LOCAL_PATH: Path = CONFIG_DIR / "settings.local.yaml"
ENV_PATH: Path = CONFIG_DIR / ".env"

# Secrets read from the environment. Names only — values never reach a log line.
_SECRET_KEYS = (
    "FRED_API_KEY",
    "SEC_USER_AGENT",       # section 4.4: SEC blocks requests without a real name + email
    "IBKR_FLEX_TOKEN",      # section 4.1: read-only by construction, cannot place orders
    "IBKR_FLEX_QUERY_ID",
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID",
)

# Required theses fields (section 5.2). `invalidation` is listed here *and* enforced
# NOT NULL in the schema: a company without a falsification criterion does not enter
# the tracking universe.
THESIS_REQUIRED_FIELDS = (
    "ticker",
    "cik",
    "thesis",
    "value_accrual",
    "key_metric",
    "invalidation",
    "review_date",
)


@dataclass(frozen=True)
class Settings:
    """Parsed configuration plus resolved paths and secrets.

    Attributes:
        raw: The full merged settings mapping, for sections without a typed accessor.
        db_path: Absolute path to the SQLite database file.
        log_level: Effective log level (LOG_LEVEL env var overrides settings.yaml).
        secrets: Secrets present in the environment/.env (may be empty).
        public_mode: Hide the real account and the fiscal layer (sections 5.1, 11).
    """

    raw: dict[str, Any]
    db_path: Path
    log_level: str
    secrets: dict[str, str] = field(default_factory=dict)
    public_mode: bool = False

    # --- Universe (section 5.1): three concentric circles ---------------------

    @property
    def market_references(self) -> list[str]:
        """Flat list of market-reference tickers (section 5.1). Not positions."""
        refs = self.raw.get("universe", {}).get("market_references", {})
        out: list[str] = []
        for group in refs.values():
            out.extend(group or [])
        # Deduplicate preserving order: a ticker may legitimately appear in two groups.
        return list(dict.fromkeys(out))

    @property
    def tracked_companies(self) -> list[dict[str, Any]]:
        """Companies with a written thesis (section 5.2). Lives in settings.local.yaml."""
        return list(self.raw.get("universe", {}).get("tracked", []) or [])

    def source(self, name: str) -> dict[str, Any]:
        """Return the parameter block for a named data source (e.g. 'fred', 'sec')."""
        return dict(self.raw.get("sources", {}).get(name, {}))

    def secret(self, key: str) -> str | None:
        """Return a secret by name, or None when absent."""
        return self.secrets.get(key)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict, failing loudly if it is missing or malformed."""
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Settings file {path} did not parse to a mapping.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (override wins; nested dicts merge).

    Non-dict values (including lists like ``universe.tracked``) are replaced wholesale,
    so a local override sets only the keys it names and leaves the rest untouched.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _cik_problem(cik: Any) -> str | None:
    """Reject a CIK that YAML may already have corrupted, before anything uses it.

    **A CIK must be quoted in YAML.** Written bare it is a number, and PyYAML resolves a
    number with leading zeros as *octal* — so Alphabet's ``0001652044`` silently becomes
    ``480292``, which zero-fills back to ``0000480292``: a different company's key, with
    no error anywhere. Nothing downstream can detect it, because by the time the config is
    loaded the original digits are gone (section 9.3 — the CIK is the primary key, and
    section 12 — a wrong number is worse than a missing one).

    Requiring a string is the only check that survives that, so it is the check made here:
    the damage happens during parsing, not after.
    """
    if not isinstance(cik, str):
        return (
            f"cik is {type(cik).__name__} {cik!r}, not text. Quote it in the YAML "
            '(cik: "0001652044"): unquoted, a CIK with leading zeros is read as an '
            "octal number and silently becomes a different company."
        )
    digits = cik.strip()
    if not digits.isdigit() or len(digits) > 10:
        return f"cik {cik!r} is not a CIK: expected up to 10 digits."
    return None


def validate_theses(companies: list[dict[str, Any]]) -> None:
    """Reject any thesis card missing a required field (section 5.2).

    The schema also enforces ``invalidation`` NOT NULL, but failing here means the
    user learns at config-load time rather than at write time.

    Raises:
        ValueError: If a company entry is missing any of THESIS_REQUIRED_FIELDS,
            listing the offending ticker and fields.
    """
    problems: list[str] = []
    for entry in companies:
        missing = [
            f for f in THESIS_REQUIRED_FIELDS if not str(entry.get(f) or "").strip()
        ]
        if missing:
            label = entry.get("ticker") or entry.get("cik") or "<unnamed>"
            problems.append(f"{label}: missing {missing}")
            continue
        problem = _cik_problem(entry["cik"])
        if problem:
            problems.append(f"{entry['ticker']}: {problem}")
    if problems:
        raise ValueError(
            "Incomplete thesis cards in config (CLAUDE.md section 5.2 — a company "
            "without an invalidation criterion does not enter the universe):\n  "
            + "\n  ".join(problems)
        )


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load and cache the effective settings.

    Precedence for overlapping values: environment variable > .env file >
    settings.local.yaml > settings.yaml. The result is cached; call
    ``load_settings.cache_clear()`` in tests that need a reload.

    Returns:
        A frozen :class:`Settings` instance.

    Raises:
        ValueError: If a tracked company has an incomplete thesis card (section 5.2).
    """
    # load_dotenv does not override already-set environment variables, so real env
    # vars keep precedence over .env by construction.
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    raw = _load_yaml(SETTINGS_PATH)

    # Optional gitignored local override. An all-comments file parses to None -> empty.
    if SETTINGS_LOCAL_PATH.exists():
        local = yaml.safe_load(SETTINGS_LOCAL_PATH.read_text(encoding="utf-8")) or {}
        if not isinstance(local, dict):
            raise ValueError(f"{SETTINGS_LOCAL_PATH} did not parse to a mapping.")
        raw = _deep_merge(raw, local)

    db_rel = raw.get("database", {}).get("path", "equitydash.db")
    db_path = Path(db_rel)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path

    log_level = os.getenv("LOG_LEVEL") or raw.get("logging", {}).get("level", "INFO")

    # Public/demo mode hides the real account (level 4) and the fiscal layer (section 11).
    # The env var wins so a deployed instance never depends on editing config.
    env_public = os.getenv("PUBLIC_MODE")
    if env_public is not None:
        public_mode = env_public.strip().lower() in {"1", "true", "yes", "on"}
    else:
        public_mode = bool(raw.get("app", {}).get("public_mode", False))

    secrets = {key: os.environ[key] for key in _SECRET_KEYS if os.environ.get(key)}

    settings = Settings(
        raw=raw,
        db_path=db_path,
        log_level=str(log_level).upper(),
        secrets=secrets,
        public_mode=public_mode,
    )
    validate_theses(settings.tracked_companies)
    return settings
