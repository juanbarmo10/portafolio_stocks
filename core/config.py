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
    # Section 8 phase 5: the PUBLIC copy's database. Deliberately not DATABASE_URL, which
    # would switch the whole local pipeline to it — the local base stays SQLite and full.
    "PUBLIC_DATABASE_URL",
    # Read by db.database directly; listed here so its value is redacted like any other
    # secret wherever the settings' secrets are used (logs, Telegram).
    "DATABASE_URL",
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

    # --- Universe (section 5.1): four concentric circles ----------------------

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

    @property
    def watchlist_companies(self) -> list[dict[str, Any]]:
        """Candidates **under study**: ticker and CIK only, no thesis yet (section 5.1).

        The fourth circle, added on 2026-09-23. Without it the panel had a deadlock: it
        would not fetch a single figure about a company until its thesis card was complete,
        so the tool built to support the research could not be used *during* it — and the
        only way out was to invent an invalidation criterion, which is exactly what section
        5.2 exists to prevent.

        A watchlist entry buys data and nothing else. It never reaches the invalidation
        board, never counts as universe, and never acquires an opinion it has not earned.
        """
        return list(self.raw.get("universe", {}).get("watchlist", []) or [])

    @property
    def researched_companies(self) -> list[dict[str, Any]]:
        """Every company the ingesters should fetch: tracked **and** under study.

        The union exists only to decide *what to download*. Anything that expresses a
        judgement reads :attr:`tracked_companies`, which is the set with a written thesis.
        """
        return [*self.tracked_companies, *self.watchlist_companies]

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
    seen_rules: set[str] = set()
    for entry in companies:
        problems.extend(_exit_ladder_problems(entry, seen_rules))
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


EXIT_KINDS = ("take_profit", "thesis_invalidation", "rebalance")
EXIT_REQUIRED_FIELDS = ("rule_id", "kind", "trigger", "action")


def _exit_ladder_problems(entry: dict[str, Any], seen: set[str]) -> list[str]:
    """What is wrong with a card's ``exit_ladder``, if anything (section 2, level 4).

    Exit rules are written *before* buying, so they are held to the thesis card's standard:
    the prose is mandatory, the structured ``rule`` optional. A duplicated ``rule_id`` is
    rejected because it is the rule's key in ``exit_ladder`` and in the alert dedup — two
    rules sharing it would silence one another.
    """
    ladder = entry.get("exit_ladder")
    if ladder is None:
        return []
    label = entry.get("ticker") or entry.get("cik") or "<unnamed>"
    if not isinstance(ladder, list):
        return [f"{label}: exit_ladder must be a list"]
    problems: list[str] = []
    for rule in ladder:
        if not isinstance(rule, dict):
            problems.append(f"{label}: exit_ladder entries must be mappings")
            continue
        missing = [f for f in EXIT_REQUIRED_FIELDS if not str(rule.get(f) or "").strip()]
        if missing:
            problems.append(f"{label}: exit rule missing {missing}")
            continue
        if rule["kind"] not in EXIT_KINDS:
            problems.append(f"{label}: exit rule {rule['rule_id']} has kind "
                            f"{rule['kind']!r}; use one of {list(EXIT_KINDS)}")
        if rule["rule_id"] in seen:
            problems.append(f"{label}: duplicated exit rule_id {rule['rule_id']!r}")
        seen.add(str(rule["rule_id"]))
    return problems


WATCHLIST_REQUIRED_FIELDS = ("ticker", "cik")


def validate_watchlist(
    watchlist: list[dict[str, Any]], tracked: list[dict[str, Any]]
) -> None:
    """Validate the under-study circle (section 5.1).

    Only ``ticker`` and ``cik`` are required, and that is the entire point: a candidate is
    something you are still reading about. Thesis fields may be present as a draft and are
    simply carried; what a draft never does is promote itself, because promotion is the
    act of moving the entry to ``tracked``, where :func:`validate_theses` applies in full.

    Raises:
        ValueError: On a missing or malformed field, or on a ticker that appears in both
            circles — that one is ambiguous rather than harmless: the two answer the
            question "does this company have a thesis?" differently.
    """
    problems: list[str] = []
    for entry in watchlist:
        missing = [
            f for f in WATCHLIST_REQUIRED_FIELDS if not str(entry.get(f) or "").strip()
        ]
        if missing:
            label = entry.get("ticker") or entry.get("cik") or "<unnamed>"
            problems.append(f"{label}: missing {missing}")
            continue
        problem = _cik_problem(entry["cik"])
        if problem:
            problems.append(f"{entry['ticker']}: {problem}")

    both = {str(e.get("ticker")) for e in watchlist} & {
        str(e.get("ticker")) for e in tracked
    }
    if both:
        problems.append(
            f"{sorted(both)}: in `tracked` and in `watchlist` at once. A company either "
            "has a written thesis or is still being read about; it cannot be both. "
            "Promoting means REMOVING it from watchlist."
        )

    if problems:
        raise ValueError(
            "Invalid entries in universe.watchlist (CLAUDE.md section 5.1 — a candidate "
            "under study needs a ticker and a CIK, nothing more):\n  "
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
    validate_watchlist(settings.watchlist_companies, settings.tracked_companies)
    return settings
