"""Shared test isolation.

Tests must describe the *repository*, not the machine they run on. Without this,
a developer who fills in ``config/.env`` or writes a personal
``config/settings.local.yaml`` gets different results from CI — and the failure looks
like a code bug instead of leaked local state.

The autouse fixture below cuts both paths and every ambient secret, so a test that
wants a key sets it explicitly with ``monkeypatch.setenv``.
"""

from __future__ import annotations

import pytest

from core import config

# Runtime overrides that would otherwise leak the developer's shell into a test.
_RUNTIME_ENV = ("DATABASE_URL", "PUBLIC_MODE", "LOG_LEVEL")


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch, tmp_path):
    """Run every test against the tracked config only, with no secrets present."""
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / "absent.env")
    monkeypatch.setattr(config, "SETTINGS_LOCAL_PATH", tmp_path / "absent.local.yaml")
    for key in (*config._SECRET_KEYS, *_RUNTIME_ENV):
        monkeypatch.delenv(key, raising=False)

    # The settings loader is lru_cached; clear it on both sides so neither a previous
    # test nor this one leaves a cached Settings behind.
    config.load_settings.cache_clear()
    yield
    config.load_settings.cache_clear()
