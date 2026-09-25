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


_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Refuse any connection that leaves the machine.

    Found on 2026-09-24: a FRED test stubbed one request function, a new code path used
    another, and the suite reached the real API — and passed or failed depending on the
    network. Loopback stays allowed: several tests point at ``127.0.0.1:1`` precisely to
    get a refused connection.
    """
    import socket

    real_connect = socket.socket.connect

    def guarded(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _LOCAL_HOSTS and not host.startswith("/"):
            raise RuntimeError(f"test tried to reach the network: {address!r}")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
