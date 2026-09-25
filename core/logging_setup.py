"""Structured logging setup (CLAUDE.md section 10).

Provides a single ``configure_logging`` entry point and a ``get_logger`` helper.
Records carry the ``source`` and ``series_id`` context fields when supplied via
``extra=``, so ingest logs are filterable by data source and by series.

The level is configurable at runtime through the LOG_LEVEL env var, resolved by
:func:`core.config.load_settings`.
"""

from __future__ import annotations

import logging
import sys
from typing import Iterable

# Context fields we want on every record; default to '-' when not provided.
_CONTEXT_FIELDS = ("source", "series_id")

_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "src=%(source)s series=%(series_id)s | %(message)s"
)


class _ContextFilter(logging.Filter):
    """Guarantee context fields exist so the formatter never raises KeyError."""

    def filter(self, record: logging.LogRecord) -> bool:
        for field_name in _CONTEXT_FIELDS:
            if not hasattr(record, field_name):
                setattr(record, field_name, "-")
        return True


# Shorter values are not redacted: a two-character "secret" would black out half the log.
_MIN_SECRET_LENGTH = 8
REDACTED = "[REDACTED]"


class _RedactingFormatter(logging.Formatter):
    """Replace every configured secret in the *finished* line, traceback included.

    Redacting at the formatter rather than at each call site is the point: a secret
    reaches the log through channels nobody writes on purpose. ``requests`` puts the
    query string — FRED's ``api_key`` among it — into the text of a ``ConnectionError``,
    and :func:`ingest.base.retry` logs that text on every attempt. Scrubbing each ingester
    would leave the next one to remember; this covers them all.
    """

    def __init__(self, fmt: str, secrets: Iterable[str]) -> None:
        super().__init__(fmt)
        self._secrets = sorted(
            {str(s) for s in secrets if s and len(str(s)) >= _MIN_SECRET_LENGTH},
            key=len, reverse=True,
        )

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        for secret in self._secrets:
            line = line.replace(secret, REDACTED)
        return line


def configure_logging(level: str = "INFO", secrets: Iterable[str] = ()) -> None:
    """Configure the root logger idempotently.

    Args:
        level: Log level name (e.g. 'DEBUG', 'INFO'). Case-insensitive.
        secrets: Values that must never appear in a log line (API keys, tokens); every
            occurrence is replaced by ``[REDACTED]``.

    Calling this more than once replaces existing handlers rather than stacking
    them, so repeated ingest runs do not duplicate log lines.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(_RedactingFormatter(_FORMAT, secrets))
    handler.addFilter(_ContextFilter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Assumes :func:`configure_logging` ran at startup."""
    return logging.getLogger(name)
