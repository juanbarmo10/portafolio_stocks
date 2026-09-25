"""Secrets never reach a log line, whichever channel carries them (section 4.1)."""

from __future__ import annotations

import io
import logging

import requests

from core.logging_setup import REDACTED, configure_logging

KEY = "abcdef0123456789abcdef0123456789"


def captured(secrets, emit) -> str:
    configure_logging("INFO", secrets)
    stream = io.StringIO()
    logging.getLogger().handlers[0].setStream(stream)
    emit(logging.getLogger("t"))
    return stream.getvalue()


def test_a_key_in_an_exception_text_is_redacted():
    """The real channel: requests writes the query string into a ConnectionError, and
    retry() logs that text on every attempt."""
    def emit(log):
        try:
            requests.get("http://127.0.0.1:1/fred/series", params={"api_key": KEY}, timeout=1)
        except requests.RequestException as exc:
            log.warning("Attempt 1/4 failed (%s)", exc)

    text = captured([KEY], emit)
    assert KEY not in text
    assert REDACTED in text, "guards the guard: the key really was in the message"


def test_a_key_in_a_traceback_is_redacted():
    def emit(log):
        try:
            raise RuntimeError(f"url?api_key={KEY}")
        except RuntimeError:
            log.exception("boom")

    assert KEY not in captured([KEY], emit)


def test_short_or_empty_values_are_not_redacted():
    """A blank TELEGRAM_CHAT_ID must not black out every empty string in the log."""
    text = captured(["", None, "123"], lambda log: log.info("chat 123 ok"))
    assert "chat 123 ok" in text
