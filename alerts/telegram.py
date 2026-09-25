"""Telegram delivery for alerts (CLAUDE.md sections 3, 8 phase 4). Ported from cryptodash.

One :class:`TelegramSender` with ``send(text) -> bool``. Without ``TELEGRAM_TOKEN`` and
``TELEGRAM_CHAT_ID`` it runs **disabled**: the message is logged, nothing leaves the
machine, and the pipeline stays testable without a bot.

Two differences from cryptodash, both about what leaves the machine:

- **Plain text, no Markdown.** Telegram rejects a whole message over one unbalanced ``_``
  or ``*`` in Markdown mode, and alert texts carry series ids and exception messages that
  contain both. A rejected alert is a silent alert.
- **Secrets are redacted from the text before sending.** An ingest-failure alert quotes
  the exception, and an exception can quote a URL with a key in it. The log already
  redacts (``core.logging_setup``); a message to a third-party service must too.

Setup (free): create a bot with @BotFather, put its token in ``TELEGRAM_TOKEN`` and your
chat id in ``TELEGRAM_CHAT_ID`` (config/.env).
"""

from __future__ import annotations

from typing import Iterable

import requests

from core.config import Settings
from core.logging_setup import REDACTED, get_logger

log = get_logger(__name__)

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
# Telegram's limit is 4096 characters; the margin leaves room for the truncation note.
MAX_LENGTH = 4000
MIN_SECRET_LENGTH = 8


def redact(text: str, secrets: Iterable[str | None]) -> str:
    """Replace every secret value in ``text``. Short or empty values are left alone."""
    for secret in sorted({s for s in secrets if s and len(s) >= MIN_SECRET_LENGTH},
                         key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


class TelegramSender:
    """Sends plain-text messages to one chat, or logs them when not configured."""

    def __init__(self, settings: Settings, timeout_s: int = 15) -> None:
        self._token = settings.secret("TELEGRAM_TOKEN")
        self._chat_id = settings.secret("TELEGRAM_CHAT_ID")
        self._secrets = [v for v in settings.secrets.values() if v]
        self._timeout = timeout_s

    @property
    def enabled(self) -> bool:
        """True when both the token and the chat id are configured."""
        return bool(self._token and self._chat_id)

    def send(self, text: str) -> bool:
        """Deliver ``text``. ``True`` only when Telegram accepted it."""
        text = redact(text, self._secrets)
        if len(text) > MAX_LENGTH:
            text = text[:MAX_LENGTH] + "\n… (mensaje recortado; el log tiene el texto completo)"
        if not self.enabled:
            log.info("Telegram not configured; alert logged only:\n%s", text)
            return False
        try:
            response = requests.post(
                API_URL.format(token=self._token),
                json={"chat_id": self._chat_id, "text": text,
                      "disable_web_page_preview": True},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            # The token is in the URL; the redacting log formatter keeps it out of the line.
            log.error("Telegram send failed: %s", exc)
            return False
        if response.status_code != 200:
            description = ""
            try:
                description = response.json().get("description", "")
            except ValueError:
                pass
            log.error("Telegram rejected the message: HTTP %d %s",
                      response.status_code, description)
            return False
        return True
