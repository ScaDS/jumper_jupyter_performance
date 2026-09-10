"""Collecting the answers that only exist in the extension's log.

The reviewer degrades a fast replay mode to the full one with a warning rather
than failing, so "which mode actually ran" is stated once, in a log line, and
nowhere in the state the experiment can read afterwards. Without capturing it,
a report could show two replay modes of which one is secretly the other.
"""
from __future__ import annotations

import logging

EXTENSION_LOGGER = "extension"

# Both degradation paths - the registry refusing a mode up front, and the
# runner giving up on one mid-benchmark - end their warning with this.
_FALLBACK_MARKER = "falling back to the full replay"


class WarningCollector(logging.Handler):
    """Keeps the extension's warnings for the current window."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:
            # A logging handler that raises would take the review down with
            # it; a lost warning is the cheaper failure.
            pass

    def reset(self) -> None:
        self.messages.clear()

    def drain(self) -> list[str]:
        messages = list(self.messages)
        self.messages.clear()
        return messages

    def attach(self) -> None:
        logger = logging.getLogger(EXTENSION_LOGGER)
        if self not in logger.handlers:
            logger.addHandler(self)


def degraded_to_full(messages: list[str]) -> bool:
    """Whether any warning says the requested replay mode was abandoned."""
    return any(_FALLBACK_MARKER in message for message in messages)


def actual_replay_mode(requested: str, messages: list[str]) -> str:
    """The mode that really ran, given what was asked for and what was said."""
    return "full" if degraded_to_full(messages) else requested
