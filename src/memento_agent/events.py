"""Lightweight progress events so the CLI (and later a tray UI) can listen in."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    kind: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[Event], None]


class EventBus:
    def __init__(self, handler: EventHandler | None = None) -> None:
        self._handler = handler

    def emit(self, kind: str, message: str, **data: Any) -> None:
        if self._handler is None:
            return
        self._handler(Event(kind=kind, message=message, data=data))
