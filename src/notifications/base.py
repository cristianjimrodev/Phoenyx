"""Notification system base classes."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class NotificationEvent:
    event_type: str  # signal, trade_opened, trade_closed, risk_blocked, error, system
    symbol: str
    message: str
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class Notifier(ABC):
    """Abstract base for notification channels."""

    @abstractmethod
    async def send(self, event: NotificationEvent) -> bool:
        """Send a notification. Returns True on success."""
