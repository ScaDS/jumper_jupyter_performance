from __future__ import annotations

from abc import abstractmethod

from jumper_extension.monitor.metrics.common import CollectorBackend
from jumper_extension.monitor.metrics.context import CollectionContext


class NetworkCollectorBackend(CollectorBackend):
    """Base for network metric backends."""

    name = "network-base"

    @abstractmethod
    def collect(self, level: str, context: CollectionContext) -> list[int]: ...
