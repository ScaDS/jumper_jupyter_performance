from abc import abstractmethod

from jumper_extension.monitor.metrics.common import CollectorBackend
from jumper_extension.monitor.metrics.context import CollectionContext


class IoCollectorBackend(CollectorBackend):
    """Base for I/O metric backends."""

    name = "io-base"

    @abstractmethod
    def collect(self, level: str, context: CollectionContext) -> list[int]: ...
