from abc import abstractmethod

from jumper_extension.adapters.data import NodeInfo
from jumper_extension.monitor.metrics.common import CollectorBackend
from jumper_extension.monitor.metrics.context import CollectionContext


class MemoryCollectorBackend(CollectorBackend):
    """Base for memory metric backends."""

    name = "memory-base"

    def __init__(self, node_info: NodeInfo):
        self._node_info = node_info

    @abstractmethod
    def collect(self, level: str, context: CollectionContext) -> float: ...
