import psutil

from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.memory.common import MemoryCollectorBackend


class PsutilMemoryCollector(MemoryCollectorBackend):
    """Memory backend implemented via psutil."""

    name = "memory-psutil"

    def collect(self, level: str, context: CollectionContext) -> float:
        if level == "system":
            return (
                psutil.virtual_memory().total
                - psutil.virtual_memory().available
            ) / (1024**3)
        elif level == "process":
            memory_total = sum(
                context["rss"].get(pid, 0) for pid in context["process_pids"]
            )
            return memory_total / (1024**3)
        elif level == "user":
            memory_total = sum(
                context["rss"].get(pid, 0) for pid in context["user_pids"]
            )
            return memory_total / (1024**3)
        else:  # slurm
            memory_total = sum(
                context["rss"].get(pid, 0) for pid in context["slurm_pids"]
            )
            return memory_total / (1024**3)
