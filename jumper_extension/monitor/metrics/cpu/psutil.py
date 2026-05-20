import psutil

from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.cpu.common import CpuCollectorBackend


class PsutilCpuCollector(CpuCollectorBackend):
    """CPU backend implemented via psutil."""

    name = "cpu-psutil"

    def collect(self, level: str, context: CollectionContext) -> list[float]:
        num_cpus = self._node_info.num_cpus
        if level == "system":
            return psutil.cpu_percent(percpu=True)
        elif level == "process":
            cpu_total = sum(
                context["cpu"].get(pid, 0.0) for pid in context["process_pids"]
            )
            return [cpu_total / num_cpus] * num_cpus
        elif level == "user":
            cpu_total = sum(
                context["cpu"].get(pid, 0.0) for pid in context["user_pids"]
            )
            return [cpu_total / num_cpus] * num_cpus
        else:  # slurm
            cpu_total = sum(
                context["cpu"].get(pid, 0.0) for pid in context["slurm_pids"]
            )
            return [cpu_total / num_cpus] * num_cpus
