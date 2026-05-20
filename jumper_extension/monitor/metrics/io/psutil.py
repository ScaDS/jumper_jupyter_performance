from typing import Any

import psutil

from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.io.common import IoCollectorBackend


class PsutilIoCollector(IoCollectorBackend):
    """I/O backend implemented via psutil."""

    name = "io-psutil"

    def _add_io(self, totals: list[int], io_data: Any):
        if io_data:
            totals[0] += io_data.read_count
            totals[1] += io_data.write_count
            totals[2] += io_data.read_bytes
            totals[3] += io_data.write_bytes

    def collect(self, level: str, context: CollectionContext) -> list[int]:
        totals = [0, 0, 0, 0]
        if level == "process":
            for pid in context["process_pids"]:
                self._add_io(totals, context["io"].get(pid))
        elif level == "system":
            # Use disk_io_counters for a single-syscall system total
            dio = psutil.disk_io_counters()
            if dio:
                totals = [
                    dio.read_count, dio.write_count,
                    dio.read_bytes, dio.write_bytes,
                ]
        elif level == "user":
            for pid in context["user_pids"]:
                self._add_io(totals, context["io"].get(pid))
        else:  # slurm
            for pid in context["slurm_pids"]:
                self._add_io(totals, context["io"].get(pid))
        return totals
