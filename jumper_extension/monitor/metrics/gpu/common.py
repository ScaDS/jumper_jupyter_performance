from typing import Iterable

import psutil

from jumper_extension.utilities import is_slurm_available
from jumper_extension.monitor.metrics.common import CollectorBackend
from jumper_extension.monitor.metrics.context import CollectionContext


class GpuCollectorBackend(CollectorBackend):
    """A pluggable backend that provides GPU discovery and metric collection."""

    name = "gpu-base"

    def __init__(self, uid: int, slurm_job: str):
        self._uid = uid
        self._slurm_job = slurm_job

    def shutdown(self) -> None:
        """Clean up resources if needed."""
        return None

    def _iter_handles(self) -> Iterable[object]:
        return []

    def _filter_process(self, pid: int, mode: str) -> bool:
        """Filter a GPU process by user/slurm membership."""
        try:
            proc = psutil.Process(pid)
            if mode == "user":
                return proc.uids().real == self._uid
            elif mode == "slurm":
                if not is_slurm_available():
                    return False
                return proc.environ().get("SLURM_JOB_ID") == str(self._slurm_job)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        return False

    def _collect_system(self, handle: object) -> tuple[float, float, float]:
        raise NotImplementedError

    def _collect_process(
        self, handle: object, context: CollectionContext
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    def _collect_other(
        self, handle: object, level: str, context: CollectionContext
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    def collect(
        self,
        level: str,
        context: CollectionContext,
    ) -> tuple[list[float], list[float], list[float]]:
        """Collect metrics for the given level.

        Returns: (gpu_util, gpu_band, gpu_mem)
        """
        gpu_util, gpu_band, gpu_mem = [], [], []

        for handle in self._iter_handles():
            if level == "system":
                util, band, mem = self._collect_system(handle)
            elif level == "process":
                util, band, mem = self._collect_process(handle, context)
            else:  # user or slurm
                util, band, mem = self._collect_other(handle, level, context)
            gpu_util.append(util)
            gpu_band.append(band)
            gpu_mem.append(mem)

        return gpu_util, gpu_band, gpu_mem


class NullGpuCollector(GpuCollectorBackend):
    """A no-op backend used when no GPU backend is available."""

    name = "gpu-disabled"

    def _iter_handles(self) -> Iterable[object]:
        return []


class GpuDiscovery:
    """Finds all available GPU device backends at runtime."""

    def __init__(self, uid: int, slurm_job: str):
        self._uid = uid
        self._slurm_job = slurm_job

    def discover(self) -> list[GpuCollectorBackend]:
        from jumper_extension.monitor.metrics.gpu.nvml import NvmlGpuCollector
        from jumper_extension.monitor.metrics.gpu.adlx import AdlxGpuCollector

        return [
            NvmlGpuCollector(uid=self._uid, slurm_job=self._slurm_job),
            AdlxGpuCollector(uid=self._uid, slurm_job=self._slurm_job),
        ]


class MultiGpuCollector:
    """Combined GPU pipeline member — aggregates all discovered device backends."""

    def __init__(self, uid: int, slurm_job: str):
        self._uid = uid
        self._slurm_job = slurm_job
        self._backends: list[GpuCollectorBackend] = []

    def setup(self) -> dict:
        self._backends = GpuDiscovery(self._uid, self._slurm_job).discover()
        gpu_memory = 0.0
        gpu_name_parts = []
        for backend in self._backends:
            meta = backend.setup() or {}
            if meta.get("gpu_memory", 0) > 0 and gpu_memory == 0:
                gpu_memory = meta["gpu_memory"]
            if meta.get("gpu_name"):
                gpu_name_parts.append(meta["gpu_name"])
        num_gpus = sum(len(b._handles) for b in self._backends)
        return {
            "num_gpus": num_gpus,
            "gpu_memory": gpu_memory,
            "gpu_name": ", ".join(gpu_name_parts),
        }

    def snapshot(self, context: CollectionContext) -> None:
        for backend in self._backends:
            backend.snapshot(context)

    def collect(
        self,
        level: str,
        context: CollectionContext,
    ) -> tuple[list[float], list[float], list[float]]:
        util, band, mem = [], [], []
        for backend in self._backends:
            backend_util, backend_band, backend_mem = backend.collect(level, context)
            util.extend(backend_util)
            band.extend(backend_band)
            mem.extend(backend_mem)
        return util, band, mem
