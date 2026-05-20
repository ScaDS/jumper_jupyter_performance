import logging
from typing import Iterable

from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.gpu.common import GpuCollectorBackend

logger = logging.getLogger("extension")


class NvmlGpuCollector(GpuCollectorBackend):
    """NVIDIA NVML backend (uses pynvml)."""

    name = "nvidia-nvml"

    def __init__(self, uid: int, slurm_job: str):
        super().__init__(uid, slurm_job)
        self._pynvml = None
        self._handles = []

    def _iter_handles(self) -> Iterable[object]:
        return self._handles

    def _get_util_rates(self, handle: object):
        if self._pynvml is None:
            class DefaultUtilRates:
                gpu = 0.0
                memory = 0.0

            return DefaultUtilRates()
        try:
            return self._pynvml.nvmlDeviceGetUtilizationRates(handle)
        except self._pynvml.NVMLError:
            # If permission denied or other error, use default values
            class DefaultUtilRates:
                gpu = 0.0
                memory = 0.0

            return DefaultUtilRates()

    def setup(self) -> dict:
        # Logic is intentionally kept identical to the previous implementation.
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            globals()["pynvml"] = pynvml
            ngpus = self._pynvml.nvmlDeviceGetCount()
            self._handles = [
                self._pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(ngpus)
            ]
            if self._handles:
                handle = self._handles[0]
                gpu_mem = round(
                    self._pynvml.nvmlDeviceGetMemoryInfo(handle).total
                    / (1024**3),
                    2,
                    )
                name = self._pynvml.nvmlDeviceGetName(handle)
                gpu_name = name.decode() if isinstance(name, bytes) else name
                return {"gpu_memory": gpu_mem, "gpu_name": gpu_name}
        except ImportError:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.PYNVML_NOT_AVAILABLE
                ]
            )
            self._handles = []
        except Exception:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.NVIDIA_DRIVERS_NOT_AVAILABLE
                ]
            )
            self._handles = []
        return {}

    def _collect_system(self, handle: object) -> tuple[float, float, float]:
        util_rates = self._get_util_rates(handle)
        memory_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
        return util_rates.gpu, 0.0, memory_info.used / (1024**3)

    def _collect_process(
        self,
        handle: object,
        context: CollectionContext,
    ) -> tuple[float, float, float]:
        util_rates = self._get_util_rates(handle)
        pids = context["process_pids"]
        process_mem = (
                sum(
                    p.usedGpuMemory
                    for p in self._pynvml.nvmlDeviceGetComputeRunningProcesses(
                        handle
                    )
                    if p.pid in pids and p.usedGpuMemory
                )
                / (1024**3)
        )
        return util_rates.gpu if process_mem > 0 else 0.0, 0.0, process_mem

    def _collect_other(
        self,
        handle: object,
        level: str,
        context: CollectionContext,
    ) -> tuple[float, float, float]:
        util_rates = self._get_util_rates(handle)
        if self._pynvml is None:
            return 0.0, 0.0, 0.0
        try:
            all_processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(
                handle
            )
            filtered_gpu_processes = [
                p for p in all_processes
                if self._filter_process(p.pid, level)
            ]
        except Exception:
            return 0.0, 0.0, 0.0
        filtered_mem = (
                sum(
                    p.usedGpuMemory
                    for p in filtered_gpu_processes
                    if p.usedGpuMemory
                )
                / (1024**3)
        )
        filtered_util = (
            (
                    util_rates.gpu
                    * len(filtered_gpu_processes)
                    / max(len(all_processes), 1)
            )
            if filtered_gpu_processes
            else 0.0
        )
        return filtered_util, 0.0, filtered_mem

    def shutdown(self) -> None:
        return None
