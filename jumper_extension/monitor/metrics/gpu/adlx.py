import logging
from typing import Iterable

from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.gpu.common import GpuCollectorBackend

logger = logging.getLogger("extension")


class AdlxGpuCollector(GpuCollectorBackend):
    """AMD ADLX backend (uses ADLXPybind)."""

    name = "amd-adlx"

    def __init__(self, uid: int, slurm_job: str):
        super().__init__(uid, slurm_job)
        self._adlx_helper = None
        self._adlx_system = None
        self._handles = []

    def _iter_handles(self) -> Iterable[object]:
        return self._handles

    def setup(self) -> dict:
        # Logic is intentionally kept identical to the previous implementation.
        try:
            from ADLXPybind import ADLXHelper, ADLX_RESULT

            self._adlx_helper = ADLXHelper()
            if self._adlx_helper.Initialize() != ADLX_RESULT.ADLX_OK:
                self._handles = []
                return {}
            self._adlx_system = self._adlx_helper.GetSystemServices()
            gpus_list = self._adlx_system.GetGPUs()
            num_amd_gpus = gpus_list.Size()
            self._handles = [
                gpus_list.At(i) for i in range(num_amd_gpus)
            ]
            if self._handles:
                gpu = self._handles[0]
                # Get memory info
                gpu_mem_info = gpu.TotalVRAM()
                gpu_mem = round(gpu_mem_info / (1024**3), 2)
                # Get GPU name
                gpu_name = gpu.Name()
                return {"gpu_memory": gpu_mem, "gpu_name": gpu_name}
        except ImportError:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.ADLX_NOT_AVAILABLE
                ]
            )
        except Exception:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.AMD_DRIVERS_NOT_AVAILABLE
                ]
            )
        self._handles = []
        return {}

    def _collect_system(self, handle: object) -> tuple[float, float, float]:
        try:
            if self._adlx_system is None:
                return 0.0, 0.0, 0.0
            # Get performance metrics interface
            perf_monitoring = (
                self._adlx_system.GetPerformanceMonitoringServices()
            )

            # Get current metrics
            current_metrics = perf_monitoring.GetCurrentPerformanceMetrics(
                handle
            )

            # Get GPU utilization
            util = current_metrics.GPUUsage()

            # Get memory info
            mem_info = current_metrics.GPUVRAMUsage()

            # AMD ADLX doesn't provide memory bandwidth easily
            return util, 0.0, mem_info / 1024.0
        except Exception:
            # If we can't get metrics, return zeros
            return 0.0, 0.0, 0.0

    def _collect_process(
        self,
        handle: object,
        context: CollectionContext,
    ) -> tuple[float, float, float]:
        # AMD ADLX doesn't provide per-process metrics easily
        return 0.0, 0.0, 0.0

    def _collect_other(
        self,
        handle: object,
        level: str,
        context: CollectionContext,
    ) -> tuple[float, float, float]:
        # AMD ADLX doesn't provide per-user metrics easily
        return 0.0, 0.0, 0.0

    def shutdown(self) -> None:
        return None
