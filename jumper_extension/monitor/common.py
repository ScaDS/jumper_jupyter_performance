import logging
from typing import Dict, Optional, Protocol, runtime_checkable

import pandas as pd

from jumper_extension.adapters.data import PerformanceData

logger = logging.getLogger("extension")

@runtime_checkable
class MonitorProtocol(Protocol):
    # required readable attributes
    interval: float
    data: "PerformanceData"
    start_time: Optional[float]
    wallclock_start_time: Optional[float]
    wallclock_stop_time: Optional[float]
    num_cpus: int
    num_system_cpus: int
    num_gpus: int
    gpu_memory: float
    memory_limits: dict
    cpu_handles: list[int]
    gpu_name: str
    # session state
    is_imported: bool
    session_source: Optional[str]

    # required control & lifecycle
    running: bool
    def start(self, interval: float = 1.0) -> None: ...
    def stop(self) -> None: ...


# Backward-compatible re-export
from jumper_extension.monitor.backends.thread import PerformanceMonitor  # noqa: F401


class MonitorUnavailableError(RuntimeError):
    """This monitor is a stub and cannot be used."""


class UnavailablePerformanceMonitor:
    """
    A stub that type-checks against PerformanceMonitor Protocol but fails at runtime.

    - Declares all required attributes for structural typing.
    - Any attribute access or method call raises MonitorUnavailableError,
      except 'running', which is always readable and returns False.
    """

    # --- Protocol surface ---
    interval: float
    data: "PerformanceData"
    start_time: Optional[float]
    wallclock_start_time: Optional[float]
    wallclock_stop_time: Optional[float]
    num_cpus: int
    num_system_cpus: int
    num_gpus: int
    gpu_memory: float
    memory_limits: dict
    cpu_handles: list[int]
    gpu_name: str
    running: bool

    def start(self, interval: float = 1.0) -> None: ...
    def stop(self) -> None: ...

    # --- Runtime behavior ---
    def __init__(self, reason: str = "Performance monitor is not available"):
        object.__setattr__(self, "_reason", reason)

    def __getattribute__(self, name: str):
        # allow a few safe attributes + running
        if name in {
            "_reason", "__class__", "__repr__", "__str__",
            "__init__", "__getattribute__", "__setattr__",
            "__dict__", "__annotations__"
        }:
            return object.__getattribute__(self, name)

        if name == "running":
            return False

        reason = object.__getattribute__(self, "_reason")
        raise MonitorUnavailableError(f"Access to '{name}' is not allowed: {reason}")

    def __setattr__(self, name: str, value):
        if name in {"_reason", "__dict__", "__annotations__"}:
            return object.__setattr__(self, name, value)
        reason = object.__getattribute__(self, "_reason")
        raise MonitorUnavailableError(f"Setting '{name}' is not allowed: {reason}")

    def __repr__(self) -> str:
        return f"<UnavailablePerformanceMonitor: {self._reason}>"


class OfflinePerformanceMonitor:
    """Offline monitor that satisfies MonitorProtocol.

    It holds static data frames plus metadata from a manifest; does not collect live data.
    """

    def __init__(
        self,
        *,
        manifest: Dict,
        perf_dfs: Dict[str, pd.DataFrame],
        source: Optional[str] = None,
    ):
        monitor_info = manifest.get("monitor", {})

        # Protocol surface
        self.interval = float(monitor_info.get("interval", 1.0) or 1.0)
        self.running = False
        self.start_time = monitor_info.get("start_time")
        self.stop_time = monitor_info.get("stop_time")
        self.wallclock_start_time = monitor_info.get("wallclock_start_time")
        self.wallclock_stop_time = monitor_info.get("wallclock_stop_time")

        # Hardware/context
        self.num_cpus = int(monitor_info.get("num_cpus", 0) or 0)
        self.num_system_cpus = int(monitor_info.get("num_system_cpus", self.num_cpus) or self.num_cpus)
        self.num_gpus = int(monitor_info.get("num_gpus", 0) or 0)
        self.gpu_memory = float(monitor_info.get("gpu_memory", 0.0) or 0.0)
        self.gpu_name = monitor_info.get("gpu_name", "") or ""
        self.cpu_handles = monitor_info.get("cpu_handles", []) or []
        self.memory_limits = monitor_info.get("memory_limits", {}) or {}

        # Performance data container
        self.data = PerformanceData(
            self.num_cpus,
            self.num_system_cpus,
            self.num_gpus,
        )
        for level, df in (perf_dfs or {}).items():
            try:
                self.data._validate_level(level)
            except Exception:
                pass
            self.data.data[level] = df

        # Imported session state
        self.is_imported = True
        self.session_source = source

    # No-op lifecycle
    def start(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.running = False

    def stop(self) -> None:
        self.running = False
