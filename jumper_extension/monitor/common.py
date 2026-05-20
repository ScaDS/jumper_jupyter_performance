from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import pandas as pd

from jumper_extension.adapters.data import NodeInfo, NodeDataStore

logger = logging.getLogger("extension")


@runtime_checkable
class MonitorProtocol(Protocol):
    interval: float
    running: bool
    start_time: Optional[float]
    wallclock_start_time: Optional[float]
    wallclock_stop_time: Optional[float]
    nodes: NodeDataStore
    is_imported: bool
    session_source: Optional[str]

    def start(self, interval: float = 1.0) -> None: ...
    def stop(self) -> None: ...


# Backward-compatible re-export
from jumper_extension.monitor.backends.thread import PerformanceMonitor  # noqa: F401


class MonitorUnavailableError(RuntimeError):
    """This monitor is a stub and cannot be used."""


class UnavailablePerformanceMonitor:
    """
    A stub that type-checks against MonitorProtocol but fails at runtime.

    Any attribute access or method call raises MonitorUnavailableError,
    except 'running', which always returns False.
    """

    interval: float
    running: bool
    start_time: Optional[float]
    wallclock_start_time: Optional[float]
    wallclock_stop_time: Optional[float]
    nodes: NodeDataStore
    is_imported: bool
    session_source: Optional[str]

    def start(self, interval: float = 1.0) -> None: ...
    def stop(self) -> None: ...

    def __init__(self, reason: str = "Performance monitor is not available"):
        object.__setattr__(self, "_reason", reason)

    def __getattribute__(self, name: str) -> Any:
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

    Holds static DataFrames plus metadata from a manifest; does not collect live data.
    """

    def __init__(
        self,
        *,
        manifest: Dict,
        perf_dfs: Dict[str, pd.DataFrame],
        source: Optional[str] = None,
    ):
        monitor_info = manifest.get("monitor", {})

        self.interval = float(monitor_info.get("interval", 1.0) or 1.0)
        self.running = False
        self.start_time = monitor_info.get("start_time")
        self.stop_time = monitor_info.get("stop_time")
        self.wallclock_start_time = monitor_info.get("wallclock_start_time")
        self.wallclock_stop_time = monitor_info.get("wallclock_stop_time")

        num_cpus = int(monitor_info.get("num_cpus", 0) or 0)
        num_system_cpus = int(monitor_info.get("num_system_cpus", num_cpus) or num_cpus)
        num_gpus = int(monitor_info.get("num_gpus", 0) or 0)

        node_info = NodeInfo(
            node="local",
            num_cpus=num_cpus,
            num_system_cpus=num_system_cpus,
            num_gpus=num_gpus,
            gpu_memory=float(monitor_info.get("gpu_memory", 0.0) or 0.0),
            gpu_name=monitor_info.get("gpu_name", "") or "",
            memory_limits=monitor_info.get("memory_limits", {}) or {},
            cpu_handles=monitor_info.get("cpu_handles", []) or [],
        )

        self.nodes = NodeDataStore()
        self.nodes.register_node(node_info)
        self.nodes.load_frames("local", perf_dfs or {})

        self.is_imported = True
        self.session_source = source

    def start(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.running = False

    def stop(self) -> None:
        self.running = False
