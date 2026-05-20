from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jumper_extension.monitor.backends.thread.monitor import PerformanceMonitor

from jumper_extension.adapters.data import NodeInfo
from jumper_extension.config.utils import instantiate, load_collectors_config


class PipelineBuilder:
    """Builds the collector pipeline for a PerformanceMonitor.

    Backends that inject node_info are deferred until after GPU discovery
    so they receive a complete NodeInfo at construction.
    """

    def __init__(self, monitor: PerformanceMonitor):
        self._monitor = monitor

    def build(self, deferred_keys: list[str] | None = None):
        deferred = self._build_main(deferred_keys or [])
        self._build_deferred(deferred)

    def _defer(self, inject_keys: list[str], deferred_keys: list[str]) -> bool:
        return bool(set(inject_keys) & set(deferred_keys))

    def _build_main(
        self,
        deferred_keys: list[str],
    ) -> list[tuple[dict, dict, list[str]]]:
        cfg = load_collectors_config()
        self._monitor._pipeline = []
        deferred = []
        num_gpus, gpu_memory, gpu_name = 0, 0.0, ""
        for collector_cfg in cfg["collectors"].values():
            collector_cfg = dict(collector_cfg)
            handler_cfg = collector_cfg.pop("handler")
            inject_keys = collector_cfg.pop("inject", [])
            if self._defer(inject_keys, deferred_keys):
                deferred.append((collector_cfg, handler_cfg, inject_keys))
                continue
            injected = {k: getattr(self._monitor, k) for k in inject_keys}
            backend = instantiate(collector_cfg, **injected)
            meta = backend.setup() or {}
            if "num_gpus" in meta:
                num_gpus = meta["num_gpus"]
                gpu_memory = meta.get("gpu_memory", 0.0)
                gpu_name = meta.get("gpu_name", "")
            self._monitor._pipeline.append((backend, instantiate(handler_cfg)))

        self._monitor.node_info = NodeInfo(
            node="local",
            num_cpus=self._monitor.num_cpus,
            num_system_cpus=self._monitor.num_system_cpus,
            num_gpus=num_gpus,
            gpu_memory=gpu_memory,
            gpu_name=gpu_name,
            memory_limits=self._monitor.memory_limits,
            cpu_handles=self._monitor.cpu_handles,
        )
        return deferred

    def _build_deferred(
        self,
        deferred: list[tuple[dict, dict, list[str]]],
    ):
        for collector_cfg, handler_cfg, inject_keys in deferred:
            injected = {k: getattr(self._monitor, k) for k in inject_keys}
            backend = instantiate(collector_cfg, **injected)
            backend.setup()
            self._monitor._pipeline.append((backend, instantiate(handler_cfg)))
