from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import psutil

from jumper_extension.adapters.data import NodeDataStore
from jumper_extension.monitor.pipeline import PipelineBuilder
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.utilities import detect_memory_limit, get_available_levels

logger = logging.getLogger("extension")


class PerformanceMonitor:
    def __init__(self):
        self.interval: float = 1.0
        self.running: bool = False
        self.start_time: float | None = None
        self.stop_time: float | None = None
        self.wallclock_start_time: float | None = None
        self.wallclock_stop_time: float | None = None
        self.monitor_thread: threading.Thread | None = None
        self.process = psutil.Process()
        self.n_measurements: int = 0
        self.n_missed_measurements: int = 0
        """
        on MacOS cpu_affinity is not implemented in psutil
        (raises AttributeError)
        set the num_cpus to the number of cpus in the system
        same for cpu_affinity
        """
        try:
            self.cpu_handles = self.process.cpu_affinity()
            self.num_cpus = len(self.cpu_handles)
        except AttributeError:
            self.cpu_handles = []
            self.num_cpus = len(psutil.cpu_percent(percpu=True))
        self.num_system_cpus = len(psutil.cpu_percent(percpu=True))
        self.pid: int = os.getpid()
        self.uid: int = os.getuid()
        self.slurm_job: str | int = os.environ.get("SLURM_JOB_ID", 0)
        self.levels: list[str] = get_available_levels()

        self.memory_limits: dict = {
            level: detect_memory_limit(level, self.uid, self.slurm_job)
            for level in self.levels
        }

        # Ordered list of (backend, handler) pairs built from collectors.yaml.
        # Each tick: all backends snapshot() the process state into a shared
        # context, then collect() + handler.transform() produce flat metric rows.
        self._pipeline: list[tuple[Any, Any]] | None = None
        PipelineBuilder(self).build(deferred_keys=["node_info"])

        self.nodes = NodeDataStore()
        self.nodes.register_node(self.node_info)

        self._bootstrap_schema()

        # session state
        self.is_imported: bool = False
        self.session_source: str | None = None

    @property
    def _process_backend(self):
        """Index 0: process collector is always first in collectors.yaml."""
        return self._pipeline[0][0]

    def _bootstrap_schema(self):
        """Warm up all pipeline backends and derive per-level column names.

        Two reasons this must run before the first real tick:
        - psutil.cpu_percent() returns 0.0 on its first call per process object;
          IO counters need a baseline snapshot to compute rates.
        - Column names are not known statically — they depend on how many CPUs/GPUs
          are present. A dry collect() pass discovers them so NodeDataStore can
          pre-allocate the schema before any data arrives.
        """
        bootstrap_context: CollectionContext = {
            "process_pids": set(),
            "user_pids": set(),
            "slurm_pids": set(),
            "cpu": {},
            "rss": {},
            "io": {},
        }
        for backend, _ in self._pipeline:
            backend.snapshot(bootstrap_context)
        columns_by_level: dict[str, list[str]] = {}
        for level in self.levels:
            row: dict = {"time": 0.0}
            for collector, handler in self._pipeline:
                try:
                    row.update(handler.transform(collector.collect(level, bootstrap_context), level))
                except Exception:
                    pass
            columns_by_level[level] = list(row.keys())
        self.nodes.init_node_schema("local", columns_by_level)

    def _validate_level(self, level: str):
        if level not in self.levels:
            raise ValueError(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_LEVEL
                ].format(level=level, levels=self.levels)
            )

    def _collect_metrics(self) -> list[dict[str, float]]:
        """Collect one sample per level; return a list of flat dicts."""
        context: CollectionContext = {
            "process_pids": set(),
            "user_pids": set(),
            "slurm_pids": set(),
            "cpu": {},
            "rss": {},
            "io": {},
        }
        for backend, _ in self._pipeline:
            backend.snapshot(context)

        time_mark = time.perf_counter()
        rows = []
        for level in self.levels:
            row: dict = {"time": time_mark}
            for collector, handler in self._pipeline:
                row.update(handler.transform(collector.collect(level, context), level))
            rows.append(row)
        return rows

    def _collect_data(self):
        """Collect metrics at a fixed cadence anchored to an absolute timeline.

        Uses ``threading.Event.wait`` instead of ``time.sleep`` so that:
        * each tick is scheduled relative to a fixed epoch, preventing
          per-iteration sleep-overshoot from accumulating into drift;
        * ``stop()`` can signal the event and wake the thread instantly
          instead of blocking up to one full interval on ``thread.join``.

        The GIL is released during ``Event.wait`` just like ``time.sleep``.
        """
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            rows = self._collect_metrics()
            for level, row in zip(self.levels, rows):
                self.nodes.add_sample("local", level, row)
            self.n_measurements += 1

            next_tick += self.interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                self.n_missed_measurements += 1
                next_tick = time.perf_counter()

    def start(self, interval: float = 1.0):
        if self.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.MONITOR_ALREADY_RUNNING]
            )
            return
        self.interval = interval
        self.start_time = time.perf_counter()
        self.wallclock_start_time = time.time()
        self.running = True
        self._stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._collect_data, daemon=True
        )
        self.monitor_thread.start()
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MONITOR_STARTED].format(
                pid=self.pid,
                interval=self.interval,
            )
        )

    def stop(self):
        self.running = False
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.stop_time = time.perf_counter()
        self.wallclock_stop_time = time.time()

        elapsed = self.stop_time - self.start_time
        expected = int(elapsed / self.interval) if self.interval > 0 else 0
        self.n_missed_measurements = max(0, expected - self.n_measurements)

        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MONITOR_STOPPED].format(
                seconds=elapsed
            )
        )
        if self.n_measurements > 0:
            logger.info(
                EXTENSION_INFO_MESSAGES[ExtensionInfoCode.MISSED_MEASUREMENTS].format(
                    perc_missed_measurements=(
                        self.n_missed_measurements / expected
                        if expected > 0 else 0
                    )
                )
            )
