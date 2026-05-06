import logging
import os
import threading
import time
from typing import Optional

import psutil

from jumper_extension.adapters.data import PerformanceData
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.monitor.metrics.cpu.psutil import PsutilCpuBackend
from jumper_extension.monitor.metrics.gpu.common import GpuBackendDiscovery
from jumper_extension.monitor.metrics.io.psutil import PsutilIoBackend
from jumper_extension.monitor.metrics.memory.psutil import PsutilMemoryBackend
from jumper_extension.monitor.metrics.process.psutil import PsutilProcessBackend
from jumper_extension.utilities import detect_memory_limit, get_available_levels

logger = logging.getLogger("extension")


class PerformanceMonitor:
    def __init__(self):
        self.interval = 1.0
        self.running = False
        self.start_time = None
        self.stop_time = None
        self.wallclock_start_time = None
        self.wallclock_stop_time = None
        self.monitor_thread = None
        self.process = psutil.Process()
        self.n_measurements = 0
        self.n_missed_measurements = 0
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
        self.pid = os.getpid()
        self.uid = os.getuid()
        self.slurm_job = os.environ.get("SLURM_JOB_ID", 0)
        self.levels = get_available_levels()
        self.process_pids = []

        self.memory_limits = {
            level: detect_memory_limit(level, self.uid, self.slurm_job)
            for level in self.levels
        }

        self._process_backend = PsutilProcessBackend(self)
        self._cpu_backend = PsutilCpuBackend(self)
        self._memory_backend = PsutilMemoryBackend(self)
        self._io_backend = PsutilIoBackend(self)
        for backend in (
            self._process_backend,
            self._cpu_backend,
            self._memory_backend,
            self._io_backend,
        ):
            backend.setup()

        self.nvidia_gpu_handles = []
        self.amd_gpu_handles = []
        self.gpu_memory = 0
        self.gpu_name = ""
        self._gpu_backends = GpuBackendDiscovery(self).discover()
        for backend in self._gpu_backends:
            backend.setup()
        self.num_gpus = len(self.nvidia_gpu_handles) + len(
            self.amd_gpu_handles
        )
        self.metrics = [
            "cpu",
            "memory",
            "io_read",
            "io_write",
            "io_read_count",
            "io_write_count",
        ]

        if self.num_gpus:
            self.metrics.extend(["gpu_util", "gpu_band", "gpu_mem"])

        self.data = PerformanceData(
            self.num_cpus, self.num_system_cpus, self.num_gpus
        )
        # session state
        self.is_imported = False
        self.session_source = None

    def _get_process_pids(self):
        return self._process_backend.get_process_pids()

    def _validate_level(self, level):
        if level not in self.levels:
            raise ValueError(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_LEVEL
                ].format(level=level, levels=self.levels)
            )

    def _filter_process(self, proc, mode):
        return self._process_backend.filter_process(proc, mode)

    def _get_filtered_processes(self, level="user", mode="cpu", handle=None):
        return self._process_backend.get_filtered_processes(
            level, mode, handle
        )

    def _safe_proc_call(self, proc, proc_func, default=0):
        return self._process_backend.safe_proc_call(proc, proc_func, default)

    def _collect_cpu(self, level="process"):
        return self._cpu_backend.collect(level)

    def _collect_memory(self, level="process"):
        return self._memory_backend.collect(level)

    def _collect_io(self, level="process"):
        return self._io_backend.collect(level)

    def _collect_gpu(self, level="process"):
        if self.num_gpus == 0:
            return [], [], []

        self._validate_level(level)
        gpu_util, gpu_band, gpu_mem = [], [], []

        for backend in self._gpu_backends:
            b_util, b_band, b_mem = backend.collect(level)
            gpu_util.extend(b_util)
            gpu_band.extend(b_band)
            gpu_mem.extend(b_mem)

        return gpu_util, gpu_band, gpu_mem


    def _collect_metrics(self):
        # Snapshot all per-PID metrics once; backends read from the cache.
        self._process_backend.snapshot_metrics()
        time_mark = time.perf_counter()
        return tuple(
            (
                time_mark,
                self._collect_cpu(level),
                self._collect_memory(level),
                *self._collect_gpu(level),
                self._collect_io(level),
            )
            for level in self.levels
        )

    def _collect_data(self):
        """Collect metrics at a fixed cadence anchored to an absolute timeline.

        Uses ``threading.Event.wait`` instead of ``time.sleep`` so that:
        * each tick is scheduled relative to a fixed epoch, preventing
          per-iteration sleep-overshoot from accumulating into drift;
        * ``stop()`` can signal the event and wake the thread instantly
          instead of blocking up to one full interval on ``thread.join``.

        The GIL is released during ``Event.wait`` just like ``time.sleep``.
        """
        next_tick = time.perf_counter()          # first tick: now
        while not self._stop_event.is_set():
            # A single tick must never be able to kill the collector
            # thread.  If it does, the exception propagates through
            # ``threading._bootstrap_inner`` → ``invoke_excepthook``,
            # which under heavy load (pandas/psutil state corruption,
            # fork churn from 256 burn workers, etc.) has been
            # observed to segfault the whole process.  Any exception
            # here is therefore logged and the tick is counted as
            # missed; the loop continues until ``stop()`` is called.
            # ``BaseException`` is deliberately excluded so that
            # ``KeyboardInterrupt`` / ``SystemExit`` still propagate.
            try:
                self.process_pids = self._get_process_pids()
                metrics = self._collect_metrics()
                for level, data_tuple in zip(self.levels, metrics):
                    self.data.add_sample(level, *data_tuple)
                self.n_measurements += 1
            except Exception as exc:
                logger.warning(
                    "[JUmPER]: monitor tick failed (%s: %s); "
                    "skipping this sample",
                    type(exc).__name__, exc,
                )
                self.n_missed_measurements += 1

            # schedule next tick on the absolute timeline
            next_tick += self.interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                # we're behind schedule — skip forward to the next
                # achievable tick so we don't rapid-fire to "catch up"
                self.n_missed_measurements += 1
                next_tick = time.perf_counter()

    # ---- original implementation kept as backup ----
    def _collect_data_legacy(self):
        while self.running:
            time_start_measurement = time.perf_counter()
            self.process_pids = self._get_process_pids()
            metrics = self._collect_metrics()
            for level, data_tuple in zip(self.levels, metrics):
                self.data.add_sample(level, *data_tuple)
            time_measurement = time.perf_counter() - time_start_measurement
            self.n_measurements += 1
            if time_measurement > self.interval:
                """
                logger.warning(
                    EXTENSION_INFO_MESSAGES[
                        ExtensionInfoCode.IMPRECISE_INTERVAL
                    ].format(interval=self.interval),
                    end="\r",
                )
                """
                self.n_missed_measurements += 1
            else:
                time.sleep(self.interval - time_measurement)

    def start(self, interval: float = 1.0):
        if self.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.MONITOR_ALREADY_RUNNING
                ]
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
            # The collector may currently be inside _collect_metrics(),
            # which under heavy load (e.g. thousands of processes from a
            # 256-worker burn) can take well over a second to finish a
            # single tick.  Returning from stop() while the thread is
            # still inside ``data.add_sample(...)`` produces a data race
            # on the per-level DataFrames: any caller that subsequently
            # reads ``monitor.data`` can land in pandas concat at the
            # same moment the collector is reallocating blocks, which
            # segfaults inside ``concatenate_managers``.  Wait long
            # enough to cover even pathologically slow ticks; the
            # benchmark/service watchdog (SIGALRM / faulthandler) is
            # the real upper bound on hangs.
            self.monitor_thread.join(timeout=30.0)
            if self.monitor_thread.is_alive():
                logger.warning(
                    "[JUmPER]: Monitor thread did not terminate within "
                    "30s of stop(); leaving it running to avoid blocking, "
                    "but ``monitor.data`` is unsafe to read until it "
                    "exits."
                )
        self.stop_time = time.perf_counter()
        self.wallclock_stop_time = time.time()

        # Recompute missed measurements from elapsed time vs actual samples
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
