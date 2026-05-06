"""Subprocess-based performance monitor.

Spawns the metric-collection loop in a child process so it runs
completely outside the main interpreter's GIL.  A lightweight reader
thread in the parent consumes the JSON-lines stream coming from the
child and feeds samples into the same
:class:`~jumper_extension.adapters.data.PerformanceData` container that
the threaded :class:`PerformanceMonitor` uses.

Implements :class:`MonitorProtocol` so it can be used as a drop-in
replacement everywhere the default monitor is accepted.
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

from jumper_extension.adapters.data import PerformanceData
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.utilities import get_available_levels

logger = logging.getLogger("extension")


class SubprocessPerformanceMonitor:
    """Performance monitor that delegates collection to a child process.

    Usage is identical to :class:`PerformanceMonitor`::

        monitor = SubprocessPerformanceMonitor()
        monitor.start(interval=1.0)
        # … CPU-heavy work in the main process …
        monitor.stop()

    Because the collection runs in a separate OS process, the GIL of the
    main interpreter cannot block it.
    """

    def __init__(self):
        # MonitorProtocol surface — populated from collector handshake
        self.interval: float = 1.0
        self.running: bool = False
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.wallclock_start_time: Optional[float] = None
        self.wallclock_stop_time: Optional[float] = None
        self.num_cpus: int = 0
        self.num_system_cpus: int = 0
        self.num_gpus: int = 0
        self.gpu_memory: float = 0.0
        self.gpu_name: str = ""
        self.cpu_handles: list = []
        self.memory_limits: dict = {}

        self.data: Optional[PerformanceData] = None
        self.levels: List[str] = get_available_levels()

        self.n_measurements: int = 0
        self.n_missed_measurements: int = 0

        # session state
        self.is_imported: bool = False
        self.session_source: Optional[str] = None

        # internal
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, interval: float = 1.0) -> None:
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

        collector_cmd = self._build_agent_cmd(interval)

        self._process = subprocess.Popen(
            collector_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # own process group for clean kill
        )
        # Safety net: kill the collector if the parent exits without stop()
        atexit.register(self._cleanup_at_exit)

        # Wait for the "ready" handshake
        if not self._wait_for_ready():
            logger.error(
                "[JUmPER]: Subprocess monitor collector failed to start."
            )
            self._kill_process()
            return

        # Initialise data container with info from the collector
        self.data = PerformanceData(
            self.num_cpus, self.num_system_cpus, self.num_gpus
        )

        # Start the reader thread (lightweight — just JSON parsing + append)
        self.running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="jumper-subprocess-reader",
        )
        self._reader_thread.start()

        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.MONITOR_STARTED
            ].format(
                pid=self._process.pid,
                interval=self.interval,
            )
        )

    def stop(self) -> None:
        self.running = False

        # Terminate the child process
        self._kill_process()

        # Wait for the reader thread to drain
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5.0)

        # Ensure the process is fully dead before touching stderr
        if self._process and self._process.poll() is None:
            self._process.kill()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

        # Surface any collector-side errors.
        # Use communicate() with a timeout to avoid blocking forever
        # on a full pipe buffer.
        if self._process:
            try:
                _, err = self._process.communicate(timeout=3)
                if err and err.strip():
                    logger.warning(
                        f"[JUmPER]: Collector stderr: {err.strip()}"
                    )
            except (subprocess.TimeoutExpired, ValueError, OSError):
                # Last resort: just close the pipes
                try:
                    self._process.kill()
                except OSError:
                    pass
                try:
                    if self._process.stderr:
                        self._process.stderr.close()
                    if self._process.stdout:
                        self._process.stdout.close()
                except OSError:
                    pass

        self.stop_time = time.perf_counter()
        self.wallclock_stop_time = time.time()

        # Compute missed measurements from elapsed time vs actual samples
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
                EXTENSION_INFO_MESSAGES[
                    ExtensionInfoCode.MISSED_MEASUREMENTS
                ].format(
                    perc_missed_measurements=(
                        self.n_missed_measurements / expected
                        if expected > 0 else 0
                    )
                )
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_ready(self, max_attempts: int = 50) -> bool:
        """Block until the collector sends a ``{"status": "ready", …}`` line."""
        for _ in range(max_attempts):
            line = self._read_line()
            if line is None:
                time.sleep(0.1)
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("status") == "ready":
                self.num_cpus = msg.get("num_cpus", 0)
                self.num_system_cpus = msg.get("num_system_cpus", 0)
                self.num_gpus = msg.get("num_gpus", 0)
                self.gpu_memory = msg.get("gpu_memory", 0.0)
                self.gpu_name = msg.get("gpu_name", "")
                self.cpu_handles = msg.get("cpu_handles", [])
                self.memory_limits = msg.get("memory_limits", {})
                self.levels = msg.get("levels", self.levels)
                return True

            if msg.get("status") == "error":
                logger.error(
                    f"[JUmPER]: Collector error: {msg.get('error', '?')}"
                )
                return False
        return False

    def _read_line(self) -> Optional[str]:
        if self._process and self._process.stdout:
            try:
                line = self._process.stdout.readline()
                if line:
                    return line.strip()
            except (ValueError, OSError):
                pass
        return None

    def _reader_loop(self) -> None:
        """Continuously read JSON samples from the collector's stdout."""
        while self.running:
            line = self._read_line()
            if not line:
                if self.running:
                    # Collector exited unexpectedly
                    poll = self._process.poll() if self._process else None
                    if poll is not None:
                        logger.warning(
                            "[JUmPER]: Subprocess collector exited "
                            f"(code={poll})."
                        )
                        break
                    time.sleep(0.01)
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "sample" not in msg:
                continue

            sample = msg["sample"]
            level = msg.get("level", "process")

            try:
                self.data.add_sample(
                    level,
                    msg.get("time", 0.0),
                    sample.get("cpu_util", []),
                    sample.get("memory", 0.0),
                    sample.get("gpu_util", []),
                    sample.get("gpu_band", []),
                    sample.get("gpu_mem", []),
                    sample.get("io_counters", [0, 0, 0, 0]),
                )
                self.n_measurements += 1
            except Exception as exc:
                logger.warning(
                    f"[JUmPER]: Failed to add sample: {exc}"
                )

    def _build_agent_cmd(self, interval: float) -> str:
        """Return the shell command that launches the monitoring collector.

        The collector process attempts to elevate its own scheduling
        priority via ``os.nice(-10)`` at startup (see
        :func:`~jumper_extension.monitor.backends.subprocess_python._collector._run_collector`).
        This helps the monitor keep up with the requested sampling
        frequency even when all CPU cores are fully saturated by compute
        tasks.  If the current user lacks ``CAP_SYS_NICE`` the attempt
        is silently ignored and the collector instead lowers the priority
        of the target process tree (including the root) to nice +15
        (which requires no special privileges), giving itself a relative
        advantage.  Renicing the root ensures future children inherit
        the lowered priority automatically.
        """
        levels_arg = ""
        if self.levels:
            levels_arg = f" --levels {','.join(self.levels)}"
        return (
            f"{sys.executable} -m"
            f" jumper_extension.monitor.backends.subprocess_python._collector"
            f" --interval {interval}"
            f" --target-pid {os.getpid()}"
            f"{levels_arg}"
        )

    def _cleanup_at_exit(self) -> None:
        """atexit handler — kill the collector if still alive."""
        self._kill_process()

    def _kill_process(self) -> None:
        if self._process and self._process.poll() is None:
            pgid = None
            try:
                pgid = os.getpgid(self._process.pid)
            except OSError:
                pass
            # Kill the entire process group (shell + collector binary)
            if pgid is not None and pgid != os.getpgid(0):
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except OSError:
                    pass
            else:
                try:
                    self._process.terminate()
                except OSError:
                    pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # SIGTERM didn't work — escalate to SIGKILL
                if pgid is not None and pgid != os.getpgid(0):
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        pass
                else:
                    try:
                        self._process.kill()
                    except OSError:
                        pass
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

