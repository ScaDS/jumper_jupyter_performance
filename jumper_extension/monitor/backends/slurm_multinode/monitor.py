"""SLURM multi-node performance monitor.

Orchestrates per-node monitoring collectors over SSH, collects their
streamed JSON samples, writes them to a log file, and exposes the
same :class:`MonitorProtocol` surface as the default single-node
:class:`PerformanceMonitor` so it can be plugged into the existing
service layer.
"""

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
from jumper_extension.monitor.backends.slurm_multinode._log_writer import MultinodeLogWriter
from jumper_extension.monitor.backends.slurm_multinode._node_discovery import get_slurm_nodes
from jumper_extension.utilities import get_available_levels

logger = logging.getLogger("extension")


class _NodeConnection:
    """Manages an srun connection to a single remote node."""

    def __init__(self, hostname: str, python_executable: str):
        self.hostname = hostname
        self.python_executable = python_executable
        self.process: Optional[subprocess.Popen] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.ready = False
        self.info: Dict = {}

    def start(self, interval: float, levels: Optional[List[str]] = None) -> None:
        """Launch the remote collector via srun."""
        levels_arg = ""
        if levels:
            levels_arg = f" --levels {','.join(levels)}"

        collector_cmd = (
            f"{self.python_executable} -m"
            f" jumper_extension.monitor.backends.slurm_multinode._collector"
            f" --interval {interval}{levels_arg}"
        )

        # Use srun to launch the collector on the specific node
        srun_cmd = [
            "srun",
            "--nodelist=" + self.hostname,
            "--ntasks=1",
            "--unbuffered",
            "bash", "-c", collector_cmd
        ]

        logger.info(f"[JUmPER]: Launching collector on {self.hostname} via srun")
        logger.debug(f"[JUmPER]: srun command: {' '.join(srun_cmd)}")
        
        self.process = subprocess.Popen(
            srun_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Check if the process started successfully
        try:
            # Wait a brief moment to see if process exits immediately
            return_code = self.process.poll()
            if return_code is not None:
                # Process already exited, capture error output
                stderr_output = self.process.stderr.read()
                stdout_output = self.process.stdout.read()
                logger.error(f"[JUmPER]: srun process on {self.hostname} exited immediately with code {return_code}")
                if stderr_output:
                    logger.error(f"[JUmPER]: srun stderr: {stderr_output}")
                if stdout_output:
                    logger.error(f"[JUmPER]: srun stdout: {stdout_output}")
        except Exception as e:
            logger.warning(f"[JUmPER]: Could not check srun process status on {self.hostname}: {e}")

    def stop(self) -> None:
        """Terminate the remote collector."""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            logger.info(f"[JUmPER]: Collector on {self.hostname} stopped.")

    def read_line(self) -> Optional[str]:
        """Read one line from the collector's stdout (blocking)."""
        if self.process and self.process.stdout:
            try:
                line = self.process.stdout.readline()
                if line:
                    return line.strip()
            except (ValueError, OSError):
                pass
        return None


class SlurmMultinodeMonitor:
    """Multi-node monitor that satisfies ``MonitorProtocol``.

    Usage::

        monitor = SlurmMultinodeMonitor()
        monitor.start(interval=1.0)
        # … run workload …
        monitor.stop()

    On ``start()`` it discovers the SLURM nodes, SSHes into each one,
    launches the collector, and starts reader threads that feed samples
    into a JSON-Lines log file.

    The local (head) node is **also** monitored by launching a collector
    in-process.

    Attributes required by ``MonitorProtocol`` are provided by
    aggregating information from all connected nodes.
    """

    def __init__(
        self,
        log_path: str = "jumper_multinode.jsonl",
        python_executable: Optional[str] = None,
    ):
        # Resolve the Python interpreter used on remote nodes
        self._python_executable = python_executable or sys.executable

        # MonitorProtocol surface (aggregated from nodes)
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
        self.is_imported: bool = False
        self.session_source: Optional[str] = None

        # Internal
        self._nodes: List[str] = []
        self._connections: Dict[str, _NodeConnection] = {}
        self._reader_threads: List[threading.Thread] = []
        self._log_writer = MultinodeLogWriter(log_path)

        # Data container — initialised once we know hardware from collectors
        self.data: Optional[PerformanceData] = None
        self.levels = get_available_levels()

        # Per-node metadata collected from collector "ready" messages
        self.node_info: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, interval: float = 1.0) -> None:
        """Discover nodes, SSH into each, launch collectors, start readers."""
        if self.running:
            logger.warning("[JUmPER]: Multinode monitor is already running.")
            return

        self.interval = interval
        self.start_time = time.perf_counter()
        self.wallclock_start_time = time.time()

        # Discover nodes
        try:
            self._nodes = get_slurm_nodes()
        except RuntimeError as exc:
            logger.error(f"[JUmPER]: {exc}")
            return

        if not self._nodes:
            logger.error("[JUmPER]: No SLURM nodes discovered.")
            return

        # Open log file
        self._log_writer.open()

        # Launch collectors on all nodes
        for hostname in self._nodes:
            conn = _NodeConnection(hostname, self._python_executable)
            conn.start(interval, self.levels)
            self._connections[hostname] = conn

        # Wait for "ready" handshake from each collector
        for hostname, conn in self._connections.items():
            ready_received = False
            max_attempts = 10  # Prevent infinite loop
            attempts = 0
            
            while not ready_received and attempts < max_attempts:
                line = conn.read_line()
                if line:
                    try:
                        msg = json.loads(line)
                        if msg.get("status") == "ready":
                            conn.ready = True
                            conn.info = msg
                            self.node_info[hostname] = msg
                            logger.info(
                                f"[JUmPER]: Collector on {hostname} ready "
                                f"(cpus={msg.get('num_cpus')}, "
                                f"gpus={msg.get('num_gpus')})"
                            )
                            ready_received = True
                        elif msg.get("status") == "error":
                            error = msg.get("error", "Unknown error")
                            logger.error(
                                f"[JUmPER]: Collector on {hostname} failed to initialize: {error}"
                            )
                            # Don't wait for ready message if collector failed
                            break
                        else:
                            # Got JSON but not a ready/error message, continue reading
                            logger.debug(f"[JUmPER]: Non-ready JSON from {hostname}: {line}")
                    except json.JSONDecodeError:
                        # Not JSON, likely a log message, ignore and continue
                        logger.debug(f"[JUmPER]: Ignoring non-JSON line from {hostname}: {line}")
                else:
                    # No line received, wait a bit
                    time.sleep(0.1)
                
                attempts += 1
            
            if not ready_received:
                logger.warning(f"[JUmPER]: Failed to receive ready message from {hostname} after {max_attempts} attempts")

        # Aggregate hardware info from first responding node for Protocol
        self._aggregate_hardware_info()

        # Initialise data container
        self.data = PerformanceData(
            self.num_cpus, self.num_system_cpus, self.num_gpus
        )

        # Start reader threads
        self.running = True
        for hostname, conn in self._connections.items():
            if conn.ready:
                t = threading.Thread(
                    target=self._reader_loop,
                    args=(hostname, conn),
                    daemon=True,
                    name=f"jumper-reader-{hostname}",
                )
                t.start()
                self._reader_threads.append(t)

        ready_count = sum(1 for c in self._connections.values() if c.ready)
        logger.info(
            f"[JUmPER]: Multinode monitor started on "
            f"{ready_count}/{len(self._nodes)} nodes, "
            f"interval={interval}s, "
            f"log={self._log_writer.log_path}"
        )

    def stop(self) -> None:
        """Stop all collectors and close the log file."""
        self.running = False

        # Terminate remote collectors
        for conn in self._connections.values():
            conn.stop()

        # Wait for reader threads
        for t in self._reader_threads:
            t.join(timeout=5)
        self._reader_threads.clear()

        self.stop_time = time.perf_counter()
        self.wallclock_stop_time = time.time()

        # Close log
        self._log_writer.close()

        elapsed = self.stop_time - (self.start_time or self.stop_time)
        logger.info(
            f"[JUmPER]: Multinode monitor stopped after {elapsed:.1f}s."
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _aggregate_hardware_info(self) -> None:
        """Aggregate hardware info from all connected nodes.

        For Protocol compatibility we pick representative values;
        the per-node detail lives in ``self.node_info``.
        """
        total_cpus = 0
        total_system_cpus = 0
        total_gpus = 0
        gpu_memory = 0.0
        gpu_name = ""

        for info in self.node_info.values():
            total_cpus += info.get("num_cpus", 0)
            total_system_cpus += info.get("num_system_cpus", 0)
            total_gpus += info.get("num_gpus", 0)
            gpu_memory = max(gpu_memory, info.get("gpu_memory", 0.0))
            gpu_name = info.get("gpu_name", "") or gpu_name

        self.num_cpus = total_cpus
        self.num_system_cpus = total_system_cpus
        self.num_gpus = total_gpus
        self.gpu_memory = gpu_memory
        self.gpu_name = gpu_name

        # Memory limits — per node, keyed by "node:level"
        self.memory_limits = {
            level: 0.0 for level in self.levels
        }

    def _reader_loop(self, hostname: str, conn: _NodeConnection) -> None:
        """Continuously read JSON samples from a node's collector."""
        line_count = 0
        while self.running:
            line = conn.read_line()
            if not line:
                # Collector exited or SSH closed
                if self.running:
                    logger.warning(
                        f"[JUmPER]: Collector on {hostname} disconnected after {line_count} lines."
                    )
                break

            line_count += 1
            logger.debug(f"[JUmPER]: Line {line_count} from {hostname}: {line}")

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(
                    f"[JUmPER]: Invalid JSON from {hostname}: {line}"
                )
                continue

            # Skip non-sample messages
            if "sample" not in msg:
                logger.debug(f"[JUmPER]: Non-sample message from {hostname}: {msg}")
                continue

            sample = msg["sample"]
            level = msg.get("level", "process")
            perf_time = msg.get("time", 0.0)
            wallclock = msg.get("wallclock", time.time())

            # Write to log file
            self._log_writer.write_sample(
                node=hostname,
                level=level,
                wallclock=wallclock,
                perf_time=perf_time,
                cpu_util=sample.get("cpu_util", []),
                memory=sample.get("memory", 0.0),
                gpu_util=sample.get("gpu_util", []),
                gpu_band=sample.get("gpu_band", []),
                gpu_mem=sample.get("gpu_mem", []),
                io_counters=sample.get("io_counters", [0, 0, 0, 0]),
            )
