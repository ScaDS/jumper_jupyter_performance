"""SLURM multi-node performance monitor.

Orchestrates per-node monitoring collectors over SSH, collects their
streamed JSON samples, writes them to a log file, and exposes the
same :class:`MonitorProtocol` surface as the default single-node
:class:`PerformanceMonitor` so it can be plugged into the existing
service layer.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

from jumper_extension.adapters.data import NodeInfo, NodeDataStore
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
        self.info: dict = {}

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

        # MonitorProtocol surface
        self.interval: float = 1.0
        self.running: bool = False
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.wallclock_start_time: Optional[float] = None
        self.wallclock_stop_time: Optional[float] = None
        self.is_imported: bool = False
        self.session_source: Optional[str] = None
        self.nodes: NodeDataStore = NodeDataStore()
        self.levels: List[str] = get_available_levels()

        # Internal
        self._node_hostnames: List[str] = []
        self._connections: Dict[str, _NodeConnection] = {}
        self._reader_threads: List[threading.Thread] = []
        self._log_writer = MultinodeLogWriter(log_path)

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
            self._node_hostnames = get_slurm_nodes()
        except RuntimeError as exc:
            logger.error(f"[JUmPER]: {exc}")
            return

        if not self._node_hostnames:
            logger.error("[JUmPER]: No SLURM nodes discovered.")
            return

        # Open log file
        self._log_writer.open()

        # Launch collectors on all nodes
        for hostname in self._node_hostnames:
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
                            node_info = NodeInfo(
                                node=hostname,
                                num_cpus=msg.get("num_cpus", 0),
                                num_system_cpus=msg.get("num_system_cpus", 0),
                                num_gpus=msg.get("num_gpus", 0),
                                gpu_memory=msg.get("gpu_memory", 0.0),
                                gpu_name=msg.get("gpu_name", ""),
                                memory_limits=msg.get("memory_limits", {}),
                                cpu_handles=msg.get("cpu_handles", []),
                            )
                            self.nodes.register_node(node_info)
                            columns_by_level = msg.get("columns_by_level", {})
                            if columns_by_level:
                                self.nodes.init_node_schema(hostname, columns_by_level)
                            self.levels = msg.get("levels", self.levels)
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
            f"{ready_count}/{len(self._node_hostnames)} nodes, "
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

            sample = msg["sample"]  # flat dict with "time" and all metric columns
            level = msg.get("level", "process")
            wallclock = msg.get("wallclock", time.time())

            # Remap the "time" column to the head-node perf_counter basis so
            # that filter_perfdata can align samples with cell_history.start_time.
            time_mark = (self.start_time or 0.0) + (
                wallclock - (self.wallclock_start_time or wallclock)
            )
            row = {**sample, "time": time_mark}

            try:
                self.nodes.add_sample(hostname, level, row)
            except Exception as exc:
                logger.warning(f"[JUmPER]: Failed to add sample from {hostname}: {exc}")

            # Write to log file (best-effort; keep for offline analysis)
            try:
                self._log_writer.write_sample(
                    node=hostname,
                    level=level,
                    wallclock=wallclock,
                    perf_time=sample.get("time", 0.0),
                    cpu_util=sample.get("cpu_util_avg", 0.0),
                    memory=sample.get("memory", 0.0),
                    gpu_util=sample.get("gpu_util_avg", 0.0),
                    gpu_band=sample.get("gpu_band_avg", 0.0),
                    gpu_mem=sample.get("gpu_mem_avg", 0.0),
                    io_counters=[
                        sample.get("io_read_count", 0.0),
                        sample.get("io_write_count", 0.0),
                        sample.get("io_read", 0.0),
                        sample.get("io_write", 0.0),
                    ],
                )
            except Exception:
                pass
