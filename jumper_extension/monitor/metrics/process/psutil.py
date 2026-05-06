import unittest.mock
from typing import Any, Callable, Optional

import psutil

from jumper_extension.utilities import is_slurm_available
from jumper_extension.monitor.metrics.process.common import ProcessBackend


class PsutilProcessBackend(ProcessBackend):
    """Process backend implemented via psutil."""

    name = "process-psutil"

    def setup(self) -> None:
        self._process_cache: dict[int, psutil.Process] = {}
        # Per-tick metric snapshot: populated once per tick by
        # snapshot_metrics(), consumed by CPU/memory/IO backends.
        self._snap_cpu: dict[int, float] = {}
        self._snap_rss: dict[int, int] = {}
        self._snap_io: dict[int, tuple] = {}
        # Cached filtered PID lists for user/slurm levels (per tick)
        self._snap_user_procs: list[psutil.Process] = []
        self._snap_slurm_procs: list[psutil.Process] = []

    def _get_or_create_process(self, pid: int) -> psutil.Process:
        """Return a cached Process object for *pid*, creating one if needed."""
        proc = self._process_cache.get(pid)
        if proc is None:
            proc = psutil.Process(pid)
            self._process_cache[pid] = proc
        return proc

    def get_process_pids(self) -> set[int]:
        """Get current process PID and all its children PIDs."""
        pids = {self._m.pid}
        try:
            pids.update(
                child.pid for child in self._m.process.children(recursive=True)
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        # prune cache: drop PIDs that are no longer alive
        self._process_cache = {
            p: obj for p, obj in self._process_cache.items() if p in pids
        }
        return pids

    def snapshot_metrics(self) -> None:
        """Collect cpu/memory/io for every known PID in one pass.

        Call this once per tick *after* :meth:`get_process_pids`.  The
        CPU, memory, and IO backends then read from the snapshot instead
        of issuing redundant per-PID syscalls for each level.
        """
        self._snap_cpu.clear()
        self._snap_rss.clear()
        self._snap_io.clear()

        # 1) Snapshot process-level PIDs
        for pid in self._m.process_pids:
            proc = self._get_or_create_process(pid)
            try:
                self._snap_cpu[pid] = proc.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._snap_cpu[pid] = 0.0
            try:
                self._snap_rss[pid] = proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._snap_rss[pid] = 0
            try:
                self._snap_io[pid] = proc.io_counters()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._snap_io[pid] = None

        # 2) Snapshot user/slurm-filtered processes (only the ones
        #    not already covered by the process-level set above)
        self._snap_user_procs = []
        self._snap_slurm_procs = []
        try:
            for proc in psutil.process_iter(["pid", "uids"]):
                pid = proc.pid
                if pid in self._snap_cpu:
                    continue  # already collected above
                try:
                    is_user = proc.uids().real == self._m.uid
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    is_user = False
                if not is_user:
                    continue
                self._snap_user_procs.append(proc)
                # Collect metrics for this PID too
                try:
                    self._snap_cpu[pid] = proc.cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self._snap_cpu[pid] = 0.0
                try:
                    self._snap_rss[pid] = proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self._snap_rss[pid] = 0
                try:
                    self._snap_io[pid] = proc.io_counters()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self._snap_io[pid] = None
                # Check slurm membership
                if is_slurm_available():
                    try:
                        if proc.environ().get("SLURM_JOB_ID") == str(
                            self._m.slurm_job
                        ):
                            self._snap_slurm_procs.append(proc)
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def filter_process(self, proc: psutil.Process, mode: str) -> bool:
        """Check if process matches the filtering mode."""
        try:
            if mode == "user":
                return proc.uids().real == self._m.uid
            elif mode == "slurm":
                if not is_slurm_available():
                    return False
                return proc.environ().get("SLURM_JOB_ID") == str(
                    self._m.slurm_job
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        return False

    def get_filtered_processes(
        self,
        level: str = "user",
        mode: str = "cpu",
        handle: Optional[object] = None,
    ):
        """Get filtered processes for CPU or GPU monitoring."""
        if mode == "cpu":
            return [
                proc
                for proc in psutil.process_iter(["pid", "uids"])
                if self.safe_proc_call(
                    proc, lambda p: self.filter_process(p, level), False
                )
            ]
        elif mode == "nvidia_gpu":
            try:
                import pynvml
            except ImportError:
                return [], []
            all_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            filtered = [
                p
                for p in all_procs
                if self.safe_proc_call(
                    p.pid,
                    lambda proc: self.filter_process(proc, level),
                    False,
                )
            ]
            return filtered, all_procs
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def safe_proc_call(
        self,
        proc,
        proc_func: Callable[[psutil.Process], Any],
        default=0,
    ):
        """Safely call a process method and return default on error."""
        try:
            if not isinstance(proc, psutil.Process):
                # proc might be a pid — use cache so objects persist
                # across ticks (needed for delta-based cpu_percent)
                proc = self._get_or_create_process(proc)
            result = proc_func(proc)
            return result if result is not None else default
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return default
        except TypeError:
            # in test case, where psutil is a mock
            if isinstance(psutil.Process, unittest.mock.MagicMock):
                return default
