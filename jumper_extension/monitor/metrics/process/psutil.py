import unittest.mock
from typing import Any, Callable, Optional

import psutil

from jumper_extension.utilities import is_slurm_available
from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.process.common import ProcessCollectorBackend


class PsutilProcessCollector(ProcessCollectorBackend):
    """Process backend implemented via psutil."""

    name = "process-psutil"

    def setup(self) -> None:
        self._process_cache: dict[int, psutil.Process] = {}

    def _get_or_create_process(self, pid: int) -> psutil.Process:
        """Return a cached Process object for *pid*, creating one if needed."""
        proc = self._process_cache.get(pid)
        if proc is None:
            proc = psutil.Process(pid)
            self._process_cache[pid] = proc
        return proc

    def get_process_pids(self) -> set[int]:
        """Get current process PID and all its children PIDs."""
        pids = {self._pid}
        try:
            pids.update(
                child.pid for child in self._process.children(recursive=True)
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        # prune cache: drop PIDs that are no longer alive
        self._process_cache = {
            p: obj for p, obj in self._process_cache.items() if p in pids
        }
        return pids

    def snapshot(self, context: CollectionContext) -> None:
        """Compute process_pids then collect cpu/memory/io for every known PID.

        Populates context["process_pids"], context["cpu"], context["rss"],
        context["io"], context["user_pids"], and context["slurm_pids"].
        """
        context["process_pids"] = self.get_process_pids()

        cpu = context["cpu"]
        rss = context["rss"]
        io = context["io"]
        process_pids = context["process_pids"]

        # 1) Snapshot process-level PIDs
        for pid in process_pids:
            proc = self._get_or_create_process(pid)
            try:
                cpu[pid] = proc.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu[pid] = 0.0
            try:
                rss[pid] = proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss[pid] = 0
            try:
                io[pid] = proc.io_counters()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                io[pid] = None

        # 2) Snapshot user/slurm-filtered processes (only the ones
        #    not already covered by the process-level set above)
        user_pids = set(process_pids)
        slurm_pids = set(process_pids)
        try:
            for proc in psutil.process_iter(["pid", "uids"]):
                pid = proc.pid
                if pid in cpu:
                    continue  # already collected above
                try:
                    is_user = proc.uids().real == self._uid
                except (psutil.AccessDenied, psutil.NoSuchProcess, IndexError):
                    is_user = False
                if not is_user:
                    continue
                user_pids.add(pid)
                # Collect metrics for this PID too
                try:
                    cpu[pid] = proc.cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    cpu[pid] = 0.0
                try:
                    rss[pid] = proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    rss[pid] = 0
                try:
                    io[pid] = proc.io_counters()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    io[pid] = None
                # Check slurm membership
                if is_slurm_available():
                    try:
                        if proc.environ().get("SLURM_JOB_ID") == str(
                            self._slurm_job
                        ):
                            slurm_pids.add(pid)
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            pass

        context["user_pids"] = user_pids
        context["slurm_pids"] = slurm_pids

    def collect(self, level: str, context: CollectionContext) -> None:
        return None

    def filter_process(self, proc: psutil.Process, mode: str) -> bool:
        """Check if process matches the filtering mode."""
        try:
            if mode == "user":
                return proc.uids().real == self._uid
            elif mode == "slurm":
                if not is_slurm_available():
                    return False
                return proc.environ().get("SLURM_JOB_ID") == str(
                    self._slurm_job
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
        return False

    def get_filtered_processes(
        self,
        level: str = "user",
        mode: str = "cpu",
        handle: Optional[object] = None,
    ) -> list[psutil.Process]:
        """Get filtered processes for CPU monitoring."""
        if mode == "cpu":
            return [
                proc
                for proc in psutil.process_iter(["pid", "uids"])
                if self.safe_proc_call(
                    proc, lambda p: self.filter_process(p, level), False
                )
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def safe_proc_call(
        self,
        proc,
        proc_func: Callable[[psutil.Process], Any],
        default=0,
    ) -> Any:
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
