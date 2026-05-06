import psutil

from jumper_extension.monitor.metrics.memory.common import MemoryBackend


class PsutilMemoryBackend(MemoryBackend):
    """Memory backend implemented via psutil."""

    name = "memory-psutil"

    def collect(self, level: str = "process") -> float:
        self._m._validate_level(level)
        snap = self._m._process_backend._snap_rss
        if level == "system":
            return (
                psutil.virtual_memory().total
                - psutil.virtual_memory().available
            ) / (1024**3)
        elif level == "process":
            memory_total = sum(
                snap.get(pid, 0) for pid in self._m.process_pids
            )
            return memory_total / (1024**3)
        elif level == "user":
            user_pids = set(self._m.process_pids)
            user_pids.update(
                p.pid for p in self._m._process_backend._snap_user_procs
            )
            memory_total = sum(snap.get(pid, 0) for pid in user_pids)
            return memory_total / (1024**3)
        else:  # slurm
            slurm_pids = set(self._m.process_pids)
            slurm_pids.update(
                p.pid for p in self._m._process_backend._snap_slurm_procs
            )
            memory_total = sum(snap.get(pid, 0) for pid in slurm_pids)
            return memory_total / (1024**3)
