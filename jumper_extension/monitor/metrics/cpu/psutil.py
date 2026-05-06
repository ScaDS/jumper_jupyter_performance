import psutil

from jumper_extension.monitor.metrics.cpu.common import CpuBackend


class PsutilCpuBackend(CpuBackend):
    """CPU backend implemented via psutil."""

    name = "cpu-psutil"

    def collect(self, level: str = "process") -> list[float]:
        self._m._validate_level(level)
        snap = self._m._process_backend._snap_cpu
        if level == "system":
            cpu_util_per_core = psutil.cpu_percent(percpu=True)
            return cpu_util_per_core
        elif level == "process":
            cpu_total = sum(
                snap.get(pid, 0.0) for pid in self._m.process_pids
            )
            return [cpu_total / self._m.num_cpus] * self._m.num_cpus
        elif level == "user":
            # All process-level PIDs belong to this user, plus extras
            user_pids = set(self._m.process_pids)
            user_pids.update(
                p.pid for p in self._m._process_backend._snap_user_procs
            )
            cpu_total = sum(snap.get(pid, 0.0) for pid in user_pids)
            return [cpu_total / self._m.num_cpus] * self._m.num_cpus
        else:  # slurm
            slurm_pids = set(self._m.process_pids)
            slurm_pids.update(
                p.pid for p in self._m._process_backend._snap_slurm_procs
            )
            cpu_total = sum(snap.get(pid, 0.0) for pid in slurm_pids)
            return [cpu_total / self._m.num_cpus] * self._m.num_cpus
