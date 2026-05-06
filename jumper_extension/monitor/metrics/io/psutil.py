import psutil

from jumper_extension.monitor.metrics.io.common import IoBackend


class PsutilIoBackend(IoBackend):
    """I/O backend implemented via psutil."""

    name = "io-psutil"

    def _add_io(self, totals, io_data):
        if io_data:
            totals[0] += io_data.read_count
            totals[1] += io_data.write_count
            totals[2] += io_data.read_bytes
            totals[3] += io_data.write_bytes

    def collect(self, level: str = "process") -> list[int]:
        self._m._validate_level(level)
        snap = self._m._process_backend._snap_io
        totals = [0, 0, 0, 0]
        if level == "process":
            for pid in self._m.process_pids:
                self._add_io(totals, snap.get(pid))
        elif level == "system":
            # Use disk_io_counters for a single-syscall system total
            dio = psutil.disk_io_counters()
            if dio:
                totals = [
                    dio.read_count, dio.write_count,
                    dio.read_bytes, dio.write_bytes,
                ]
        elif level == "user":
            user_pids = set(self._m.process_pids)
            user_pids.update(
                p.pid for p in self._m._process_backend._snap_user_procs
            )
            for pid in user_pids:
                self._add_io(totals, snap.get(pid))
        else:  # slurm
            slurm_pids = set(self._m.process_pids)
            slurm_pids.update(
                p.pid for p in self._m._process_backend._snap_slurm_procs
            )
            for pid in slurm_pids:
                self._add_io(totals, snap.get(pid))
        return totals
