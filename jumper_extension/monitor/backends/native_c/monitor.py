"""Native C collector performance monitor.

Inherits from :class:`SubprocessPerformanceMonitor` but launches a
compiled C binary (``jumper_collector``) instead of a Python child process.
"""

import os
from typing import List

from jumper_extension.monitor.backends.subprocess_python.monitor import (
    SubprocessPerformanceMonitor,
)


class CSubprocessPerformanceMonitor(SubprocessPerformanceMonitor):
    """Subprocess monitor that uses a compiled C collector.

    Identical to :class:`SubprocessPerformanceMonitor` but launches a
    native binary (``jumper_collector``) instead of a Python process.  This
    eliminates Python startup overhead and reduces per-tick collection
    time.

    The C collector reads metrics directly from ``/proc`` and speaks the
    same JSON-lines protocol.  NVIDIA GPU metrics are collected via
    dynamic loading of ``libnvidia-ml.so`` (no compile-time dependency).
    SLURM level is auto-detected from the target process's environment.

    The binary must be compiled first::

        make -C jumper_extension/monitor/backends/native_c/
    """

    _BINARY_NAME = "jumper_collector"

    def _build_agent_cmd(self, interval: float) -> str:
        """Return the shell command that launches the C collector binary.

        Like the Python variant, the C collector elevates its own
        scheduling priority via ``setpriority()`` at startup.  See
        :meth:`SubprocessPerformanceMonitor._build_agent_cmd` for
        rationale.
        """
        binary = os.path.join(
            os.path.dirname(__file__), self._BINARY_NAME
        )
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"C collector binary not found at {binary}.  "
                f"Run: make -C {os.path.dirname(__file__)}/"
            )
        levels_arg = ""
        if self.levels:
            levels_arg = f" --levels {','.join(self.levels)}"
        return (
            f"{binary}"
            f" --interval {interval}"
            f" --target-pid {os.getpid()}"
            f"{levels_arg}"
        )
