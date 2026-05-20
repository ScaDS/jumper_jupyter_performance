from abc import abstractmethod
from typing import Any, Callable, Optional

import psutil

from jumper_extension.monitor.metrics.common import CollectorBackend
from jumper_extension.monitor.metrics.context import CollectionContext


class ProcessCollectorBackend(CollectorBackend):
    """Base for process enumeration backends.

    Serves a dual role: populates the shared :class:`CollectionContext` via
    :meth:`snapshot` so other backends can avoid redundant syscalls, and
    exposes process-filtering utilities used by GPU backends.
    """

    name = "process-base"

    def __init__(self, pid: int, process: psutil.Process, uid: int, slurm_job: str):
        self._pid = pid
        self._process = process
        self._uid = uid
        self._slurm_job = slurm_job

    @abstractmethod
    def get_process_pids(self) -> set[int]: ...

    @abstractmethod
    def snapshot(self, context: CollectionContext) -> None: ...

    @abstractmethod
    def collect(self, level: str, context: CollectionContext) -> None: ...

    @abstractmethod
    def filter_process(self, proc: psutil.Process, mode: str) -> bool: ...

    @abstractmethod
    def get_filtered_processes(
        self,
        level: str = "user",
        mode: str = "cpu",
        handle: Optional[object] = None,
    ) -> list[psutil.Process]: ...

    @abstractmethod
    def safe_proc_call(
        self,
        proc,
        proc_func: Callable[[psutil.Process], Any],
        default=0,
    ) -> Any: ...
