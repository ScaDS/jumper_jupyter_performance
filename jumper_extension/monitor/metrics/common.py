from __future__ import annotations

"""Public interfaces for building custom metric collectors.

To add a new metric source implement:
    1. :class:`CollectorBackend` subclass.
    2. matching :class:`StorageHandler` - to convert collector's raw output into a flat `dict[str, float]`
    3. register them both in ``collectors.yaml``.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from jumper_extension.monitor.metrics.context import CollectionContext

__all__ = ["CollectionContext", "CollectorBackend", "StorageHandler"]


class CollectorBackend(ABC):
    """Abstract base for all metric collector backends.

    A backend collects one category of metrics (CPU, memory, GPU, I/O,
    process state, or custom).  The pipeline calls the three lifecycle methods on every
    backend:

        setup()             — once at startup
        snapshot(context)   — once per tick, before any collect()
        collect(level, ctx) — once per tick per active level

    Subclass contract:
        - Define a unique ``name`` class attribute, e.g. ``"cpu-psutil"``.
        - Implement :meth:`collect`.
        - Override :meth:`setup` if collector specific setup is needed
        - Override :meth:`snapshot` if per-tick computation of metrics
        that can be shared with the other collectors is needed.
        - Pair with a :class:`StorageHandler` that converts the value
          returned by :meth:`collect` into a flat ``dict[str, float]``.

    Minimal example::

        class MyBackend(CollectorBackend):
            name = "my-metric"

            def collect(self, level: str, context: CollectionContext):
                return 42.0  # pair with ScalarHandler(column="my_value")
    """

    name: str

    def setup(self) -> dict | None:
        """Initialize resources.  Called once before collection starts.

        Override to acquire handles, open connections, or discover hardware.

        Returns:
            Optional metadata dict.  GPU backends use this to report
            ``{"num_gpus": n, "gpu_memory": f, "gpu_name": s}`` so the
            pipeline can populate :class:`NodeInfo` before deferred backends
            are built.  Return ``None`` (or omit the override) otherwise.
        """
        return None

    def snapshot(self, context: CollectionContext) -> None:
        """The idea: all backends that depend on or use the same metrics for their own computations
        can access shared per-tick ``CollectionContext``

        The pipeline calls ``snapshot()`` on *all* backends before calling
        ``collect()`` on each collector, so data written here is visible to
        every ``collect()`` call in the same tick.

        Example:
        The process backend uses this to enumerate live PIDs and fill
        ``context`` with per-PID cpu/rss/io counters so that other
         backends can read them.
        """
        return None

    @abstractmethod
    def collect(self, level: str, context: CollectionContext) -> Any:
        """Collect one sample for the given aggregation level.

        Args:
            level: Aggregation scope.  One of ``"system"``, ``"process"``,
                   ``"user"``, or ``"slurm"`` (see :class:`StorageHandler`
                   for full semantics).
            context: Shared state populated by :meth:`snapshot` earlier in
                     this tick.  Contains live PIDs and per-PID cpu/rss/io
                     snapshots.

        Returns:
            Raw sample value passed directly to the paired
            :class:`StorageHandler`.  The type must match what the handler
            expects — see the built-in handlers for standard pairings.
        """
        ...


class StorageHandler(Protocol):
    def transform(self, raw, level: str) -> dict[str, float]:
        """Return a flat column -> value dict for one sample.

        Args:
            raw: The value returned by any :class:`CollectorBackend`.
                 Intentionally untyped — each handler is tailored to
                  convert a type that corresponding collector passes into a flat dict.
                 See the built-in handlers for examples of each pairing.
            level: Aggregation scope for this sample.  One of:

                 ``"system"``  — all processes on the machine,
                 ``"process"`` — the monitored process and its children,
                 ``"user"``    — all processes owned by the current user,
                 ``"slurm"``   — all processes belonging to the current
                                 Slurm job (only present when running inside
                                 a Slurm allocation).

                 Stateful handlers (e.g. rate computation) use ``level`` as
                 a key to keep per-scope state separate.

        Returns:
            A flat ``{column_name: value}`` dict.  An empty dict is valid
            and means this handler contributes no columns for this sample.
        """
        ...
