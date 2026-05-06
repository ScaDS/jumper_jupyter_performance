# Custom Monitors

JUmPER's monitoring layer is fully pluggable. The service, reporter,
visualizer and session exporter all program against a single
structural interface — `MonitorProtocol` — defined in
`jumper_extension/monitor/common.py`. Any object that satisfies it can
be used as a drop-in replacement for the built-in backends.

This guide walks through:

1. [The `MonitorProtocol` surface](#the-monitorprotocol-surface)
2. [Plugging a custom monitor into the service](#plugging-a-custom-monitor-into-the-service)
3. [Worked example — the SLURM multi-node monitor](#worked-example-slurm-multi-node-monitor)
4. [Writing your own monitor](#writing-your-own-monitor)
5. [Interaction with `--check-sanity`](#interaction-with-check-sanity)

---

## The `MonitorProtocol` surface

A monitor is any object with the following attributes and methods:

```python
from typing import Optional, Protocol, runtime_checkable
from jumper_extension.adapters.data import PerformanceData


@runtime_checkable
class MonitorProtocol(Protocol):
    # Lifecycle
    interval: float
    running: bool
    def start(self, interval: float = 1.0) -> None: ...
    def stop(self) -> None: ...

    # Collected data
    data: PerformanceData

    # Timestamps (filled in by start()/stop())
    start_time: Optional[float]
    wallclock_start_time: Optional[float]
    wallclock_stop_time: Optional[float]

    # Hardware / context metadata
    num_cpus: int
    num_system_cpus: int
    num_gpus: int
    gpu_memory: float
    gpu_name: str
    cpu_handles: list[int]
    memory_limits: dict

    # Session state (set to False / None for live monitors)
    is_imported: bool
    session_source: Optional[str]
```

`PerformanceData` (see `jumper_extension/adapters/data.py`) is the
per-level in-memory container. Built-in monitors populate it via
`self.data.add_sample(level, time, cpu_util, memory, gpu_util,
gpu_band, gpu_mem, io_counters)` from their collection loop.

The protocol is `@runtime_checkable`, so

```python
isinstance(my_monitor, MonitorProtocol)
```

tells you whether your class exposes the required attributes.

---

## Plugging a custom monitor into the service

`PerfmonitorService.start_monitoring` accepts an optional `monitor=`
parameter. A user-supplied instance takes precedence over
`monitor_type` and bypasses the built-in factory:

```python
from jumper_extension.core.service import build_perfmonitor_service

service = build_perfmonitor_service()
service.start_monitoring(interval=1.0, monitor=my_custom_monitor)

# ... workload ...

service.stop_monitoring()
```

After `start_monitoring` returns, the service transparently attaches
the visualizer, reporter, session exporter and magic commands to your
monitor. Everything that works with the default backends — live
plots, `%perfmonitor_perfreport`, `export_session`, etc. — works with
a custom monitor as long as its `data` container is populated on the
same shape as the built-in monitors (CPU/memory/GPU/IO metric columns
at the expected per-level granularity).

!!! note
    When `monitor=` is used together with `check_sanity=True` (the
    Python counterpart of `--check-sanity`), the tailored sanity check
    is automatically skipped and a warning is printed. The tailored
    check assumes the metric schema of the built-in backends; see
    [Interaction with `--check-sanity`](#interaction-with-check-sanity).

---

## Worked example — SLURM multi-node monitor

The bundled `SlurmMultinodeMonitor` is itself a custom monitor: it
satisfies `MonitorProtocol` but collects samples from all nodes of a
SLURM allocation via `srun`. It's a good reference for what a
non-trivial custom backend looks like.

```python
from jumper_extension.core.service import build_perfmonitor_service
from jumper_extension.monitor.backends.slurm_multinode import (
    SlurmMultinodeMonitor,
)

service = build_perfmonitor_service()

# Configure a multi-node SLURM monitor. Both arguments are optional;
# defaults target a standard single-allocation setup.
my_monitor = SlurmMultinodeMonitor(
    log_path="runs/2026-04-21/jumper_multinode.jsonl",
    python_executable="/opt/conda/envs/hpc/bin/python",
)

service.start_monitoring(interval=1.0, monitor=my_monitor)

# Drive a distributed workload here (MPI, torch.distributed, etc.);
# every remote collector streams samples back over its srun pipe.

service.stop_monitoring()

# The standard reporter, plotter and exporter all work against the
# monitor you just plugged in:
service.export_perfdata(file="runs/2026-04-21/perf.csv")
```

Inside `SlurmMultinodeMonitor`, the protocol attributes are populated
from the ready-handshake of each per-node collector:

- `start()` discovers the node list (`get_slurm_nodes()`), launches
  a collector on each node via `srun`, waits for a JSON `"ready"`
  message per node, aggregates hardware info (`_aggregate_hardware_info`),
  and starts per-node reader threads that feed samples into the shared
  `PerformanceData`.
- `stop()` terminates the remote collectors, joins the reader threads,
  and closes the log writer.

See `jumper_extension/monitor/backends/slurm_multinode/monitor.py` for
the full implementation.

---

## Writing your own monitor

A minimal skeleton for a new monitor looks like this:

```python
import threading
import time
from typing import List, Optional

from jumper_extension.adapters.data import PerformanceData
from jumper_extension.utilities import get_available_levels


class MyCustomMonitor:
    """Monitor that collects samples from my_metric_source."""

    def __init__(self):
        # Protocol surface
        self.interval: float = 1.0
        self.running: bool = False
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.wallclock_start_time: Optional[float] = None
        self.wallclock_stop_time: Optional[float] = None

        # Hardware metadata (fill in what applies to your source)
        self.num_cpus: int = 0
        self.num_system_cpus: int = 0
        self.num_gpus: int = 0
        self.gpu_memory: float = 0.0
        self.gpu_name: str = ""
        self.cpu_handles: list = []
        self.memory_limits: dict = {}

        # Session state
        self.is_imported: bool = False
        self.session_source: Optional[str] = None

        # Data container (set in start())
        self.data: Optional[PerformanceData] = None
        self.levels: List[str] = get_available_levels()

        # Internal
        self._thread: Optional[threading.Thread] = None

    def start(self, interval: float = 1.0) -> None:
        if self.running:
            return
        self.interval = interval
        self.start_time = time.perf_counter()
        self.wallclock_start_time = time.time()

        # Probe your source, fill hardware info ...
        self.num_cpus = self.num_system_cpus = 1

        self.data = PerformanceData(
            self.num_cpus, self.num_system_cpus, self.num_gpus
        )

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.stop_time = time.perf_counter()
        self.wallclock_stop_time = time.time()

    def _loop(self) -> None:
        while self.running:
            now = time.perf_counter() - (self.start_time or 0.0)
            # Replace these with real measurements from your source.
            cpu_util = [0.0]
            memory = 0.0
            gpu_util: list = []
            gpu_band: list = []
            gpu_mem: list = []
            io = [0, 0, 0, 0]  # read_bytes, write_bytes, reads, writes
            for level in self.levels:
                self.data.add_sample(
                    level, now, cpu_util, memory,
                    gpu_util, gpu_band, gpu_mem, io,
                )
            time.sleep(self.interval)
```

Hook it up the same way:

```python
service = build_perfmonitor_service()
service.start_monitoring(interval=0.5, monitor=MyCustomMonitor())
```

Rules of thumb:

- **Populate `data` on every tick for each monitoring level** that
  your source covers. Built-in visualizations and reports iterate over
  `monitor.levels` and read `monitor.data.data[level]`.
- **Set timestamps** (`start_time`, `wallclock_start_time`, and their
  `stop_time` counterparts) so the reporter can compute durations.
- **Keep `running` truthful** — the magic commands branch on it.
- **Be robust to `stop()` being called while `start()` is still
  wiring things up**; model lifecycle with a stop event, like the
  built-in backends do.

---

## Interaction with `--check-sanity`

`%perfmonitor_start --check-sanity` runs a short validation of the
collected samples (required columns present, no NaN, non-zero for
active-by-definition metrics, etc.).

!!! warning "IMPORTANT"
    The sanity check was tailored for the `thread`,
    `subprocess_python` and `native_c` backends. It assumes the
    per-level metric schema produced by those monitors (CPU/memory/IO
    columns, optional GPU columns). Running it against any other
    monitor — including `slurm_multinode` and any custom monitor you
    write — is **expected to fail**, because the column set produced
    by those backends is different.

    When a custom monitor is plugged in via
    `service.start_monitoring(monitor=…)`, JUmPER detects this and
    **skips** the sanity check automatically, printing a warning so
    the user is aware. A skipped check is not a failure of the monitor
    itself; it only means the tailored check does not apply.
