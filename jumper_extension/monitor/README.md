# Performance Monitoring

This package provides the performance-monitoring infrastructure for the
JUmPER IPython extension.  It defines a common **protocol** that all
monitors implement, several **concrete backends**, and the low-level
**metric collectors** (CPU, memory, GPU, I/O).

## Directory layout

```
monitor/
├── common.py                       # MonitorProtocol + utility monitors
├── metrics/                        # Pluggable metric collectors (psutil, NVML, …)
│   ├── cpu/                        #   Used by the Python-based monitors only
│   ├── gpu/                        #   (thread, subprocess_python, slurm_multinode).
│   ├── io/                         #   The C collector (native_c) reads /proc and
│   ├── memory/                     #   loads NVML directly — it has no dependency
│   └── process/                    #   on anything in metrics/.
└── backends/
    ├── thread/                     # In-process threaded monitor
    │   └── monitor.py              # PerformanceMonitor
    ├── subprocess_python/          # Out-of-process Python collector
    │   ├── _collector.py           # Python collector (run in child process)
    │   └── monitor.py              # SubprocessPerformanceMonitor
    ├── native_c/                   # Native C collector monitor
    │   ├── collector.c / Makefile  # C collector source & build
    │   └── monitor.py              # CSubprocessPerformanceMonitor
    └── slurm_multinode/            # Multi-node SLURM monitor
        ├── _collector.py           # Per-node collector (run via srun)
        ├── _log_writer.py          # JSON-Lines log writer
        ├── _node_discovery.py      # SLURM node list expansion
        └── monitor.py              # SlurmMultinodeMonitor
```

## MonitorProtocol

Every monitor exposes the same interface (`MonitorProtocol` in
`common.py`):

| Attribute / Method       | Description                              |
|--------------------------|------------------------------------------|
| `start(interval)`       | Begin collecting metrics                  |
| `stop()`                | Stop collecting and finalise timestamps   |
| `running`               | Whether the monitor is currently active   |
| `data`                  | `PerformanceData` container with results  |
| `interval`              | Sampling interval in seconds              |
| `num_cpus`              | Number of CPUs visible to the process     |
| `num_gpus`              | Number of GPUs detected                   |
| `memory_limits`         | Per-level memory limits (GiB)             |

The visualizer, reporter, and session exporter all program against this
protocol, so any backend can be swapped in transparently.

## Available monitors

### 1. Thread monitor (`"thread"`) (*deprecated, measurement resolution depends on GIL, potentially enough for non-CPU bound applications*)

```python
from jumper_extension.monitor.backends.thread import PerformanceMonitor
```

The original monitor.  Collects metrics in a daemon thread inside the
same Python process using **psutil** and **pynvml**.  Simple and
portable, but the GIL can delay sampling when the main thread is
CPU-bound.

### 2. Native C collector monitor (`"native_c"`) — **default when a C compiler is available**

```python
from jumper_extension.monitor.backends.native_c import CSubprocessPerformanceMonitor
```

Launches a **compiled C binary** (`jumper_collector`) that reads `/proc`
directly and speaks the same JSON-lines protocol.  Benefits:

- No Python startup overhead
- Minimal per-tick latency
- NVIDIA GPU metrics via dynamic loading of `libnvidia-ml.so`
  (no compile-time dependency; graceful fallback if absent)
- SLURM level auto-detected from the target process's environment

The binary is **compiled automatically** from `collector.c` the first
time the monitor is requested (or during `pip install`).  If compilation
fails (no C compiler) or the sanity check detects missing metrics, the
subprocess Python collector is used instead — no manual intervention
needed.

### 3. Subprocess monitor — Python collector (`"subprocess_python"`)

```python
from jumper_extension.monitor.backends.subprocess_python import SubprocessPerformanceMonitor
```

Spawns a **child Python process** that runs the same psutil-based
collection loop.  Results stream back to the parent over a pipe as
JSON lines.  Because collection happens in a separate process, it is
immune to GIL contention.  This is the automatic fallback when the
native C collector cannot be built or fails its sanity check.

### 4. SLURM multi-node monitor (`"slurm_multinode"`) (*experimental*)

```python
from jumper_extension.monitor.backends.slurm_multinode import SlurmMultinodeMonitor
```

Discovers all nodes allocated to the current SLURM job, launches a
collector on each via `srun`, and aggregates their JSON sample streams into
a log file.  Designed for distributed HPC workloads.

## Selecting a monitor

From the IPython magic:

```
%perfmonitor_start --monitor default          # best available (native_c → subprocess_python)
%perfmonitor_start --monitor native_c          # native C collector (explicit)
%perfmonitor_start --monitor subprocess_python # Python subprocess (explicit)
%perfmonitor_start --monitor thread           # in-process thread
%perfmonitor_start --monitor slurm_multinode  # multi-node SLURM
```

Or programmatically via the service factory:

```python
service.start_monitoring(monitor_type="native_c")
```

## Sanity check (`--check-sanity`)

`%perfmonitor_start` accepts a `--check-sanity` flag that runs a short
validation of the selected backend before starting real monitoring.
The check collects a few samples while generating background I/O, then
verifies that the expected metric columns are present, contain no NaN
values, and are not all-zero at each monitoring level.

```
%perfmonitor_start --check-sanity                    # check + start default
%perfmonitor_start --monitor native_c --check-sanity # check + start native_c
```

> **IMPORTANT.** The sanity check was tailored for the `thread`,
> `subprocess_python` and `native_c` backends.  Running it against any
> other monitor (e.g. `slurm_multinode`, or a custom monitor provided
> via the programmatic API) is **expected to fail** because those
> backends do not populate the same set of per-level metric columns.
> In that case a warning is printed and the check is skipped — a
> skipped check does **not** mean the monitor is broken; it only
> means the tailored check does not apply to it.

## Custom monitors

Any object that satisfies `MonitorProtocol` (see `common.py`) can be
plugged into the service via the `monitor=` argument of
`service.start_monitoring`. The protocol is small — lifecycle methods
plus a handful of metadata attributes:

```python
class MonitorProtocol(Protocol):
    interval: float
    data: PerformanceData
    num_cpus: int
    num_system_cpus: int
    num_gpus: int
    memory_limits: dict
    cpu_handles: list[int]
    gpu_name: str
    running: bool
    is_imported: bool
    session_source: Optional[str]

    def start(self, interval: float = 1.0) -> None: ...
    def stop(self) -> None: ...
```

A user-supplied monitor takes precedence over `monitor_type` and
bypasses the built-in factory. The bundled `SlurmMultinodeMonitor`
is itself a good example of a non-trivial custom monitor:

```python
from jumper_extension.core.service import build_perfmonitor_service
from jumper_extension.monitor.backends.slurm_multinode import (
    SlurmMultinodeMonitor,
)

service = build_perfmonitor_service()

my_monitor = SlurmMultinodeMonitor(
    log_path="runs/2026-04-21/jumper_multinode.jsonl",
    python_executable="/opt/conda/envs/hpc/bin/python",
)

service.start_monitoring(interval=1.0, monitor=my_monitor)

# ... run workload ...

service.stop_monitoring()
```

After `start_monitoring` returns, the visualizer, reporter, session
exporter and magic commands are all transparently attached to the
custom monitor. The full walk-through with a minimal end-to-end
skeleton lives in the online docs:
[Custom Monitors guide](https://scads.github.io/jumper_jupyter_performance/latest/guides/custom-monitor/).

> **Note.** `--check-sanity` is automatically skipped when a custom
> monitor is plugged in via `monitor=`, because the tailored check
> assumes the metric schema of the built-in backends (see above).
