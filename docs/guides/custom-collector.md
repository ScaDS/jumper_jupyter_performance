# Custom Collectors

JUmPER's metric pipeline is fully pluggable. If you want to extend an existing
monitor with a new group of logically related metrics — network I/O, a hardware
sensor, a custom counter — without implementing a full
[`MonitorProtocol`](custom-monitor.md) backend, you can register a **collector**:
a `CollectorBackend` + `StorageHandler` pair that JUmPER loads automatically from
`collectors.yaml`.

A collector is a pair of:

- **`CollectorBackend`** — gathers raw data each tick (see
  `jumper_extension/monitor/metrics/common.py` for the full interface contract).
- **`StorageHandler`** — converts the raw value into a flat `{column: value}` dict
  that becomes a DataFrame row.

Both are registered in `jumper_extension/config/collectors.yaml` and instantiated
automatically — no changes to the monitor or pipeline code are needed.

!!! note
    Collectors added via `collectors.yaml` are loaded by the **`thread`** and
    **`subprocess_python`** monitors only. The default `native_c` monitor uses a
    compiled C binary with its own hardcoded collection logic and does not read
    `collectors.yaml`. To use a custom collector, start the monitor explicitly:
    ```python
    %perfmonitor_start --monitor thread
    # or
    %perfmonitor_start --monitor subprocess_python
    ```

## Step 1 — Create the collector module

By convention each metric lives in its own subdirectory under
`jumper_extension/monitor/metrics/`. The standard layout is:

```
jumper_extension/monitor/metrics/
└── your_metric/
    ├── common.py    # YourCollectorBackend — ABC that narrows the collect() return type
    ├── psutil.py    # PsutilYourCollector  — concrete implementation
    └── __init__.py  # re-exports YourCollectorBackend
```

`common.py` separates the interface from the implementation so multiple backends
(psutil, native, remote) can coexist under the same metric. `__init__.py`
re-exports the base class for clean imports.

!!! note
    The `_target_:` key in `collectors.yaml` resolves any importable class, so
    you can place your collector anywhere in the package — `metrics/` is
    convention, not a requirement.

### Example — `NetworkCollector`

**`metrics/network/common.py`**
```python
from abc import abstractmethod
from jumper_extension.monitor.metrics.common import CollectorBackend
from jumper_extension.monitor.metrics.context import CollectionContext

class NetworkCollectorBackend(CollectorBackend):
    """Base for network metric backends."""
    name = "network-base"

    @abstractmethod
    def collect(self, level: str, context: CollectionContext) -> list[int]: ...
```

**`metrics/network/psutil.py`**
```python
import psutil
from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.network.common import NetworkCollectorBackend

class PsutilNetworkCollector(NetworkCollectorBackend):
    """System-wide network I/O via psutil.

    psutil does not expose per-process network counters on Linux without root,
    so only the 'system' level returns real data; other levels report zeros.
    """
    name = "network-psutil"

    def collect(self, level: str, context: CollectionContext) -> list[int]:
        if level == "system":
            net = psutil.net_io_counters()
            if net:
                return [net.bytes_sent, net.bytes_recv,
                        net.packets_sent, net.packets_recv]
        return [0, 0, 0, 0]
```

**`metrics/network/__init__.py`**
```python
from jumper_extension.monitor.metrics.network.common import NetworkCollectorBackend
```

## Step 2 — Pick or create a StorageHandler

A handler converts the value returned by `collect()` into DataFrame columns.
Built-in handlers live in `jumper_extension/monitor/metrics/handlers.py`:

| Handler | Raw type | Output columns | Use when |
|---------|----------|----------------|----------|
| `ScalarHandler(column="x")` | `float` | `{"x": v}` | Single scalar value (e.g. memory GB) |
| `PerDeviceAggregateHandler(prefix="p_")` | `list[float]` | `p_0, p_1, …, p_avg, p_min, p_max` | Per-device readings to aggregate (e.g. per-CPU utilization) |
| `PerDeviceMultiAggregateHandler(prefix="p_", metrics=[…])` | `tuple[list[float], …]` | Fan-out of PerDeviceAggregate per metric | Multiple metrics per device (e.g. GPU util + bandwidth + memory) |
| `CumulativeRateHandler(columns=[…])` | `list[int]` | Per-column delta/second rates | Monotonically increasing counters (e.g. bytes transferred) |
| `NoOpHandler()` | `None` | `{}` | Context-only backends that write no metric columns |

!!! note
    Handlers are also resolved via `_target_:`, so a custom handler can live
    anywhere — point `_target_:` at it and it works.

`NetworkCollector` returns `list[int]` cumulative byte/packet counters — so we can
pair it with the existing `CumulativeRateHandler` to get bytes/s and packets/s
automatically.

## Step 3 — Register in `collectors.yaml`

```yaml
collectors:
  # ... existing collectors ...

  network:
    _target_: jumper_extension.monitor.metrics.network.psutil.PsutilNetworkCollector
    inject: []
    handler:
      _target_: jumper_extension.monitor.metrics.handlers.CumulativeRateHandler
      columns: [net_bytes_sent, net_bytes_recv, net_packets_sent, net_packets_recv]
```

### The `inject:` key

`inject:` lists all `PerformanceMonitor` attributes you may need in your
collector. All collectors receive the `inject` list as constructor arguments.
Use it when your collector needs monitor-level context:

| Value | Type | What you get |
|-------|------|-------------|
| `[]` | — | No injected dependencies (simplest case) |
| `[node_info]` | `NodeInfo` | Hardware topology: CPU count and handles, GPU count and memory, per-level memory limits |
| `[uid, slurm_job]` | `int, str\|int` | Current user ID and SLURM job ID — filter metrics by user or job scope |
| `[pid, process, uid, slurm_job]` | mixed | Full process context — used by the built-in process collector to enumerate live PIDs |

`NetworkCollector` uses `inject: []` because `psutil.net_io_counters()` needs no
hardware context.

!!! tip
    You can add any attribute of `PerformanceMonitor` to `inject:` — for example,
    `node_info` to receive a fully populated `NodeInfo` object with the detected
    hardware layout: number of CPUs and GPUs, GPU memory size, per-level memory
    limits, and CPU handles. This lets your collector adapt its behaviour to the
    hardware — for instance, scale reporting thresholds by GPU memory, tag samples
    with the node name, or skip collection entirely when no GPUs are present.

!!! note
    Custom metric subsets are not shown by default in `%perfmonitor_plot`. The
    default widget shows only the subsets listed under `default_subsets:` in
    `plots.yaml` (initially `cpu`, `mem`, `io`, plus `gpu`/`gpu_all` when a GPU
    is detected). To include your subset in the default view, add it to that list:
    ```yaml
    # plots.yaml
    default_subsets: [cpu, mem, io, network]
    ```
    Alternatively, request it on demand without changing the config:
    ```python
    %perfmonitor_plot --metrics net_bytes_recv --level system
    ```

---

With your collector registered and emitting data, the new columns are available to
`%perfmonitor_plot --metrics`. To surface them in the plot widget and configure
chart types, continue to
[Visualizing Custom Collector Metrics](visualizing-custom-collector-metrics.md).
