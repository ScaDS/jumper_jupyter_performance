# Visualizing Custom Collector Metrics

Every column produced by a collector ends up as a column in the performance
DataFrame. To make it available to `%perfmonitor_plot --metrics`, add an entry to
`jumper_extension/config/plots.yaml` under `subsets:`:

```yaml
subsets:
  your_subset:           # group name — also usable as a shorthand key in --metrics
    your_metric_key:     # what the user types in --metrics
      type: single_series
      column: your_column    # column name declared in collectors.yaml handler columns
      title: "Chart title"
      ylim: null             # or [min, max]
      label: "Legend label"
```

Four built-in plot types are available:

| Type | Use when |
|------|----------|
| `single_series` | One line from one column |
| `summary_series` | Three lines (min / avg / max) from three named columns |
| `multi_series` | One line per device, matched by column prefix |
| `composite_series` | Multiple columns from any collector on a single panel, with individual labels and colors |

Once registered, the metric key works everywhere `--metrics` is accepted —
interactive widgets, direct plots, live mode, and exports.

## Controlling which subsets appear by default

When `%perfmonitor_plot` is called without `--metrics`, it shows the subsets listed
in `default_subsets:` at the top of `plots.yaml`:

```yaml
# plots.yaml
default_subsets: [cpu, mem, io]   # shown when %perfmonitor_plot is called with no --metrics

subsets:
  cpu: ...
  mem: ...
  io: ...
  network: ...   # not shown by default — request with --metrics or add to default_subsets
```

To make your custom subset part of the default view, append its name:

```yaml
default_subsets: [cpu, mem, io, network]
```

GPU subsets (`gpu`, `gpu_all`) are appended automatically at runtime when a GPU is
detected, regardless of what is listed here.

## Example — combining NetworkCollector with disk I/O on one panel

A common HPC and ML scenario: you want to know whether your workload is
bottlenecked by the local disk or by the network (e.g. data loaded from a
Lustre/NFS mount). Both `io_read` and `net_bytes_recv` are collected as cumulative
counters and stored in bytes/s — the same unit — so they can be plotted together on
one panel using `composite_series`.

Add to `plots.yaml`:

```yaml
subsets:
  network:
    net_vs_disk_read:
      type: composite_series
      series:
        - column: io_read
          label: "Disk Read (bytes/s)"
          color: "steelblue"
          width: 2.0
        - column: net_bytes_recv
          label: "Net Recv (bytes/s)"
          color: "darkorange"
          width: 2.0
      title: "Disk Read vs Network Receive (bytes/s)"
      ylim: null
      label: "Disk vs Network"
```

!!! note
    `composite_series` renders columns without unit conversion. `io_read` will
    appear in bytes/s here, not MB/s (which is what `single_series` applies
    automatically). Both series stay on the same scale, making the comparison valid.

Then plot it — use `--level system` because network counters are only meaningful at
the system level:

```python
%perfmonitor_plot --metrics net_vs_disk_read --level system
```

Or in live mode:

```python
%perfmonitor_plot --live --metrics net_vs_disk_read --level system
```

A high `net_bytes_recv` with low `io_read` means data is arriving over the network;
the inverse points to local disk as the bottleneck.
