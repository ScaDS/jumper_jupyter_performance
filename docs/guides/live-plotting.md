# Live Performance Plotting

The `%perfmonitor_plot --live` command creates continuously updating
performance plots.  Other notebook cells keep running while the graphs
refresh in the background.

## Prerequisites

Live plotting requires the **matplotlib** backend with **ipympl**:

```python
%matplotlib ipympl
```

## Basic Usage

```python
%perfmonitor_start
%perfmonitor_plot --live --backend matplotlib       # 2 s updates (default)
%perfmonitor_plot --live 1.0 --backend matplotlib   # 1 s updates
%perfmonitor_plot --live 5.0 --backend matplotlib   # 5 s updates
```

## What You See

- **Two plot panels** are created initially, each with a *Metric* and
  *Level* dropdown.
- **Cell execution regions** are drawn as coloured background rectangles
  with cell-index labels.  Idle periods between cells are always visible.
- An **"Add Plot Panel"** button lets you add more panels.
- A **"Stop Live Update"** button and green status indicator sit in the
  header bar.

## Examples

### Default metrics

```python
%perfmonitor_start
%perfmonitor_plot --live
```

### Specific metrics with fast updates

```python
%perfmonitor_start
%perfmonitor_plot --live 0.5 --metrics cpu,mem
```

### Monitor a long-running cell

```python
%perfmonitor_start
%perfmonitor_plot --live 1.0 --backend matplotlib

# plots keep updating while this runs
import time
for i in range(30):
    time.sleep(1)
```

## Stopping

- Click **Stop Live Update** in the header, or
- run `%perfmonitor_stop` to end the monitoring session.

## Limitations

- **Matplotlib only** — the Plotly backend does not support live mode.
- Only works with *active* monitoring sessions (not imported ones).
- Recommended minimum update interval is **0.5 s**; shorter values may
  affect notebook responsiveness.
- Best results with the `ipympl` (widget) matplotlib backend.
