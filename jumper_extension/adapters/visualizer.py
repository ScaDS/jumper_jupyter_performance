import json
import logging
import pickle
import re
import uuid
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, runtime_checkable, Protocol, Optional, Tuple

import matplotlib.pyplot as plt
from IPython.display import display, HTML
from ipywidgets import widgets, Layout
from jinja2 import Environment, FileSystemLoader, select_autoescape
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.monitor.common import UnavailablePerformanceMonitor, \
    MonitorProtocol
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES, ExtensionInfoCode, EXTENSION_INFO_MESSAGES,
)
from jumper_extension.utilities import filter_perfdata, get_available_levels
from jumper_extension.logo import logo_image, jumper_colors

logger = logging.getLogger("extension")


def is_ipympl_backend():
    try:
        backend = plt.get_backend().lower()
    except Exception:
        return False
    return ("ipympl" in backend) or ("widget" in backend)


@runtime_checkable
class VisualizerProtocol(Protocol):
    """Structural protocol for visualizers used by the service."""
    def attach(self, monitor: MonitorProtocol) -> None: ...
    def plot(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
        level=None,
        save_jpeg=None,
        pickle_file=None,
    ) -> None: ...


class PerformanceVisualizer:
    """Visualizes performance metrics collected by PerformanceMonitor.

    Supports multiple levels: 'user', 'process' (default), 'system', and
    'slurm' (if available)
    """

    def __init__(self, cell_history: CellHistory):
        self.monitor = UnavailablePerformanceMonitor(
            reason="Monitor has not been started yet."
        )
        self.cell_history = cell_history
        self.figsize = (5, 3)
        self.min_duration = None
        self._io_window = None
        self.subsets = {}

    def attach(
        self,
        monitor: MonitorProtocol,
    ):
        """Attach started PerformanceMonitor."""
        self.monitor = monitor
        self.min_duration = self.monitor.interval
        # Smooth IO with ~1s rolling window based on sampling interval
        try:
            self._io_window = max(
                1, int(round(1.0 / (self.monitor.interval or 1.0)))
            )
        except Exception:
            self._io_window = 1
        self._build_subsets()

    def _build_subsets(self):
        """Build a dictionary of metric subsets based on the provided
        configuration"""
        # Compressed metrics configuration (dict-based entries for clarity)
        self.subsets = {
            "cpu_all": {
                "cpu": {
                    "type": "multi_series",
                    "prefix": "cpu_util_",
                    "title": "CPU Utilization (%) - Across Cores",
                    "ylim": (0, 100),
                    "label": "CPU Utilization (All Cores)",
                }
            },
            "gpu_all": {
                "gpu_util": {
                    "type": "multi_series",
                    "prefix": "gpu_util_",
                    "title": "GPU Utilization (%) - Across GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Utilization (All GPUs)",
                },
                "gpu_band": {
                    "type": "multi_series",
                    "prefix": "gpu_band_",
                    "title": "GPU Bandwidth Usage (%) - Across GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Bandwidth (All GPUs)",
                },
                "gpu_mem": {
                    "type": "multi_series",
                    "prefix": "gpu_mem_",
                    "title": "GPU Memory Usage (GB) - Across GPUs",
                    "ylim": (0, self.monitor.gpu_memory),
                    "label": "GPU Memory (All GPUs)",
                },
            },
            "cpu": {
                "cpu_summary": {
                    "type": "summary_series",
                    "columns": [
                        "cpu_util_min",
                        "cpu_util_avg",
                        "cpu_util_max",
                    ],
                    "title": (
                        "CPU Utilization (%) - "
                        f"{self.monitor.num_cpus} CPUs"
                    ),
                    "ylim": (0, 100),
                    "label": "CPU Utilization Summary",
                }
            },
            "gpu": {
                "gpu_util_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_util_min",
                        "gpu_util_avg",
                        "gpu_util_max",
                    ],
                    "title": (
                        "GPU Utilization (%) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": (0, 100),
                    "label": "GPU Utilization Summary",
                },
                "gpu_band_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_band_min",
                        "gpu_band_avg",
                        "gpu_band_max",
                    ],
                    "title": (
                        "GPU Bandwidth Usage (%) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": (0, 100),
                    "label": "GPU Bandwidth Summary",
                },
                "gpu_mem_summary": {
                    "type": "summary_series",
                    "columns": ["gpu_mem_min", "gpu_mem_avg", "gpu_mem_max"],
                    "title": (
                        "GPU Memory Usage (GB) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": (0, self.monitor.gpu_memory),
                    "label": "GPU Memory Summary",
                },
            },
            "mem": {
                "memory": {
                    "type": "single_series",
                    "column": "memory",
                    "title": "Memory Usage (GB)",
                    "ylim": None,  # Will be set dynamically based on level
                    "label": "Memory Usage",
                }
            },
            "io": {
                "io_read": {
                    "type": "single_series",
                    "column": "io_read",
                    "title": "I/O Read (MB/s)",
                    "ylim": None,
                    "label": "IO Read MB/s",
                },
                "io_write": {
                    "type": "single_series",
                    "column": "io_write",
                    "title": "I/O Write (MB/s)",
                    "ylim": None,
                    "label": "IO Write MB/s",
                },
                "io_read_count": {
                    "type": "single_series",
                    "column": "io_read_count",
                    "title": "I/O Read Operations (ops/s)",
                    "ylim": None,
                    "label": "IO Read Ops",
                },
                "io_write_count": {
                    "type": "single_series",
                    "column": "io_write_count",
                    "title": "I/O Write Operations (ops/s)",
                    "ylim": None,
                    "label": "IO Write Ops",
                },
            },
        }

    def _resolve_metric_subsets(
        self,
        metrics: Optional[List[str]]
    ) -> Tuple[str, ...]:
        """Map user-specified metrics or subsets to visualizer subset keys."""
        if not metrics:
            return ("cpu", "mem", "io")

        resolved: List[str] = []
        metric_list = (
            [metrics]
            if isinstance(metrics, str)
            else list(metrics)
        )
        for metric in metric_list:
            if not metric:
                continue
            metric_key = str(metric).strip()
            if metric_key in self.subsets:
                resolved.append(metric_key)
                continue
            found_subset = next(
                (
                    subset
                    for subset, cfg in self.subsets.items()
                    if metric_key in cfg
                ),
                None,
            )
            if found_subset:
                resolved.append(found_subset)
            else:
                logger.warning(
                    EXTENSION_ERROR_MESSAGES[
                        ExtensionErrorCode.INVALID_METRIC_SUBSET
                    ].format(
                        subset=metric_key,
                        supported_subsets=", ".join(self.subsets.keys()),
                    )
                )

        # Remove duplicates while preserving order; fall back to defaults
        deduped = tuple(dict.fromkeys(resolved))
        return deduped or ("cpu", "mem", "io")

    def _compress_time_axis(self, perfdata, cell_range):
        """Compress time axis by removing idle periods between cells"""
        if perfdata.empty:
            return perfdata, []

        start_idx, end_idx = cell_range
        cell_data = self.cell_history.view(start_idx, end_idx + 1)
        compressed_perfdata, cell_boundaries, current_time = (
            perfdata.copy(),
            [],
            0,
        )

        for idx, cell in cell_data.iterrows():
            cell_mask = (perfdata["time"] >= cell["start_time"]) & (
                perfdata["time"] <= cell["end_time"]
            )
            cell_perfdata = perfdata[cell_mask]

            if not cell_perfdata.empty:
                original_start, cell_duration = (
                    cell["start_time"],
                    cell["end_time"] - cell["start_time"],
                )
                compressed_perfdata.loc[cell_mask, "time"] = current_time + (
                    cell_perfdata["time"].values - original_start
                )
                cell_boundaries.append(
                    {
                        "cell_index": cell["cell_index"],
                        "start_time": current_time,
                        "end_time": current_time + cell_duration,
                        "duration": cell_duration,
                    }
                )
                current_time += cell_duration

        return compressed_perfdata, cell_boundaries

    def _collect_metric_options(self, metric_subsets):
        metrics = []
        labeled_options = []
        for subset in metric_subsets:
            if subset in self.subsets:
                for metric_key, cfg in self.subsets[subset].items():
                    metrics.append(metric_key)
                    label = (
                        cfg.get("label")
                        if isinstance(cfg, dict)
                        else metric_key
                    )
                    labeled_options.append((label or metric_key, metric_key))
            else:
                logger.warning(
                    EXTENSION_ERROR_MESSAGES[
                        ExtensionErrorCode.INVALID_METRIC_SUBSET
                    ].format(
                        subset=subset,
                        supported_subsets=", ".join(self.subsets.keys()),
                    )
                )
        return metrics, labeled_options

    def _prepare_processed_data_for_level(self, cell_range, show_idle, level):
        start_idx, end_idx = cell_range
        filtered_cells = self.cell_history.view(start_idx, end_idx + 1)
        perfdata = filter_perfdata(
            filtered_cells,
            self.monitor.data.view(level=level),
            not show_idle,
        )

        if perfdata.empty:
            return None

        if not show_idle:
            processed_data, self._compressed_cell_boundaries = (
                self._compress_time_axis(perfdata, cell_range)
            )
        else:
            processed_data = perfdata.copy()
            processed_data["time"] -= self.monitor.start_time
        return processed_data

    def _prepare_processed_data_for_interactive(
        self,
        current_cell_range,
        current_show_idle,
    ):
        start_idx, end_idx = current_cell_range
        cells_all = self.cell_history.view()
        try:
            mask = (cells_all["cell_index"] >= start_idx) & (
                cells_all["cell_index"] <= end_idx
            )
            filtered_cells = cells_all[mask]
        except Exception:
            filtered_cells = cells_all

        perfdata_by_level = {}
        for available_level in get_available_levels():
            perfdata_by_level[available_level] = filter_perfdata(
                filtered_cells,
                self.monitor.data.view(level=available_level),
                not current_show_idle,
            )

        if all(df.empty for df in perfdata_by_level.values()):
            return None

        processed_perfdata = {}
        for level_key, perfdata in perfdata_by_level.items():
            if not perfdata.empty:
                if not current_show_idle:
                    processed_data, self._compressed_cell_boundaries = (
                        self._compress_time_axis(perfdata, current_cell_range)
                    )
                    processed_perfdata[level_key] = processed_data
                else:
                    processed_data = perfdata.copy()
                    processed_data["time"] -= self.monitor.start_time
                    processed_perfdata[level_key] = processed_data
            else:
                processed_perfdata[level_key] = perfdata

        return processed_perfdata

    def _create_interactive_wrapper(
        self,
        metrics,
        labeled_options,
        processed_perfdata,
        current_cell_range,
        current_show_idle,
    ):
        raise NotImplementedError

    def _render_direct_plot(
        self,
        processed_data,
        metrics,
        cell_range,
        show_idle,
        level,
        save_jpeg=None,
        pickle_file=None,
        metric_subsets=None,
    ):
        raise NotImplementedError

    def _plot_direct(
        self,
        metric_subsets,
        cell_range,
        show_idle,
        level,
        save_jpeg=None,
        pickle_file=None,
    ):
        processed_data = self._prepare_processed_data_for_level(
            cell_range, show_idle, level
        )
        if processed_data is None:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.NO_PERFORMANCE_DATA
                ]
            )
            return

        metrics, _ = self._collect_metric_options(metric_subsets)
        if not metrics:
            logger.warning("No valid metrics found to plot")
            return

        self._render_direct_plot(
            processed_data=processed_data,
            metrics=metrics,
            cell_range=cell_range,
            show_idle=show_idle,
            level=level,
            save_jpeg=save_jpeg,
            pickle_file=pickle_file,
            metric_subsets=metric_subsets,
        )

    def plot(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
        level=None,
        save_jpeg=None,
        pickle_file=None,
    ):
        metrics_missing = not metric_subsets
        if metrics_missing:
            metric_subsets = ("cpu", "mem", "io")
            if self.monitor.num_gpus:
                metric_subsets += (
                    "gpu",
                    "gpu_all",
                )

        """Plot performance metrics with interactive widgets for
        configuration."""
        valid_cells = self.cell_history.view()
        if len(valid_cells) == 0:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_CELL_HISTORY]
            )
            return

        # Default to all cells if no range specified
        try:
            min_cell_idx = int(valid_cells["cell_index"].min())
            max_cell_idx = int(valid_cells["cell_index"].max())
        except Exception:
            min_cell_idx, max_cell_idx = 0, len(valid_cells) - 1
        if cell_range is None:
            cell_start_index = 0
            for cell_idx in range(len(valid_cells) - 1, -1, -1):
                if valid_cells.iloc[cell_idx]["duration"] > self.min_duration:
                    cell_start_index = cell_idx
                    break
            start = int(valid_cells.iloc[cell_start_index]["cell_index"])
            end = int(valid_cells["cell_index"].max())
            if start > end:
                start, end = end, start
            cell_range = (start, end)

        # If level is specified, plot directly without widgets
        if level is not None:
            metric_subsets = self._resolve_metric_subsets(metric_subsets)
            return self._plot_direct(metric_subsets, cell_range, show_idle,
                                     level, save_jpeg, pickle_file)

        # Create interactive widgets
        style = {"description_width": "initial"}
        show_idle_checkbox = widgets.Checkbox(
            value=show_idle, description="Show idle periods"
        )
        # Sanitize slider value within bounds and ordered
        try:
            s0, s1 = cell_range
            if s0 > s1:
                s0, s1 = s1, s0
            s0 = max(min_cell_idx, min(s0, max_cell_idx))
            s1 = max(min_cell_idx, min(s1, max_cell_idx))
            slider_value = (s0, s1)
        except Exception:
            slider_value = (min_cell_idx, max_cell_idx)
        cell_range_slider = widgets.IntRangeSlider(
            value=slider_value,
            min=min_cell_idx,
            max=max_cell_idx,
            step=1,
            description="Cell range:",
            style=style,
        )

        logo_widget = widgets.HTML(
            value=f"<img src="
            f'"{logo_image}"'
            f'alt="JUmPER Logo" style="height: auto; width: 100px;">'
        )

        box_layout = Layout(
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            justify_content="space-between",
            width="100%",
        )

        config_widgets = widgets.HBox(
            [
                widgets.HTML("<b>Plot Configuration:</b>"),
                show_idle_checkbox,
                cell_range_slider,
                logo_widget,
            ],
            layout=box_layout,
        )
        plot_output = widgets.Output()
        plot_wrapper = None

        def update_plots():
            nonlocal plot_wrapper
            current_cell_range, current_show_idle = (
                cell_range_slider.value,
                show_idle_checkbox.value,
            )
            processed_perfdata = self._prepare_processed_data_for_interactive(
                current_cell_range, current_show_idle
            )
            if processed_perfdata is None:
                with plot_output:
                    plot_output.clear_output()
                    logger.warning(
                        EXTENSION_ERROR_MESSAGES[
                            ExtensionErrorCode.NO_PERFORMANCE_DATA
                        ]
                    )
                    plot_wrapper = None
                return

            metrics, labeled_options = self._collect_metric_options(
                metric_subsets
            )
            if not metrics:
                with plot_output:
                    plot_output.clear_output()
                    logger.warning("No valid metrics found to plot")
                    plot_wrapper = None
                return

            with plot_output:
                if plot_wrapper is None:
                    plot_output.clear_output()
                    plot_wrapper = self._create_interactive_wrapper(
                        metrics,
                        labeled_options,
                        processed_perfdata,
                        current_cell_range,
                        current_show_idle,
                    )
                    plot_wrapper.display_ui()
                else:
                    plot_wrapper.update_data(
                        processed_perfdata,
                        current_cell_range,
                        current_show_idle,
                    )

        for widget in [show_idle_checkbox, cell_range_slider]:
            widget.observe(lambda change: update_plots(), names="value")

        display(widgets.VBox([config_widgets, plot_output]))
        update_plots()

    # ---- Live plotting helpers ------------------------------------------ #

    def _build_live_figure(self, metric, level, window_seconds):
        """Build a Plotly Figure for a single metric panel in live mode.

        Returns a ``go.Figure`` with traces, cell boundaries, and layout
        configured for the current sliding window.  Returns *None* when
        there is no data to show yet.
        """
        df = self.monitor.data.view(level=level)
        now = time.perf_counter() - self.monitor.start_time
        t_start = max(0, now - window_seconds)
        t_end = now

        config = next(
            (subset[metric] for subset in self.subsets.values()
             if metric in subset), None,
        )
        if not config or not isinstance(config, dict):
            return None

        fig = go.Figure()
        plot_type = config.get("type")
        title = config.get("title", "")
        ylim = config.get("ylim")
        y_values: list = []

        has_data = df is not None and not df.empty
        if has_data:
            df = df.copy()
            df["time"] = df["time"] - self.monitor.start_time
            df = df[df["time"] >= t_start]
            has_data = not df.empty

        if has_data:
            if plot_type == "single_series":
                column = config.get("column")
                if column and column in df.columns:
                    series = df[column]
                    if metric in ("io_read", "io_write",
                                  "io_read_count", "io_write_count"):
                        diffs = df[column].astype(float).diff().clip(lower=0)
                        if metric in ("io_read", "io_write"):
                            diffs = diffs / (1024**2)
                        series = diffs.fillna(0.0)
                        if self._io_window and self._io_window > 1:
                            series = series.rolling(
                                window=self._io_window,
                                min_periods=1,
                            ).mean()
                    fig.add_trace(go.Scatter(
                        x=df["time"], y=series,
                        name=config.get("label", column),
                        mode="lines",
                        line=dict(color="blue", width=2),
                    ))
                    y_values.extend(series.tolist())
                    if metric == "memory" and ylim is None:
                        ylim = (0, self.monitor.memory_limits[level])

            elif plot_type == "summary_series":
                columns = config.get("columns", [])
                if level == "system":
                    title = re.sub(
                        r"\d+", str(self.monitor.num_system_cpus), title
                    )
                dashes = ["dot", "solid", "dash"]
                opacities = [0.35, 1.0, 0.35]
                labels = ["Min", "Average", "Max"]
                for i, col in enumerate(columns):
                    if col not in df.columns:
                        continue
                    fig.add_trace(go.Scatter(
                        x=df["time"], y=df[col],
                        name=labels[i % len(labels)],
                        mode="lines",
                        line=dict(color="blue",
                                  dash=dashes[i % len(dashes)],
                                  width=2),
                        opacity=opacities[i % len(opacities)],
                    ))
                    y_values.extend(df[col].tolist())

            elif plot_type == "multi_series":
                prefix = config.get("prefix", "")
                series_cols = [
                    c for c in df.columns
                    if prefix and c.startswith(prefix)
                    and not c.endswith("avg")
                ]
                avg_column = f"{prefix}avg" if prefix else None
                for col in series_cols:
                    fig.add_trace(go.Scatter(
                        x=df["time"], y=df[col],
                        name=col, mode="lines",
                        opacity=0.5, line=dict(width=1),
                    ))
                    y_values.extend(df[col].tolist())
                if avg_column and avg_column in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df["time"], y=df[avg_column],
                        name="Mean", mode="lines",
                        line=dict(color="blue", width=2),
                    ))
                    y_values.extend(df[avg_column].tolist())

        # Compute y-range
        if ylim is None:
            clean = [float(v) for v in y_values
                     if v == v and abs(float(v)) != float("inf")]
            if clean:
                ymin, ymax = min(clean), max(clean)
                pad = (ymax - ymin) * 0.05 or 1.0
                ylim = (ymin - pad, ymax + pad)
            else:
                ylim = (0, 1)

        # Cell boundaries
        shapes, annotations = [], []
        y_min, y_max = ylim
        height = y_max - y_min
        cells = self.cell_history.view()
        if cells is not None and not cells.empty:
            for _, cell in cells.iterrows():
                try:
                    c_start = (float(cell["start_time"])
                               - self.monitor.start_time)
                    dur = float(cell["duration"])
                    cidx = int(cell["cell_index"])
                except Exception:
                    continue
                if c_start + dur < t_start:
                    continue
                color = jumper_colors[cidx % len(jumper_colors)]
                shapes.append(dict(
                    type="rect", x0=c_start, x1=c_start + dur,
                    y0=y_min, y1=y_max,
                    fillcolor=color, opacity=0.4,
                    line=dict(color="black", dash="dash", width=1),
                    layer="below",
                ))
                annotations.append(dict(
                    x=c_start + dur / 2, y=y_max - height * 0.1,
                    text=f"#{cidx}", showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                ))

        # Show the currently executing cell as a live boundary
        current = self.cell_history.current_cell
        if current is not None:
            try:
                c_start = (float(current["start_time"])
                           - self.monitor.start_time)
                cidx = int(current["cell_index"])
                c_end = now  # extends to current time
                if c_end > t_start:
                    color = jumper_colors[cidx % len(jumper_colors)]
                    shapes.append(dict(
                        type="rect", x0=c_start, x1=c_end,
                        y0=y_min, y1=y_max,
                        fillcolor=color, opacity=0.25,
                        line=dict(color="black", dash="dot", width=1),
                        layer="below",
                    ))
                    annotations.append(dict(
                        x=c_start + (c_end - c_start) / 2,
                        y=y_max - height * 0.1,
                        text=f"#{cidx} ▶",
                        showarrow=False,
                        font=dict(size=10, color="green"),
                        bgcolor="rgba(255,255,255,0.8)",
                    ))
            except Exception:
                pass

        fig.update_layout(
            template="plotly_white",
            title=title or "Waiting for data…",
            xaxis=dict(title="Time (seconds)", range=[t_start, t_end]),
            yaxis=dict(range=list(ylim)),
            shapes=shapes,
            annotations=annotations,
            margin=dict(l=50, r=16, t=45, b=40),
            height=max(220, int(self.figsize[1] * 105)),
            legend=dict(
                orientation="h", yanchor="top", y=0.99,
                xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
            ),
        )
        return fig

    def plot_live(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
        level=None,
        update_interval=2.0,
        window_seconds=120.0,
    ):
        """Plot performance metrics with a sliding-window live view.

        Displays fixed panels that auto-update via a background thread
        using ``IPython.display.update_display``.  No ipywidgets, no
        JavaScript communication, no extra dependencies.

        Args:
            metric_subsets: Tuple of metric subset names to plot.
            cell_range: Ignored for live plotting.
            show_idle: Ignored for live plotting (always True).
            level: Monitoring level (``"process"``, ``"user"``,
                ``"system"``, or ``"slurm"``). Defaults to ``"process"``.
            update_interval: Seconds between refreshes (default 2.0).
            window_seconds: Width of the visible time window (default 120).
        """
        import plotly.io as pio

        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return

        # Remember which metric keys the user explicitly asked for
        user_requested_keys = None
        if metric_subsets is not None and len(metric_subsets) > 0:
            user_requested_keys = set(
                str(m).strip() for m in metric_subsets
            )

        # ---- Determine default metrics ------------------------------------ #
        if user_requested_keys is not None:
            # User specified --metrics: resolve and filter
            if not metric_subsets:
                metric_subsets = ("cpu", "mem", "io")
                if self.monitor.num_gpus:
                    metric_subsets += ("gpu", "gpu_all")
            metric_subsets = self._resolve_metric_subsets(metric_subsets)
            metrics_list, labeled_options = self._collect_metric_options(
                metric_subsets
            )
            metrics_list = [
                m for m in metrics_list if m in user_requested_keys
            ]
            labeled_options = [
                (lbl, val) for lbl, val in labeled_options
                if val in user_requested_keys
            ]
            if not metrics_list:
                all_keys = []
                for subset_cfg in self.subsets.values():
                    all_keys.extend(subset_cfg.keys())
                logger.warning(
                    f"No matching metrics found. Available metric keys: "
                    f"{', '.join(sorted(set(all_keys)))}"
                )
                return
            default_metrics = list(dict.fromkeys(metrics_list))
        else:
            # No --metrics flag: always CPU + Memory; add GPU if available
            default_metrics = ["cpu_summary", "memory"]
            if self.monitor.num_gpus:
                default_metrics += ["gpu_util_summary", "gpu_mem_summary"]
            # Resolve subsets needed for these metrics
            needed_subsets = ["cpu", "mem"]
            if self.monitor.num_gpus:
                needed_subsets.append("gpu")
            metric_subsets = self._resolve_metric_subsets(needed_subsets)
            _, labeled_options = self._collect_metric_options(metric_subsets)

        # Map metric keys → human-readable labels
        label_map = {val: lbl for lbl, val in labeled_options}
        if level is None:
            level = "process"
        session_id = str(uuid.uuid4())[:8]
        stop_event = threading.Event()
        ncols = 2

        # -- Display header (static) --------------------------------------- #
        header_html = f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;margin-bottom:8px">
  <b>Live Performance Monitor</b>
  <span>
    <span id="jumper-status-{session_id}"
          style="color:green;font-weight:bold">● Live</span>
    <span style="color:#666;margin-left:8px">
      window {window_seconds:.0f}s / update {update_interval}s
    </span>
  </span>
  <img src="{logo_image}" alt="JUmPER"
       style="height:auto;width:100px" onerror="this.style.display='none'">
</div>
<p style="color:#666;font-size:0.9em">
  Panels: {', '.join(label_map.get(m, m) for m in default_metrics)}
  (level: {level}).
</p>
"""
        display(HTML(header_html))

        # -- Single display handle for the whole grid ---------------------- #
        grid_id = f"jumper-live-grid-{session_id}"
        init_items = "".join(
            f"<div style='padding:4px'>"
            f"<p style='color:#888'>Initialising "
            f"{label_map.get(m, m)}…</p></div>"
            for m in default_metrics
        )
        display(
            HTML(
                f"<div style='display:grid;"
                f"grid-template-columns:repeat({ncols},1fr);"
                f"gap:8px'>{init_items}</div>"
            ),
            display_id=grid_id,
        )

        # -- Background thread --------------------------------------------- #
        def _render_grid():
            """Build all panels and return a single HTML grid string."""
            cells = []
            for metric in default_metrics:
                try:
                    fig = self._build_live_figure(
                        metric, level, window_seconds,
                    )
                    if fig is not None:
                        html_frag = pio.to_html(
                            fig,
                            full_html=False,
                            include_plotlyjs="cdn",
                            config={"responsive": True,
                                    "displayModeBar": True},
                        )
                    else:
                        html_frag = (
                            "<p style='color:#888;text-align:center'>"
                            "Waiting for data…</p>"
                        )
                except Exception:
                    logger.debug(
                        "Live plot update error for %s", metric,
                        exc_info=True,
                    )
                    html_frag = (
                        "<p style='color:#888;text-align:center'>"
                        "Waiting for data…</p>"
                    )
                cells.append(
                    f"<div style='padding:4px'>{html_frag}</div>"
                )
            return (
                f"<div style='display:grid;"
                f"grid-template-columns:repeat({ncols},1fr);"
                f"gap:8px'>{''.join(cells)}</div>"
            )

        def _live_loop():
            try:
                while not stop_event.is_set():
                    try:
                        grid_html = _render_grid()
                        display(
                            HTML(grid_html),
                            display_id=grid_id,
                            update=True,
                        )
                    except Exception:
                        logger.debug(
                            "Live plot grid update error",
                            exc_info=True,
                        )
                    if not self.monitor.running:
                        break
                    stop_event.wait(update_interval)
            finally:
                # Final render
                try:
                    grid_html = _render_grid()
                    display(
                        HTML(grid_html),
                        display_id=grid_id,
                        update=True,
                    )
                except Exception:
                    pass

        thread = threading.Thread(target=_live_loop, daemon=True)
        thread.start()

        logger.info(
            f"Live plotting started (update every {update_interval}s, "
            f"window {window_seconds:.0f}s). "
            f"Interrupt kernel or stop monitor to end."
        )


class MatplotlibPerformanceVisualizer(PerformanceVisualizer):
    """Matplotlib backend visualizer."""

    def _plot_metric(
        self,
        df,
        metric,
        cell_range=None,
        show_idle=False,
        ax: plt.Axes = None,
        level="process",
    ):
        """Plot a single metric using its configuration."""
        config = next(
            (
                subset[metric]
                for subset in self.subsets.values()
                if metric in subset
            ),
            None,
        )
        if not config or not isinstance(config, dict):
            return

        plot_type = config.get("type")
        if plot_type == "single_series":
            column = config.get("column")
            title = config.get("title", "")
            ylim = config.get("ylim")
            if metric == "memory" and ylim is None:
                ylim = (0, self.monitor.memory_limits[level])
            if not column or column not in df.columns:
                return
        elif plot_type == "multi_series":
            prefix = config.get("prefix", "")
            title = config.get("title", "")
            ylim = config.get("ylim")
            series_cols = [
                col
                for col in df.columns
                if prefix
                and col.startswith(prefix)
                and not col.endswith("avg")
            ]
            avg_column = f"{prefix}avg" if prefix else None
            if (
                avg_column is None or avg_column not in df.columns
            ) and not series_cols:
                return
        elif plot_type == "summary_series":
            columns = config.get("columns", [])
            title = config.get("title", "")
            ylim = config.get("ylim")
            if level == "system":
                title = re.sub(
                    r"\d+", str(self.monitor.num_system_cpus), title
                )
            available_cols = [col for col in columns if col in df.columns]
            if not available_cols:
                return
        else:
            return

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        if plot_type == "single_series":
            series = df[column]
            if metric in (
                "io_read",
                "io_write",
                "io_read_count",
                "io_write_count",
            ):
                diffs = df[column].astype(float).diff().clip(lower=0)
                if metric in ("io_read", "io_write"):
                    diffs = diffs / (1024**2)
                series = diffs.fillna(0.0)
                if self._io_window > 1:
                    series = series.rolling(
                        window=self._io_window, min_periods=1
                    ).mean()
            ax.plot(df["time"], series, color="blue", linewidth=2)
        elif plot_type == "summary_series":
            line_styles, alpha_vals = ["dotted", "-", "--"], [0.35, 1.0, 0.35]
            for i, (col, label) in enumerate(
                zip(columns, ["Min", "Average", "Max"])
            ):
                if col in df.columns:
                    ax.plot(
                        df["time"],
                        df[col],
                        color="blue",
                        linestyle=line_styles[i % len(line_styles)],
                        linewidth=2,
                        alpha=alpha_vals[i % len(alpha_vals)],
                        label=label,
                    )
            ax.legend()
        elif plot_type == "multi_series":
            for col in series_cols:
                ax.plot(df["time"], df[col], "-", alpha=0.5, label=col)
            if avg_column in df.columns:
                ax.plot(
                    df["time"], df[avg_column], "b-", linewidth=2, label="Mean"
                )
            ax.legend()

        ax.set_title(title + (" (No Idle)" if not show_idle else ""))
        ax.set_xlabel("Time (seconds)")
        ax.grid(True)
        if ylim:
            ax.set_ylim(ylim)
        self._draw_cell_boundaries(ax, cell_range, show_idle)

    def _draw_cell_boundaries(self, ax, cell_range=None, show_idle=False):
        """Draw cell boundaries as colored rectangles with cell indices."""
        colors = jumper_colors
        y_min, y_max = ax.get_ylim()
        x_max, height = ax.get_xlim()[1], y_max - y_min
        min_duration = self.min_duration or 0

        def draw_cell_rect(start_time, duration, cell_num, alpha):
            if (
                duration < min_duration
                or start_time > x_max
                or start_time + duration < 0
            ):
                return
            color = colors[cell_num % len(colors)]
            ax.add_patch(
                plt.Rectangle(
                    (start_time, y_min),
                    duration,
                    height,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    linestyle="--",
                    linewidth=1,
                    zorder=0,
                )
            )
            ax.text(
                start_time + duration / 2,
                y_max - height * 0.1,
                f"#{cell_num}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                zorder=1,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                ),
            )

        if not show_idle and hasattr(self, "_compressed_cell_boundaries"):
            for cell in self._compressed_cell_boundaries:
                draw_cell_rect(
                    cell["start_time"],
                    cell["duration"],
                    int(cell["cell_index"]),
                    0.4,
                )
        else:
            filtered_cells = self.cell_history.view()
            if cell_range:
                try:
                    mask = (filtered_cells["cell_index"] >= cell_range[0]) & (
                        filtered_cells["cell_index"] <= cell_range[1]
                    )
                    cells = filtered_cells[mask]
                except Exception:
                    cells = filtered_cells
            else:
                cells = filtered_cells
            for _, cell in cells.iterrows():
                start_time = cell["start_time"] - self.monitor.start_time
                draw_cell_rect(
                    start_time, cell["duration"], int(cell["cell_index"]), 0.5
                )

    def _create_interactive_wrapper(
        self,
        metrics,
        labeled_options,
        processed_perfdata,
        current_cell_range,
        current_show_idle,
    ):
        return InteractivePlotWrapper(
            self._plot_metric,
            metrics,
            labeled_options,
            processed_perfdata,
            current_cell_range,
            current_show_idle,
            self.figsize,
        )

    def _render_direct_plot(
        self,
        processed_data,
        metrics,
        cell_range,
        show_idle,
        level,
        save_jpeg=None,
        pickle_file=None,
        metric_subsets=None,
    ):
        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            n_metrics,
            1,
            figsize=(10, 3 * n_metrics),
            constrained_layout=True,
        )
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            self._plot_metric(
                processed_data, metric, cell_range, show_idle, axes[i], level
            )

        if save_jpeg:
            if not save_jpeg.endswith(".jpg") and not save_jpeg.endswith(
                ".jpeg"
            ):
                save_jpeg += ".jpg"
            fig.savefig(
                save_jpeg, format="jpeg", dpi=300, bbox_inches="tight"
            )
            print(f"Plot saved as JPEG: {save_jpeg}")

        if pickle_file:
            if not pickle_file.endswith(".pkl"):
                pickle_file += ".pkl"
            plot_data = {
                "figure": fig,
                "axes": axes,
                "metrics": metrics,
                "processed_data": processed_data,
                "cell_range": cell_range,
                "level": level,
                "show_idle": show_idle,
                "metric_subsets": metric_subsets,
            }
            with open(pickle_file, "wb") as f:
                pickle.dump(plot_data, f)

            print(f"Plot objects serialized to: {pickle_file}")
            print("\n# Python code to reload and display the plot:")
            print("import pickle")
            print("import matplotlib.pyplot as plt")
            print("")
            print(f"# Load the pickled plot data")
            print(f"with open('{pickle_file}', 'rb') as f:")
            print("    plot_data = pickle.load(f)")
            print("")
            print("# Extract the figure and display")
            print("fig = plot_data['figure']")
            print("plt.show()")
            print("")
            print("# Access other data:")
            print("# axes = plot_data['axes']")
            print("# metrics = plot_data['metrics']")
            print("# processed_data = plot_data['processed_data']")
            print("# cell_range = plot_data['cell_range']")
            print("# level = plot_data['level']")

        plt.show()


class UnavailableVisualizer:
    """
    A stub that type-checks against VisualizerProtocol but
    only logs that visualization is unavailable.
    """
    def __init__(self, reason: str = "Plotting not available."):
        self._reason = reason

    def attach(self, monitor: MonitorProtocol) -> None: ...

    def plot(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
    ) -> None:
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.PLOTS_NOT_AVAILABLE].format(
                reason=self._reason
            )
        )



class InteractivePlotWrapper:
    """Interactive plotter with dropdown selection and reusable matplotlib
    axes."""

    def __init__(
        self,
        plot_callback,
        metrics: List[str],
        labeled_options,
        perfdata_by_level,
        cell_range=None,
        show_idle=False,
        figsize=None,
    ):
        self.plot_callback, self.perfdata_by_level, self.metrics = (
            plot_callback,
            perfdata_by_level,
            metrics,
        )
        self.labeled_options = labeled_options
        self.cell_range, self.show_idle, self.figsize = (
            cell_range,
            show_idle,
            figsize,
        )
        self.shown_metrics, self.panel_count, self.max_panels = (
            set(),
            0,
            len(metrics) * 4,
        )
        # Store plot panels for updates
        self.plot_panels = []

        self.output_container = widgets.HBox(
            layout=Layout(
                display="flex",
                flex_flow="row wrap",
                align_items="center",
                justify_content="space-between",
                width="100%",
            )
        )
        self.add_panel_button = widgets.Button(
            description="Add Plot Panel",
            layout=Layout(margin="0 auto 20px auto"),
        )
        self.add_panel_button.on_click(self._on_add_panel_clicked)

    def display_ui(self):
        """Display the Add button and all interactive panels."""
        display(widgets.VBox([self.add_panel_button, self.output_container]))
        self._on_add_panel_clicked(None)

    def _on_add_panel_clicked(self, _):
        """Add a new plot panel with dropdown and persistent matplotlib
        axis."""
        if self.panel_count >= self.max_panels:
            self.add_panel_button.disabled = True
            self.output_container.children += (
                widgets.HTML("<b>All panels have been added.</b>"),
            )
            return

        self.output_container.children += (
            widgets.HBox(
                [
                    self._create_dropdown_plot_panel(),
                    self._create_dropdown_plot_panel(),
                ],
            ),
        )
        self.panel_count += 2

        if self.panel_count >= self.max_panels:
            self.add_panel_button.disabled = True

    def _create_dropdown_plot_panel(self):
        """Create metric and level dropdown + matplotlib figure panel with
        persistent Axes."""
        metric_dropdown = widgets.Dropdown(
            options=self.labeled_options,
            value=self._get_next_metric(),
            description="Metric:",
        )
        level_dropdown = widgets.Dropdown(
            options=get_available_levels(),
            value="process",
            description="Level:",
        )
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
        if not is_ipympl_backend():
            plt.close(fig)
        output = widgets.Output()

        def update_plot():
            metric = metric_dropdown.value
            level = level_dropdown.value
            df = self.perfdata_by_level.get(level)
            if not is_ipympl_backend():
                output.clear_output(wait=True)
            with output:
                ax.clear()
                if df is not None and not df.empty:
                    self.plot_callback(
                        df, metric, self.cell_range, self.show_idle, ax, level
                    )
                fig.canvas.draw_idle()
                if not is_ipympl_backend():
                    display(fig)

        def on_dropdown_change(change):
            if change["type"] == "change" and change["name"] == "value":
                update_plot()

        metric_dropdown.observe(on_dropdown_change)
        level_dropdown.observe(on_dropdown_change)

        # Store panel data for updates
        panel_data = {
            "metric_dropdown": metric_dropdown,
            "level_dropdown": level_dropdown,
            "figure": fig,
            "axes": ax,
            "output": output,
            "update_plot": update_plot,
        }
        self.plot_panels.append(panel_data)

        # Initial plot
        update_plot()
        if is_ipympl_backend():
            with output:
                plt.show()

        return widgets.VBox(
            [widgets.HBox([metric_dropdown, level_dropdown]), output]
        )

    def _get_next_metric(self):
        for metric in self.metrics:
            if metric not in self.shown_metrics:
                self.shown_metrics.add(metric)
                return metric
        return None

    def update_data(self, perfdata_by_level, cell_range, show_idle):
        self.perfdata_by_level = perfdata_by_level
        self.cell_range = cell_range
        self.show_idle = show_idle
        for panel in self.plot_panels:
            panel["output"].clear_output(wait=True)
            panel["update_plot"]()


class PlotlyPerformanceVisualizer(PerformanceVisualizer):
    """Plotly-based visualizer compatible with VisualizerProtocol."""

    def _build_metric_plot(self, df, metric, show_idle=False, level="process"):
        config = next(
            (
                subset[metric]
                for subset in self.subsets.values()
                if metric in subset
            ),
            None,
        )
        if not config or not isinstance(config, dict):
            return None

        traces = []
        y_values = []
        plot_type = config.get("type")
        title = config.get("title", "")
        ylim = config.get("ylim")

        if plot_type == "single_series":
            column = config.get("column")
            if not column or column not in df.columns:
                return None

            series = df[column]
            if metric in (
                "io_read",
                "io_write",
                "io_read_count",
                "io_write_count",
            ):
                diffs = df[column].astype(float).diff().clip(lower=0)
                if metric in ("io_read", "io_write"):
                    diffs = diffs / (1024**2)
                series = diffs.fillna(0.0)
                if self._io_window and self._io_window > 1:
                    series = series.rolling(
                        window=self._io_window, min_periods=1
                    ).mean()

            trace = go.Scatter(
                x=df["time"].tolist(),
                y=series.tolist(),
                mode="lines",
                line=dict(color="blue", width=2),
                name=config.get("label", column),
            )
            traces.append(trace)
            y_values.extend(series.tolist())
            if metric == "memory" and ylim is None:
                ylim = (0, self.monitor.memory_limits[level])

        elif plot_type == "summary_series":
            columns = config.get("columns", [])
            if level == "system":
                title = re.sub(
                    r"\d+", str(self.monitor.num_system_cpus), title
                )
            line_styles = ["dot", "solid", "dash"]
            alpha_vals = [0.35, 1.0, 0.35]
            labels = ["Min", "Average", "Max"]

            for i, col in enumerate(columns):
                if col not in df.columns:
                    continue
                y_series = df[col]
                trace = go.Scatter(
                    x=df["time"].tolist(),
                    y=y_series.tolist(),
                    mode="lines",
                    line=dict(
                        color="blue",
                        dash=line_styles[i % len(line_styles)],
                        width=2,
                    ),
                    opacity=alpha_vals[i % len(alpha_vals)],
                    name=labels[i % len(labels)],
                )
                traces.append(trace)
                y_values.extend(y_series.tolist())

        elif plot_type == "multi_series":
            prefix = config.get("prefix", "")
            series_cols = [
                col
                for col in df.columns
                if prefix
                and col.startswith(prefix)
                and not col.endswith("avg")
            ]
            avg_column = f"{prefix}avg" if prefix else None
            if (
                avg_column is None or avg_column not in df.columns
            ) and not series_cols:
                return None

            for col in series_cols:
                y_series = df[col]
                traces.append(
                    go.Scatter(
                        x=df["time"].tolist(),
                        y=y_series.tolist(),
                        mode="lines",
                        line=dict(width=1),
                        opacity=0.5,
                        name=col,
                    )
                )
                y_values.extend(y_series.tolist())

            if avg_column in df.columns:
                avg_series = df[avg_column]
                traces.append(
                    go.Scatter(
                        x=df["time"].tolist(),
                        y=avg_series.tolist(),
                        mode="lines",
                        line=dict(color="blue", width=2),
                        name="Mean",
                    )
                )
                y_values.extend(avg_series.tolist())
        else:
            return None

        clean_values = []
        for value in y_values:
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if val != val:
                continue
            if val == float("inf") or val == float("-inf"):
                continue
            clean_values.append(val)

        if ylim is None:
            if clean_values:
                y_min = min(clean_values)
                y_max = max(clean_values)
                if y_min == y_max:
                    pad = abs(y_min) * 0.05 or 1.0
                else:
                    pad = (y_max - y_min) * 0.05
                ylim = (y_min - pad, y_max + pad)
            else:
                ylim = (0, 1)

        title = title + (" (No Idle)" if not show_idle else "")
        return {"traces": traces, "title": title, "ylim": ylim}

    def _get_plotly_cell_boundaries(self, cell_range=None, show_idle=False):
        min_duration = self.min_duration or 0
        boundaries = []

        if not show_idle and hasattr(self, "_compressed_cell_boundaries"):
            for cell in self._compressed_cell_boundaries:
                duration = cell.get("duration", 0)
                if duration < min_duration:
                    continue
                boundaries.append(
                    {
                        "start_time": float(cell["start_time"]),
                        "duration": float(duration),
                        "cell_index": int(cell["cell_index"]),
                    }
                )
            return boundaries

        filtered_cells = self.cell_history.view()
        if cell_range:
            try:
                mask = (filtered_cells["cell_index"] >= cell_range[0]) & (
                    filtered_cells["cell_index"] <= cell_range[1]
                )
                filtered_cells = filtered_cells[mask]
            except Exception:
                pass

        monitor_start = self.monitor.start_time or 0.0
        for _, cell in filtered_cells.iterrows():
            try:
                duration = float(cell["duration"])
                if duration < min_duration:
                    continue
                boundaries.append(
                    {
                        "start_time": float(cell["start_time"]) - monitor_start,
                        "duration": duration,
                        "cell_index": int(cell["cell_index"]),
                    }
                )
            except Exception:
                continue
        return boundaries

    def _draw_cell_boundaries_plotly(
        self,
        fig,
        row,
        ylim,
        cell_range=None,
        show_idle=False,
    ):
        y_min, y_max = ylim
        height = (y_max - y_min) or 1.0
        axis_suffix = "" if row == 1 else str(row)
        xref = f"x{axis_suffix}"
        yref = f"y{axis_suffix}"

        for cell in self._get_plotly_cell_boundaries(cell_range, show_idle):
            start_time = cell["start_time"]
            duration = cell["duration"]
            cell_num = cell["cell_index"]
            color = jumper_colors[cell_num % len(jumper_colors)]
            fig.add_shape(
                type="rect",
                x0=start_time,
                x1=start_time + duration,
                y0=y_min,
                y1=y_max,
                xref=xref,
                yref=yref,
                fillcolor=color,
                opacity=0.4,
                line=dict(color="black", dash="dash", width=1),
                layer="below",
            )
            fig.add_annotation(
                x=start_time + duration / 2,
                y=y_max - height * 0.1,
                xref=xref,
                yref=yref,
                text=f"#{cell_num}",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
            )

    def _compute_cell_boundaries_json(self, cell_range, show_idle):
        """Return cell boundary data as a list of plain dicts for JS consumption."""
        result = []
        for cell in self._get_plotly_cell_boundaries(cell_range, show_idle):
            cell_num = int(cell["cell_index"])
            result.append(
                {
                    "cell_index": cell_num,
                    "x0": float(cell["start_time"]),
                    "x1": float(cell["start_time"] + cell["duration"]),
                    "color": jumper_colors[cell_num % len(jumper_colors)],
                }
            )
        return result

    def _build_single_metric_figure(
        self,
        df,
        metric,
        cell_range=None,
        show_idle=False,
        level="process",
        include_boundaries=True,
    ):
        metric_plot = self._build_metric_plot(
            df, metric, show_idle=show_idle, level=level
        )
        if not metric_plot:
            return None

        fig = go.Figure()
        for trace in metric_plot["traces"]:
            fig.add_trace(trace)

        fig.update_layout(
            title=metric_plot["title"],
            xaxis_title="Time (seconds)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            margin=dict(l=24, r=8, t=45, b=35),
            # Keep width container-driven, but use a compact height close to
            # former matplotlib proportions.
            height=max(220, int(self.figsize[1] * 105)),
            autosize=True,
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True, range=list(metric_plot["ylim"]))
        if include_boundaries:
            self._draw_cell_boundaries_plotly(
                fig,
                row=1,
                ylim=metric_plot["ylim"],
                cell_range=cell_range,
                show_idle=show_idle,
            )
        return fig

    def _render_direct_plot(
        self,
        processed_data,
        metrics,
        cell_range,
        show_idle,
        level,
        save_jpeg=None,
        pickle_file=None,
        metric_subsets=None,
    ):
        prepared = []
        for metric in metrics:
            metric_plot = self._build_metric_plot(
                processed_data, metric, show_idle=show_idle, level=level
            )
            if metric_plot:
                prepared.append((metric, metric_plot))

        if not prepared:
            logger.warning("No valid metrics found to plot")
            return

        fig = make_subplots(
            rows=len(prepared),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[item[1]["title"] for item in prepared],
        )

        for row, (metric, metric_plot) in enumerate(prepared, start=1):
            for trace in metric_plot["traces"]:
                fig.add_trace(trace, row=row, col=1)
            fig.update_yaxes(
                range=list(metric_plot["ylim"]),
                showgrid=True,
                row=row,
                col=1,
            )
            fig.update_xaxes(showgrid=True, row=row, col=1)
            self._draw_cell_boundaries_plotly(
                fig,
                row=row,
                ylim=metric_plot["ylim"],
                cell_range=cell_range,
                show_idle=show_idle,
            )

        fig.update_xaxes(
            title_text="Time (seconds)",
            row=len(prepared),
            col=1,
        )
        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            # Roughly match legacy matplotlib subplot density while staying
            # responsive in width.
            height=max(260, int(270 * len(prepared))),
            margin=dict(l=24, r=8, t=40, b=35),
            autosize=True,
        )

        if save_jpeg:
            if not save_jpeg.endswith(".jpg") and not save_jpeg.endswith(
                ".jpeg"
            ):
                save_jpeg += ".jpg"
            fig.write_image(save_jpeg, format="jpeg", scale=2)
            print(f"Plot saved as JPEG: {save_jpeg}")

        if pickle_file:
            if not pickle_file.endswith(".pkl"):
                pickle_file += ".pkl"
            plot_data = {
                "figure_dict": fig.to_dict(),
                "metrics": [item[0] for item in prepared],
                "processed_data": processed_data,
                "cell_range": cell_range,
                "level": level,
                "show_idle": show_idle,
                "metric_subsets": metric_subsets,
            }
            with open(pickle_file, "wb") as f:
                pickle.dump(plot_data, f)

            print(f"Plot objects serialized to: {pickle_file}")
            print("\n# Python code to reload and display the plot:")
            print("import pickle")
            print("import plotly.graph_objects as go")
            print("")
            print(f"with open('{pickle_file}', 'rb') as f:")
            print("    plot_data = pickle.load(f)")
            print("")
            print("fig = go.Figure(plot_data['figure_dict'])")
            print("fig.show()")

        fig.show(config={"responsive": True})

    def _precompute_figures_for_wrapper(
        self,
        metrics,
        perfdata_no_idle,
        perfdata_with_idle,
        full_range,
    ):
        """Pre-compute all metric × level × show_idle figure dicts.

        Builds figures without embedded cell boundaries (JS draws them from the
        separately stored ``boundaries_false`` / ``boundaries_true`` lists).

        ``perfdata_no_idle`` must already be prepared before this call so that
        ``self._compressed_cell_boundaries`` is set to the correct full-range
        boundaries.  The with-idle figures are built second; they never read
        ``_compressed_cell_boundaries`` (gated on the ``show_idle`` flag inside
        ``_get_plotly_cell_boundaries``).
        """
        figures = {}  # metric → level → key → dict|None
        ylims   = {}  # metric → level → key → [ymin, ymax]
        all_levels = set(
            list(perfdata_no_idle or {}) + list(perfdata_with_idle or {})
        )

        for metric in metrics:
            figures[metric] = {}
            ylims[metric]   = {}
            for level in all_levels:
                figures[metric][level] = {}
                ylims[metric][level]   = {}
                # Build no-idle first (so _compressed_cell_boundaries is used)
                for key, df in [
                    ("false", (perfdata_no_idle or {}).get(level)),
                    ("true",  (perfdata_with_idle or {}).get(level)),
                ]:
                    if df is None or df.empty:
                        figures[metric][level][key] = None
                        ylims[metric][level][key]   = [0, 1]
                        continue
                    try:
                        fig = self._build_single_metric_figure(
                            df, metric, full_range,
                            show_idle=(key == "true"),
                            level=level,
                            include_boundaries=False,
                        )
                        if fig is not None:
                            fd = json.loads(fig.to_json())
                            figures[metric][level][key] = fd
                            try:
                                ylims[metric][level][key] = (
                                    fd["layout"]["yaxis"]["range"]
                                )
                            except (KeyError, TypeError):
                                ylims[metric][level][key] = [0, 1]
                        else:
                            figures[metric][level][key] = None
                            ylims[metric][level][key]   = [0, 1]
                    except Exception:
                        figures[metric][level][key] = None
                        ylims[metric][level][key]   = [0, 1]

        boundaries_false = self._compute_cell_boundaries_json(
            full_range, show_idle=False
        )
        boundaries_true = self._compute_cell_boundaries_json(
            full_range, show_idle=True
        )

        return {
            "figures":          figures,
            "ylims":            ylims,
            "boundaries_false": boundaries_false,
            "boundaries_true":  boundaries_true,
        }

    def _create_interactive_wrapper(
        self,
        metrics,
        labeled_options,
        processed_perfdata,
        current_cell_range,
        current_show_idle,
    ):
        # Kept for compatibility with the parent class contract; the Plotly
        # version delegates entirely to plot() and does not use this path.
        raise NotImplementedError(
            "_create_interactive_wrapper is not used by PlotlyPerformanceVisualizer; "
            "see plot() override instead."
        )

    def plot(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
        level=None,
        save_jpeg=None,
        pickle_file=None,
    ):
        """Plot performance metrics using a self-contained pure HTML/JS output."""
        metrics_missing = not metric_subsets
        if metrics_missing:
            metric_subsets = ("cpu", "mem", "io")
            if self.monitor.num_gpus:
                metric_subsets += ("gpu", "gpu_all")

        valid_cells = self.cell_history.view()
        if len(valid_cells) == 0:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_CELL_HISTORY]
            )
            return

        try:
            min_cell_idx = int(valid_cells["cell_index"].min())
            max_cell_idx = int(valid_cells["cell_index"].max())
        except Exception:
            min_cell_idx, max_cell_idx = 0, len(valid_cells) - 1

        # Determine the initial slider position (default: last significant cell)
        if cell_range is None:
            cell_start_index = 0
            for cell_idx in range(len(valid_cells) - 1, -1, -1):
                if valid_cells.iloc[cell_idx]["duration"] > self.min_duration:
                    cell_start_index = cell_idx
                    break
            start = int(valid_cells.iloc[cell_start_index]["cell_index"])
            end   = int(valid_cells["cell_index"].max())
            if start > end:
                start, end = end, start
            initial_cell_range = (start, end)
        else:
            initial_cell_range = (
                max(min_cell_idx, min(cell_range[0], max_cell_idx)),
                max(min_cell_idx, min(cell_range[1], max_cell_idx)),
            )

        # When a specific level is requested, use the direct (non-interactive)
        # path which supports save_jpeg and pickle_file.
        if level is not None:
            metric_subsets = self._resolve_metric_subsets(metric_subsets)
            return self._plot_direct(
                metric_subsets, initial_cell_range, show_idle, level,
                save_jpeg, pickle_file,
            )

        metric_subsets = self._resolve_metric_subsets(metric_subsets)
        metrics, labeled_options = self._collect_metric_options(metric_subsets)
        if not metrics:
            logger.warning("No valid metrics found to plot")
            return

        # Use the FULL cell range so the slider can range across all cells.
        full_range = (min_cell_idx, max_cell_idx)

        # Prepare no-idle data first — this sets _compressed_cell_boundaries
        # for the full range, which is required before building no-idle figures
        # and before calling _compute_cell_boundaries_json(show_idle=False).
        processed_no_idle = self._prepare_processed_data_for_interactive(
            full_range, False
        )
        if processed_no_idle is None:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_PERFORMANCE_DATA]
            )
            return

        # Prepare with-idle data (does not modify _compressed_cell_boundaries)
        processed_with_idle = self._prepare_processed_data_for_interactive(
            full_range, True
        )

        precomputed = self._precompute_figures_for_wrapper(
            metrics,
            processed_no_idle,
            processed_with_idle,
            full_range,
        )
        precomputed["min_cell_index"]    = min_cell_idx
        precomputed["max_cell_index"]    = max_cell_idx
        precomputed["initial_cell_range"] = list(initial_cell_range)

        perfdata_for_fallback = (
            processed_no_idle if not show_idle else processed_with_idle
        ) or {}

        wrapper = InteractivePlotlyWrapper(
            self._build_single_metric_figure,
            metrics,
            labeled_options,
            perfdata_for_fallback,
            initial_cell_range,
            show_idle,
            _precomputed_figures=precomputed,
        )
        wrapper.display_ui()


class InteractivePlotlyWrapper:
    """Interactive plotter that renders controls and figures via pure HTML/JS.

    All metric × level × show-idle combinations are pre-computed server-side
    and embedded as Plotly JSON in a single self-contained HTML block.
    The browser handles dropdown changes and the show-idle toggle without any
    Python round-trips.
    """

    _TEMPLATES_DIR = (
        Path(__file__).parent.parent / "templates" / "plotly_visualizer"
    )

    def __init__(
        self,
        plot_callback,
        metrics: List[str],
        labeled_options,
        perfdata_by_level,
        cell_range=None,
        show_idle=False,
        _precomputed_figures=None,
    ):
        self.plot_callback    = plot_callback
        self.perfdata_by_level = perfdata_by_level
        self.metrics          = metrics
        self.labeled_options  = labeled_options
        self.cell_range       = cell_range
        self.show_idle        = show_idle
        self.max_panels       = len(metrics) * 4
        # Pre-computed figures supplied by PlotlyPerformanceVisualizer; if
        # None we fall back to computing lazily from plot_callback.
        self._precomputed_figures = _precomputed_figures
        self._display_handle  = None
        self._container_id    = f"jump-vis-{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------ #
    # Public API (kept stable with the old ipywidgets implementation)     #
    # ------------------------------------------------------------------ #

    def display_ui(self):
        html = self._render_html()
        self._display_handle = display(HTML(html), display_id=True)

    def update_data(self, perfdata_by_level, cell_range, show_idle):
        self.perfdata_by_level    = perfdata_by_level
        self.cell_range           = cell_range
        self.show_idle            = show_idle
        self._precomputed_figures = None  # stale; recompute on render
        if self._display_handle is not None:
            self._display_handle.update(HTML(self._render_html()))
        else:
            self.display_ui()

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _compute_figures_from_callback(self):
        """Fallback: compute figures for the current show_idle state only.

        Both show_idle variants are attempted but the 'false' variant relies
        on ``_compressed_cell_boundaries`` being set correctly on the
        visualizer instance.  When called from ``update_data`` this may not
        hold, so only the current variant is guaranteed to be accurate.
        """
        result = {}
        levels = list(self.perfdata_by_level.keys())
        for _label, metric in self.labeled_options:
            result[metric] = {}
            for level in levels:
                df = self.perfdata_by_level.get(level)
                result[metric][level] = {}
                for key in ("true", "false"):
                    if df is None or df.empty:
                        result[metric][level][key] = None
                        continue
                    try:
                        fig = self.plot_callback(
                            df,
                            metric,
                            self.cell_range,
                            key == "true",
                            level,
                        )
                        result[metric][level][key] = (
                            json.loads(fig.to_json()) if fig is not None else None
                        )
                    except Exception:
                        result[metric][level][key] = None
        return result

    # CSS and JS components are loaded in this order.
    _CSS_COMPONENTS = [
        "toolbar",
        "show_idle_checkbox",
        "cell_range_slider",
        "add_panel_button",
        "panel",
    ]
    _JS_COMPONENTS = [
        "show_idle_checkbox",
        "cell_range_slider",
        "add_panel_button",
        "panel",
    ]

    def _read_component_file(self, component: str, ext: str) -> str:
        path = self._TEMPLATES_DIR / component / f"{component}.{ext}"
        try:
            return path.read_text(encoding="utf-8") if path.exists() else ""
        except Exception:
            return ""

    def _render_html(self):
        pre = (
            self._precomputed_figures
            if self._precomputed_figures is not None
            else self._compute_figures_from_callback()
        )

        # Unpack rich pre-computed structure or fall back to plain figures dict
        if isinstance(pre, dict) and "figures" in pre:
            figures            = pre["figures"]
            ylims              = pre.get("ylims", {})
            boundaries_false   = pre.get("boundaries_false", [])
            boundaries_true    = pre.get("boundaries_true", [])
            min_cell_index     = pre.get("min_cell_index", 0)
            max_cell_index     = pre.get("max_cell_index", 0)
            initial_cell_range = pre.get(
                "initial_cell_range", [min_cell_index, max_cell_index]
            )
        else:
            figures            = pre
            ylims              = {}
            boundaries_false   = []
            boundaries_true    = []
            min_cell_index     = 0
            max_cell_index     = 0
            initial_cell_range = [0, 0]

        levels  = get_available_levels()
        init_lo = initial_cell_range[0] if initial_cell_range else min_cell_index
        init_hi = (
            initial_cell_range[1] if len(initial_cell_range) > 1 else max_cell_index
        )

        # ── 1. Collect component CSS ──────────────────────────────────────
        css_parts = [
            self._read_component_file(c, "css") for c in self._CSS_COMPONENTS
        ]

        # ── 2. Render HTML via Jinja2 (visualizer.html includes sub-templates)
        env = Environment(
            loader=FileSystemLoader(str(self._TEMPLATES_DIR)),
            autoescape=False,
        )
        env.filters["tojson"] = json.dumps
        html_ctx = dict(
            container_id=self._container_id,
            logo_src=logo_image,
            initial_show_idle=self.show_idle,
            min_cell_index=min_cell_index,
            max_cell_index=max_cell_index,
            init_lo=init_lo,
            init_hi=init_hi,
        )
        body_html = env.get_template("visualizer.html").render(**html_ctx)

        # ── 3. Collect component JS then main orchestration script ────────
        js_parts = [
            self._read_component_file(c, "js") for c in self._JS_COMPONENTS
        ]
        main_js_path = self._TEMPLATES_DIR / "main.js"
        try:
            js_parts.append(
                main_js_path.read_text(encoding="utf-8")
                if main_js_path.exists()
                else ""
            )
        except Exception:
            pass

        # ── 4. Embedded Python data (must precede component + main JS) ────
        data_js = "\n".join([
            f"var CID      = {json.dumps(self._container_id)};",
            f"var FIGS     = {json.dumps(figures)};",
            f"var YLIMS    = {json.dumps(ylims)};",
            f"var BND_F    = {json.dumps(boundaries_false)};",
            f"var BND_T    = {json.dumps(boundaries_true)};",
            f"var OPTS     = {json.dumps(self.labeled_options)};",
            f"var LEVS     = {json.dumps(levels)};",
            f"var MAX      = {self.max_panels};",
            f"var MIN_CELL = {min_cell_index};",
            f"var MAX_CELL = {max_cell_index};",
            f"var INIT_RNG = {json.dumps(initial_cell_range)};",
        ])

        # ── 5. Plotly CDN loader (injected once per output block) ─────────
        plotly_loader = (
            "<script>\n"
            "(function(){\n"
            "  if(typeof window.Plotly!=='undefined')return;\n"
            "  var s=document.createElement('script');\n"
            "  s.src='https://cdn.plot.ly/plotly-2.35.2.min.js';\n"
            "  s.charset='utf-8';\n"
            "  document.head.appendChild(s);\n"
            "})();\n"
            "</script>"
        )

        # ── 6. Assemble final HTML ────────────────────────────────────────
        return "\n".join([
            plotly_loader,
            "<style>\n" + "\n".join(css_parts) + "\n</style>",
            body_html,
            "<script>\n" + data_js + "\n\n" + "\n\n".join(js_parts) + "\n</script>",
        ])


def build_performance_visualizer(
    cell_history: CellHistory,
    plots_disabled: bool = False,
    plots_disabled_reason: str = "Plotting not available.",
    backend: str = "matplotlib",
) -> VisualizerProtocol:
    """
    Build visualizer object with selected backend.

    Supported backends:
    - matplotlib (default)
    - plotly
    """
    if plots_disabled:
        return UnavailableVisualizer(reason=plots_disabled_reason)

    backend_name = (backend or "matplotlib").strip().lower()
    if backend_name == "plotly":
        return PlotlyPerformanceVisualizer(cell_history)
    if backend_name != "matplotlib":
        logger.warning(
            f"Unknown visualizer backend '{backend}'. "
            "Falling back to matplotlib."
        )
    return MatplotlibPerformanceVisualizer(cell_history)
