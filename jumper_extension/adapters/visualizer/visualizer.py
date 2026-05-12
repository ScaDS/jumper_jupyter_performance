import json
import logging
import re
import uuid
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, runtime_checkable, Protocol, Optional, Tuple

import plotly.graph_objects as go
from IPython.display import display, HTML
from jinja2 import Environment, FileSystemLoader
from ipywidgets import widgets, Layout

from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.monitor.common import UnavailablePerformanceMonitor, \
    MonitorProtocol
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES, ExtensionInfoCode, EXTENSION_INFO_MESSAGES,
)
from jumper_extension.utilities import filter_perfdata, get_available_levels
from jumper_extension.logo import logo_image, jumper_colors
from jumper_extension.bali_adapter import BaliVisualizationMixin

logger = logging.getLogger("extension")


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


class PerformanceVisualizer(BaliVisualizationMixin):
    """Visualizes performance metrics collected by PerformanceMonitor.

    Supports multiple levels: 'user', 'process' (default), 'system', and
    'slurm' (if available)
    """

    def __init__(self, cell_history: CellHistory, bali_adapter=None):
        self.monitor = UnavailablePerformanceMonitor(
            reason="Monitor has not been started yet."
        )
        self.cell_history = cell_history
        self.figsize = (5, 3)
        self.min_duration = None
        self._io_window = None
        self.subsets = {}

        # Initialize BALI functionality via mixin
        super().__init__(bali_adapter=bali_adapter)

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
                "gpu_power": {
                    "type": "multi_series",
                    "prefix": "gpu_power_",
                    "title": "GPU Power Usage (W) - Across GPUs",
                    "ylim": None,
                    "label": "GPU Power (All GPUs)",
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
                "gpu_power_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_power_min",
                        "gpu_power_avg",
                        "gpu_power_max",
                    ],
                    "title": (
                        "GPU Power Usage (W) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": None,
                    "label": "GPU Power Summary",
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
            cell_perfdata = perfdata.loc[cell_mask.values]

            if not cell_perfdata.empty:
                original_start, cell_duration = (
                    cell["start_time"],
                    cell["end_time"] - cell["start_time"],
                )
                compressed_perfdata.loc[cell_mask.values, "time"] = current_time + (
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
        current_show_bali=False,
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
        show_bali=False,
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
        show_bali_checkbox = widgets.Checkbox(
            value=show_bali, description="Show BALI segments"
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
                show_bali_checkbox,
                cell_range_slider,
                logo_widget,
            ],
            layout=box_layout,
        )
        plot_output = widgets.Output()
        plot_wrapper = None

        def update_plots():
            nonlocal plot_wrapper
            current_cell_range, current_show_idle, current_show_bali = (
                cell_range_slider.value,
                show_idle_checkbox.value,
                show_bali_checkbox.value,
            )
            # Invalidate cache so segments are loaded once per render cycle
            self._invalidate_bali_cache()
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

            # Handle BALI segments compression
            if current_show_bali and not current_show_idle:
                bali_segments = self._load_bali_segments()
                primary_level = get_available_levels()[0]
                reference_perfdata = processed_perfdata.get(primary_level)
                self._compressed_bali_segments = (
                    self.bali_adapter.compress_segments(
                        bali_segments,
                        current_cell_range,
                        reference_perfdata,
                        self.cell_history,
                        compressed_cell_boundaries=getattr(
                            self, "_compressed_cell_boundaries", None
                        ),
                    )
                    if reference_perfdata is not None
                    and not reference_perfdata.empty
                    else []
                )

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
                # Recreate wrapper if needed or BALI state changed
                if (
                    plot_wrapper is None
                    or getattr(plot_wrapper, "show_bali", None)
                    != current_show_bali
                ):
                    plot_output.clear_output()
                    plot_wrapper = self._create_interactive_wrapper(
                        metrics,
                        labeled_options,
                        processed_perfdata,
                        current_cell_range,
                        current_show_idle,
                        current_show_bali,
                    )
                    plot_wrapper.display_ui()
                else:
                    plot_wrapper.update_data(
                        processed_perfdata,
                        current_cell_range,
                        current_show_idle,
                    )

        for widget in [
            show_idle_checkbox,
            show_bali_checkbox,
            cell_range_slider,
        ]:
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
        template_dir = (
            Path(__file__).parent.parent.parent
            / "templates" / "visualizer" / "plotly"
        )
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
        )
        header_css = (template_dir / "live_header" / "live_header.css").read_text(encoding="utf-8")
        header_html = env.get_template("live_header/live_header.html").render(
            session_id=session_id,
            window_seconds=f"{window_seconds:.0f}",
            update_interval=update_interval,
            logo_src=logo_image,
            panel_labels=", ".join(label_map.get(m, m) for m in default_metrics),
            level=level,
        )
        display(HTML(f"<style>{header_css}</style>\n{header_html}"))

        # -- Fixed panel IDs — stable for the entire live session ---------- #
        panel_ids = {m: f"jumper-live-{session_id}-{m}" for m in default_metrics}
        grid_id = f"jumper-live-grid-{session_id}"
        update_id = f"jumper-live-update-{session_id}"
        grid_css = (template_dir / "live_grid" / "live_grid.css").read_text(encoding="utf-8")

        # Skeleton grid displayed once — the Plotly divs stay in DOM for the
        # whole session so Plotly.react() updates them in-place each tick.
        init_html = (
            f"<style>{grid_css}</style>\n"
            + env.get_template("live_grid/live_grid.html").render(
                ncols=ncols,
                panels=[(panel_ids[m], label_map.get(m, m)) for m in default_metrics],
            )
        )
        display(HTML(init_html), display_id=grid_id)
        # Initialise the script-update slot before the first update=True call.
        display(HTML(""), display_id=update_id)

        # -- Background thread --------------------------------------------- #
        def _render_update():
            """Build Plotly.react() script tags for each panel via template.

            Plotly.react() updates the existing chart in-place; if the div has
            no chart yet (first call) it initialises one like newPlot(), so no
            separate first-render path is needed.
            """
            updates = []
            for metric in default_metrics:
                try:
                    fig = self._build_live_figure(metric, level, window_seconds)
                    if fig is None:
                        continue
                    fig_dict = fig.to_dict()
                    updates.append((
                        panel_ids[metric],
                        json.dumps(fig_dict["data"]),
                        json.dumps(fig_dict["layout"]),
                        json.dumps({"responsive": True, "displayModeBar": True}),
                    ))
                except Exception:
                    logger.debug("Live plot update error for %s", metric, exc_info=True)
            return env.get_template("live_update/live_update.html").render(updates=updates)

        def _live_loop():
            try:
                while not stop_event.is_set():
                    try:
                        display(HTML(_render_update()), display_id=update_id, update=True)
                    except Exception:
                        logger.debug("Live plot grid update error", exc_info=True)
                    if not self.monitor.running:
                        break
                    stop_event.wait(update_interval)
            finally:
                try:
                    display(HTML(_render_update()), display_id=update_id, update=True)
                except Exception:
                    pass

        thread = threading.Thread(target=_live_loop, daemon=True)
        thread.start()

        logger.info(
            f"Live plotting started (update every {update_interval}s, "
            f"window {window_seconds:.0f}s). "
            f"Interrupt kernel or stop monitor to end."
        )


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



def build_performance_visualizer(
    cell_history: CellHistory,
    plots_disabled: bool = False,
    plots_disabled_reason: str = "Plotting not available.",
    backend: str = "matplotlib",
    bali_adapter=None,
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
        from jumper_extension.adapters.visualizer.backends.plotly import PlotlyPerformanceVisualizer
        return PlotlyPerformanceVisualizer(cell_history, bali_adapter=bali_adapter)
    if backend_name != "matplotlib":
        logger.warning(
            f"Unknown visualizer backend '{backend}'. "
            "Falling back to matplotlib."
        )
    from jumper_extension.adapters.visualizer.backends.matplotlib import MatplotlibPerformanceVisualizer
    return MatplotlibPerformanceVisualizer(cell_history, bali_adapter=bali_adapter)
