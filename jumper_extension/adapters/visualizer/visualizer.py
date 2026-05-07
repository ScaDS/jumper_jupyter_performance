import json
import logging
import uuid
import threading
import time
from pathlib import Path
from typing import List, runtime_checkable, Protocol, Optional, Tuple

import plotly.graph_objects as go
from IPython.display import display, HTML
from jinja2 import Environment, FileSystemLoader
from ipywidgets import widgets, Layout

from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.adapters.data import aggregate_node_info
from jumper_extension.adapters.visualizer.render import RENDERERS
from jumper_extension.config.models import (
    MultiSeriesConfig,
    SummarySeriesConfig,
    validate_metric_config,
)

_LINESTYLE_PLOTLY = {"solid": "solid", "dashed": "dash", "dotted": "dot"}
from jumper_extension.monitor.common import UnavailablePerformanceMonitor, \
    MonitorProtocol
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES, ExtensionInfoCode, EXTENSION_INFO_MESSAGES,
)
from jumper_extension.utilities import filter_perfdata, get_available_levels
from jumper_extension.logo import logo_image, jumper_colors

logger = logging.getLogger("extension")


@runtime_checkable
class VisualizerProtocol(Protocol):
    """Structural protocol for visualizers used by the service."""
    def attach(self, monitor: MonitorProtocol) -> None: ...
    def plot(
        self,
        metric_subsets=None,
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
        self.default_subsets: tuple[str, ...] = ("cpu", "mem", "io")
        self._hardware = None  # cached NodeInfo aggregate, populated in attach()

    def attach(
        self,
        monitor: MonitorProtocol,
        metrics_config_path=None,
    ):
        """Attach started PerformanceMonitor."""
        self.monitor = monitor
        self._hardware = aggregate_node_info(monitor.nodes.hardware)
        self.min_duration = self.monitor.interval
        try:
            self._io_window = max(
                1, int(round(1.0 / (self.monitor.interval or 1.0)))
            )
        except Exception:
            self._io_window = 1
        config_path = metrics_config_path or self._default_config_path()
        self._load_subsets_from_config(config_path)
        self._patch_hardware_dependent_ylims()

    @staticmethod
    def _default_config_path() -> Path:
        return (
            Path(__file__).parent.parent.parent
            / "config" / "plots.yaml"
        )

    def _load_subsets_from_config(self, path) -> None:
        import yaml
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw_defaults = raw.get("default_subsets", ["cpu", "mem", "io"])
        self.default_subsets = tuple(str(s) for s in raw_defaults)
        self.subsets = {}
        for subset_key, metrics in raw.get("subsets", {}).items():
            self.subsets[subset_key] = {}
            for metric_key, cfg in metrics.items():
                if cfg is None:
                    cfg = {}
                self.subsets[subset_key][metric_key] = validate_metric_config(cfg)

    def _patch_hardware_dependent_ylims(self) -> None:
        """Fill in ylim fields that depend on hardware (gpu_memory)."""
        hw = self._hardware
        gpu_mem_ylim = (0.0, float(hw.gpu_memory)) if hw.gpu_memory else None
        for subset_dict in self.subsets.values():
            for cfg in subset_dict.values():
                if cfg.ylim is None and isinstance(
                    cfg, (MultiSeriesConfig, SummarySeriesConfig)
                ):
                    if hasattr(cfg, "prefix") and "gpu_mem" in getattr(cfg, "prefix", ""):
                        cfg.ylim = gpu_mem_ylim
                    elif hasattr(cfg, "columns") and any(
                        "gpu_mem" in c for c in getattr(cfg, "columns", [])
                    ):
                        cfg.ylim = gpu_mem_ylim

    def _resolve_metric_subsets(
        self,
        metrics: Optional[List[str]]
    ) -> Tuple[str, ...]:
        """Map user-specified metrics or subsets to visualizer subset keys."""
        if not metrics:
            return self.default_subsets

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
        return deduped or self.default_subsets

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
                    label = getattr(cfg, "label", metric_key) or metric_key
                    labeled_options.append((label, metric_key))
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
            self.monitor.nodes.view(level=level),
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
                self.monitor.nodes.view(level=available_level),
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
        metric_subsets=None,
        cell_range=None,
        show_idle=False,
        level=None,
        save_jpeg=None,
        pickle_file=None,
    ):
        metrics_missing = not metric_subsets
        if metrics_missing:
            metric_subsets = self.default_subsets
            if self._hardware and self._hardware.num_gpus:
                gpu_extra = tuple(
                    s for s in ("gpu", "gpu_all")
                    if s not in metric_subsets
                )
                metric_subsets = metric_subsets + gpu_extra

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

        metric_subsets = self._resolve_metric_subsets(metric_subsets)

        # If level is specified, plot directly without widgets
        if level is not None:
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
        df = self.monitor.nodes.view(level=level)
        now = time.perf_counter() - self.monitor.start_time
        t_start = max(0, now - window_seconds)
        t_end = now

        config = next(
            (subset[metric] for subset in self.subsets.values()
             if metric in subset), None,
        )
        if config is None:
            return None

        render_fn = RENDERERS.get(config.type)
        if render_fn is None:
            return None

        fig = go.Figure()
        y_values: list = []

        has_data = df is not None and not df.empty
        if has_data:
            df = df.copy()
            df["time"] = df["time"] - self.monitor.start_time
            df = df[df["time"] >= t_start]
            has_data = not df.empty

        result = render_fn(df, config, level, self._hardware, self._io_window) if has_data else None

        if result is not None:
            for item in result.series:
                fig.add_trace(go.Scatter(
                    x=df["time"],
                    y=item.data,
                    name=item.label,
                    mode="lines",
                    opacity=item.opacity,
                    line=dict(
                        color=item.color,
                        dash=_LINESTYLE_PLOTLY.get(item.linestyle, "solid"),
                        width=item.width,
                    ),
                ))
                y_values.extend(item.data.tolist())

        ylim = result.ylim if result is not None else config.ylim

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

        title = result.title if result is not None else config.title

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
        metric_subsets=None,
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
                metric_subsets = self.default_subsets
                if self._hardware and self._hardware.num_gpus:
                    gpu_extra = tuple(
                        s for s in ("gpu", "gpu_all")
                        if s not in metric_subsets
                    )
                    metric_subsets = metric_subsets + gpu_extra
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
            if self._hardware and self._hardware.num_gpus:
                default_metrics += ["gpu_util_summary", "gpu_mem_summary"]
            # Resolve subsets needed for these metrics
            needed_subsets = ["cpu", "mem"]
            if self._hardware and self._hardware.num_gpus:
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
        metric_subsets=None,
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
        return PlotlyPerformanceVisualizer(cell_history)
    if backend_name != "matplotlib":
        logger.warning(
            f"Unknown visualizer backend '{backend}'. "
            "Falling back to matplotlib."
        )
    from jumper_extension.adapters.visualizer.backends.matplotlib import MatplotlibPerformanceVisualizer
    return MatplotlibPerformanceVisualizer(cell_history)
