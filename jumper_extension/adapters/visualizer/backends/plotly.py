import json
import logging
import pickle
import uuid
from pathlib import Path
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader, select_autoescape
from IPython.display import display, HTML

from jumper_extension.adapters.visualizer.render import RENDERERS
from jumper_extension.adapters.visualizer.visualizer import PerformanceVisualizer
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
from jumper_extension.utilities import get_available_levels
from jumper_extension.logo import jumper_colors, logo_image

_LINESTYLE_PLOTLY = {"solid": "solid", "dashed": "dash", "dotted": "dot"}

logger = logging.getLogger("extension")


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
        if config is None:
            return None

        render_fn = RENDERERS.get(config.type)
        if render_fn is None:
            return None
        result = render_fn(df, config, level, self._hardware, self._io_window)
        if result is None:
            return None

        traces = []
        y_values = []
        for item in result.series:
            traces.append(go.Scatter(
                x=df["time"].tolist(),
                y=item.data.tolist(),
                mode="lines",
                line=dict(
                    color=item.color,
                    dash=_LINESTYLE_PLOTLY.get(item.linestyle, "solid"),
                    width=item.width,
                ),
                opacity=item.opacity,
                name=item.label,
            ))
            y_values.extend(item.data.tolist())

        ylim = result.ylim
        if ylim is None:
            clean_values = [
                float(value) for value in y_values
                if value == value
                and float(value) not in (float("inf"), float("-inf"))
            ]
            if clean_values:
                y_min, y_max = min(clean_values), max(clean_values)
                pad = (y_max - y_min) * 0.05 if y_min != y_max else abs(y_min) * 0.05 or 1.0
                ylim = (y_min - pad, y_max + pad)
            else:
                ylim = (0, 1)

        title = result.title + (" (No Idle)" if not show_idle else "")
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

    def _prepare_processed_data_for_node(self, node_name, cell_range, show_idle):
        """Like _prepare_processed_data_for_interactive but restricted to one node."""
        from jumper_extension.utilities import filter_perfdata, get_available_levels
        start_idx, end_idx = cell_range
        cells_all = self.cell_history.view()
        try:
            mask = (
                (cells_all["cell_index"] >= start_idx)
                & (cells_all["cell_index"] <= end_idx)
            )
            filtered_cells = cells_all[mask]
        except Exception:
            filtered_cells = cells_all

        perfdata_by_level = {}
        for available_level in get_available_levels():
            perfdata_by_level[available_level] = filter_perfdata(
                filtered_cells,
                self.monitor.nodes.view(level=available_level, node=node_name),
                not show_idle,
            )

        if all(df.empty for df in perfdata_by_level.values()):
            return None

        processed_perfdata = {}
        for level_key, perfdata in perfdata_by_level.items():
            if not perfdata.empty:
                if not show_idle:
                    # Discard returned boundaries — aggregate boundaries are
                    # already snapshotted in _precompute_figures_for_wrapper.
                    processed_data, _ = self._compress_time_axis(perfdata, cell_range)
                    processed_perfdata[level_key] = processed_data
                else:
                    processed_data = perfdata.copy()
                    processed_data["time"] -= self.monitor.start_time
                    processed_perfdata[level_key] = processed_data
            else:
                processed_perfdata[level_key] = perfdata

        return processed_perfdata

    def _build_node_figures(self, metrics, perfdata_no_idle, perfdata_with_idle, full_range):
        """Build metric × level × show_idle figure dicts for one node (or aggregate).

        Returns ``(figures, ylims)`` where each is ``{metric: {level: {key: …}}}``.
        """
        figures = {}
        ylims   = {}
        all_levels = set(
            list(perfdata_no_idle or {}) + list(perfdata_with_idle or {})
        )
        for metric in metrics:
            figures[metric] = {}
            ylims[metric]   = {}
            for level in all_levels:
                figures[metric][level] = {}
                ylims[metric][level]   = {}
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
        return figures, ylims

    def _precompute_figures_for_wrapper(
        self,
        metrics,
        perfdata_no_idle,
        perfdata_with_idle,
        full_range,
        node_names=None,
    ):
        """Pre-compute all node × metric × level × show_idle figure dicts.

        ``FIGS`` always has the shape ``{node: {metric: {level: {key: …}}}}``.
        The ``""`` key holds the aggregate view; each hostname key holds the
        per-node view.  Single-node sessions produce ``["local"]`` in
        ``node_names`` — the structure is identical, the node-selector dropdown
        is simply hidden by the frontend.

        Boundaries are snapshotted at the top of this method while
        ``_compressed_cell_boundaries`` still reflects the aggregate no-idle
        data prepared in ``plot()``; per-node ``_compress_time_axis`` calls
        intentionally discard their returned boundaries.
        """
        boundaries_false = self._compute_cell_boundaries_json(
            full_range, show_idle=False
        )
        boundaries_true = self._compute_cell_boundaries_json(
            full_range, show_idle=True
        )

        figures = {}
        ylims   = {}

        figures[""], ylims[""] = self._build_node_figures(
            metrics, perfdata_no_idle, perfdata_with_idle, full_range
        )
        for node_name in (node_names or []):
            node_no_idle   = self._prepare_processed_data_for_node(
                node_name, full_range, False
            )
            node_with_idle = self._prepare_processed_data_for_node(
                node_name, full_range, True
            )
            figures[node_name], ylims[node_name] = self._build_node_figures(
                metrics, node_no_idle, node_with_idle, full_range
            )

        return {
            "figures":          figures,
            "ylims":            ylims,
            "boundaries_false": boundaries_false,
            "boundaries_true":  boundaries_true,
            "node_names":       node_names or [],
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
        metric_subsets=None,
        cell_range=None,
        show_idle=False,
        level=None,
        save_jpeg=None,
        pickle_file=None,
    ):
        """Plot performance metrics using a self-contained pure HTML/JS output."""
        metrics_missing = not metric_subsets
        if metrics_missing:
            metric_subsets = self.default_subsets
            if self._hardware and self._hardware.num_gpus:
                gpu_extra = tuple(
                    s for s in ("gpu", "gpu_all")
                    if s not in metric_subsets
                )
                metric_subsets = metric_subsets + gpu_extra

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

        node_names = self.monitor.nodes.node_names()
        precomputed = self._precompute_figures_for_wrapper(
            metrics,
            processed_no_idle,
            processed_with_idle,
            full_range,
            node_names=node_names,
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
            node_names=node_names,
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
        Path(__file__).parent.parent.parent.parent / "templates" / "visualizer" / "plotly"
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
        node_names=None,
    ):
        self.plot_callback     = plot_callback
        self.perfdata_by_level = perfdata_by_level
        self.metrics           = metrics
        self.labeled_options   = labeled_options
        self.cell_range        = cell_range
        self.show_idle         = show_idle
        self.max_panels        = len(metrics) * 4
        self._node_names       = node_names or []
        # Pre-computed figures supplied by PlotlyPerformanceVisualizer; if
        # None we fall back to computing lazily from plot_callback.
        self._precomputed_figures = _precomputed_figures
        self._display_handle   = None
        self._container_id     = f"jump-vis-{uuid.uuid4().hex[:8]}"

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

        Returns the same ``{node: {metric: …}}`` shape used by precomputed
        figures, with the flat figures stored under the ``""`` (aggregate) key.
        """
        flat = {}
        levels = list(self.perfdata_by_level.keys())
        for _label, metric in self.labeled_options:
            flat[metric] = {}
            for level in levels:
                df = self.perfdata_by_level.get(level)
                flat[metric][level] = {}
                for key in ("true", "false"):
                    if df is None or df.empty:
                        flat[metric][level][key] = None
                        continue
                    try:
                        fig = self.plot_callback(
                            df,
                            metric,
                            self.cell_range,
                            key == "true",
                            level,
                        )
                        flat[metric][level][key] = (
                            json.loads(fig.to_json()) if fig is not None else None
                        )
                    except Exception:
                        flat[metric][level][key] = None
        return {"": flat}

    # CSS and JS components are loaded in this order.
    _CSS_COMPONENTS = [
        "toolbar",
        "show_idle_checkbox",
        "cell_range_slider",
        "node_selector",
        "add_panel_button",
        "panel",
    ]
    _JS_COMPONENTS = [
        "show_idle_checkbox",
        "cell_range_slider",
        "node_selector",
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
            node_names = pre.get("node_names", self._node_names)
        else:
            figures            = {"": pre}
            ylims              = {}
            boundaries_false   = []
            boundaries_true    = []
            min_cell_index     = 0
            max_cell_index     = 0
            initial_cell_range = [0, 0]
            node_names         = self._node_names

        is_multinode = len(node_names) > 1
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
            is_multinode=is_multinode,
            node_names=node_names,
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
            f"var NODES    = {json.dumps(node_names)};",
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
