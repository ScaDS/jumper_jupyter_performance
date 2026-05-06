import json
import logging
import pickle
import re
import uuid
from pathlib import Path
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader, select_autoescape
from IPython.display import display, HTML

from jumper_extension.adapters.visualizer.visualizer import PerformanceVisualizer
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)
from jumper_extension.utilities import get_available_levels
from jumper_extension.logo import jumper_colors, logo_image

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
