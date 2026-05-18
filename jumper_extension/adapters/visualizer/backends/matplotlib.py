import pickle
import re
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from IPython.display import display
from ipywidgets import widgets, Layout

from jumper_extension.adapters.visualizer.visualizer import PerformanceVisualizer
from jumper_extension.utilities import get_available_levels
from jumper_extension.logo import jumper_colors


def is_ipympl_backend():
    try:
        backend = plt.get_backend().lower()
    except Exception:
        return False
    return ("ipympl" in backend) or ("widget" in backend)


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
        show_bali=False,
        custom_vmin_vmax=None,
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
        if not show_bali:
            self._draw_cell_boundaries(ax, cell_range, show_idle)

        self._draw_bali_segments(
            ax, show_bali, show_idle, cell_range, custom_vmin_vmax, metric
        )

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
        current_show_bali=False,
    ):
        wrapper = InteractivePlotWrapper(
            self._plot_metric,
            metrics,
            labeled_options,
            processed_perfdata,
            current_cell_range,
            current_show_idle,
            current_show_bali,
            self.figsize,
        )
        wrapper.monitor = self.monitor
        return wrapper

    def _draw_bali_segments(
        self,
        ax,
        show_bali=False,
        show_idle=True,
        cell_range=None,
        custom_vmin_vmax=None,
        metric=None,
    ):
        """Draw BALI segments as colored rectangles with click selection."""
        if not show_bali:
            return
        segments = self._load_bali_segments()
        if not segments:
            return

        y_min, y_max = ax.get_ylim()
        x_max, height = ax.get_xlim()[1], y_max - y_min

        # Use compressed or adjusted segments
        if not show_idle and hasattr(self, "_compressed_bali_segments"):
            draw_segments = self._compressed_bali_segments
        else:
            draw_segments = [
                {**s, "start_time": s["start_time"] - self.monitor.start_time}
                for s in segments
            ]

        # Use custom vmin/vmax if provided, otherwise use data range
        if custom_vmin_vmax:
            vmin, vmax = custom_vmin_vmax
        else:
            vmin, vmax = self.bali_adapter.get_tokens_per_sec_range(segments)

        # TODO: no hardcoded vmin and max
        vmin_e, vmax_e = 0, 30

        # Clean up previous event handlers
        if hasattr(ax, "_bali_click"):
            ax.figure.canvas.mpl_disconnect(ax._bali_click)

        ax._bali_patches = []
        ax._bali_selected_patch = None

        is_power_metric = metric in ("gpu_power_summary", "gpu_power")
        for s in draw_segments:
            if is_power_metric:
                start, dur, tps, is_error = (
                    s["start_time"],
                    s["duration"],
                    s.get("token_per_joule_full_segment"),
                    s.get("is_error", False),
                )
            else:
                start, dur, tps, is_error = (
                    s["start_time"],
                    s["duration"],
                    s.get("tokens_per_sec"),
                    s.get("is_error", False),
                )
            if start > x_max or start + dur < 0:
                continue

            if is_error or tps is None:
                color = "none"
                hatch = None
                edgecolor = "gray"
                alpha = 1.0
                is_error_segment = True
            else:
                if is_power_metric:
                    color = self.bali_adapter.get_color_for_energy_efficiency(
                        tps, vmin_e, vmax_e
                    )
                else:
                    color = self.bali_adapter.get_color_for_tokens_per_sec(
                        tps, vmin, vmax
                    )
                hatch = None
                edgecolor = "gray"
                alpha = 0.75
                is_error_segment = False

            rect = plt.Rectangle(
                (start, y_min),
                dur,
                height,
                facecolor=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linestyle="--",
                linewidth=1,
                hatch=hatch,
                zorder=0.5,
            )
            rect.set_edgecolor(edgecolor)
            rect._bali_info = {
                "Model": s.get("model", "n/a"),
                "Framework": s.get("framework", "n/a"),
                "Batch Size": s.get("batch_size", "n/a"),
                "Input Length": s.get("input_len", "n/a"),
                "Output Length": s.get("output_len", "n/a"),
                "Output Tokens per Second": (
                    f"{tps:.2f}" if tps else "NaN"
                ),
                "Segment Throughput (Tok/s)": s.get(
                    "segment_throughput", "n/a"
                ),
                "Text Generation Throughput (Tok/s)": s.get(
                    "text_gen_throughput", "n/a"
                ),
                "Segment Energy Efficiency (Tok/J)": s.get(
                    "token_per_joule_full_segment", "n/a"
                ),
                "Text Generation Energy Efficiency": s.get(
                    "token_per_joule_text_gen", "n/a"
                ),
                "Duration": f"{s.get('duration', 0):.2f}",
                "Duration Text generation": (
                    f"{s.get('duration_text_gen', 0):.2f}"
                ),
                "Error": is_error,
                "Error Message": (
                    s.get("error_message", "") if is_error else None
                ),
            }
            rect._original_style = {
                "facecolor": color,
                "edgecolor": edgecolor,
                "hatch": hatch,
                "linewidth": 1,
                "is_error_segment": is_error_segment,
            }
            ax.add_patch(rect)
            ax._bali_patches.append(rect)

        def on_click(event):
            if event.inaxes != ax:
                return
            for patch in ax._bali_patches:
                if patch.contains(event)[0]:
                    if ax._bali_selected_patch:
                        prev_patch = ax._bali_selected_patch
                        original = prev_patch._original_style
                        prev_patch.set_hatch(original["hatch"])
                        prev_patch.set_linewidth(original["linewidth"])
                        prev_patch.set_edgecolor(original["edgecolor"])
                        prev_patch.set_facecolor(original["facecolor"])

                    patch.set_hatch("///")
                    patch.set_linewidth(1.5)
                    patch.set_edgecolor("black")
                    ax._bali_selected_patch = patch

                    if hasattr(ax, "_bali_selection_output"):
                        with ax._bali_selection_output:
                            ax._bali_selection_output.clear_output(wait=True)
                            info = patch._bali_info
                            display(
                                pd.DataFrame([info]).T.rename(
                                    columns={0: "Value"}
                                )
                            )

                    ax.figure.canvas.draw_idle()
                    return

        ax._bali_click = ax.figure.canvas.mpl_connect(
            "button_press_event", on_click
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
        show_bali=False,
        figsize=None,
    ):
        self.plot_callback, self.perfdata_by_level, self.metrics = (
            plot_callback,
            perfdata_by_level,
            metrics,
        )
        self.labeled_options = labeled_options
        self.cell_range, self.show_idle, self.show_bali, self.figsize = (
            cell_range,
            show_idle,
            show_bali,
            figsize,
        )
        self.shown_metrics, self.panel_count, self.max_panels = (
            set(),
            0,
            len(metrics) * 4,
        )
        # Store plot panels for updates
        self.plot_panels = []
        # Initialize custom_vmin_vmax for all instances
        self.custom_vmin_vmax = None
        # The owning visualizer sets this so BALI lookups can resolve
        # ``monitor.bali_pid_directory``.  Defaulted here for safety.
        self.monitor = None

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

        # BALI colorbar components
        if show_bali:
            self.bali_colorbar_output = widgets.Output()
            self.vmin_widget = widgets.FloatText(
                value=0.0, description="vmin:", step=0.1
            )
            self.vmax_widget = widgets.FloatText(
                value=100.0, description="vmax:", step=0.1
            )
            self.bali_colorbar_container = widgets.HBox(
                [
                    self.vmin_widget,
                    self.vmax_widget,
                    self.bali_colorbar_output,
                ],
                layout=Layout(
                    display="flex",
                    justify_content="center",
                    width="100%",
                ),
            )
            # Reuse the BALI adapter from the parent visualizer
            self.bali_adapter = getattr(
                self.plot_callback.__self__, "bali_adapter", None
            )
        else:
            self.bali_colorbar_output = None
            self.bali_colorbar_container = None
            self.bali_adapter = None
            self.vmin_widget = None
            self.vmax_widget = None

    def _create_bali_colorbar(self):
        """Create and display the BALI colorbar."""
        if (
            not self.show_bali
            or not self.bali_colorbar_output
            or not self.bali_adapter
            or self.monitor is None
        ):
            return

        # ``bali_pid_directory`` is only present on the BALI-aware monitor
        # backend; fall back to ``pid`` for the other monitor backends.
        bali_pid = getattr(self.monitor, "bali_pid_directory", None) \
            or getattr(self.monitor, "pid", None)
        if bali_pid is None:
            return
        segments = self.bali_adapter.get_segments_for_visualization(bali_pid)
        vmin_e, vmax_e = 0, 30

        if not segments:
            return

        # Use custom vmin/vmax if available, otherwise use data range
        if self.custom_vmin_vmax:
            vmin, vmax = self.custom_vmin_vmax
        else:
            vmin, vmax = self.bali_adapter.get_tokens_per_sec_range(segments)
            # Initialize widgets with data range on first creation
            if self.vmin_widget and self.vmax_widget:
                self.vmin_widget.value = vmin
                self.vmax_widget.value = vmax
                self.custom_vmin_vmax = (vmin, vmax)

        with self.bali_colorbar_output:
            self.bali_colorbar_output.clear_output(wait=True)
            fig = plt.figure(figsize=(8, 0.8))
            ax = fig.add_subplot(111)
            ax.set_visible(False)

            sm = ScalarMappable(
                norm=Normalize(vmin=vmin, vmax=vmax),
                cmap=self.bali_adapter.get_colormap(),
            )
            sm.set_array([])
            cbar = fig.colorbar(
                sm,
                cax=fig.add_axes([0.1, 0.3, 0.8, 0.4]),
                orientation="horizontal",
            )
            cbar.set_label("Output Tokens/Second", fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_facecolor("none")

            sm_energy = ScalarMappable(
                norm=Normalize(vmin=vmin_e, vmax=vmax_e),
                cmap=self.bali_adapter.get_energy_colormap(),
            )
            sm_energy.set_array([])
            cbar_e = fig.colorbar(
                sm_energy,
                cax=fig.add_axes([0.1, -0.8, 0.8, 0.4]),
                orientation="horizontal",
            )
            cbar_e.set_label("Energy efficiency (Tok/Joule)", fontsize=12)
            cbar_e.ax.tick_params(labelsize=10)
            cbar_e.ax.set_facecolor("none")

            fig.patch.set_facecolor("none")
            plt.close(fig)
            display(fig)

    def display_ui(self):
        """Display the UI components."""
        ui_components = [self.add_panel_button]

        if self.show_bali:
            self._create_bali_colorbar()

            def on_vmin_vmax_change(change):
                if (
                    change["type"] == "change"
                    and change["name"] == "value"
                ):
                    self.custom_vmin_vmax = (
                        self.vmin_widget.value,
                        self.vmax_widget.value,
                    )
                    self._create_bali_colorbar()
                    for panel in self.plot_panels:
                        panel["update_plot"]()

            self.vmin_widget.observe(on_vmin_vmax_change)
            self.vmax_widget.observe(on_vmin_vmax_change)
            ui_components.append(self.bali_colorbar_container)

        ui_components.append(self.output_container)
        display(widgets.VBox(ui_components))
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
            # Prevent automatic display of the figure outside the Output widget
            plt.close(fig)
        output = widgets.Output()
        selection_output = widgets.Output()

        def update_plot():
            metric = metric_dropdown.value
            level = level_dropdown.value
            df = self.perfdata_by_level.get(level)

            # Always clear the output and redraw the figure to ensure
            # in-place updates
            output.clear_output(wait=True)
            selection_output.clear_output(wait=True)
            with output:
                ax.clear()
                # Provide a sink for BALI selection details
                ax._bali_selection_output = selection_output
                self._create_bali_colorbar()
                if df is not None and not df.empty:
                    self.plot_callback(
                        df,
                        metric,
                        self.cell_range,
                        self.show_idle,
                        ax,
                        level,
                        self.show_bali,
                        self.custom_vmin_vmax,
                    )
                    fig.canvas.draw_idle()
                    if not is_ipympl_backend():
                        display(fig)
                else:
                    print("No data available for the selected level/metric")

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
            [
                widgets.HBox([metric_dropdown, level_dropdown]),
                output,
                selection_output,
            ]
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
