import pickle
from typing import List

import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import widgets, Layout

from jumper_extension.adapters.visualizer.render import RENDERERS
from jumper_extension.adapters.visualizer.visualizer import PerformanceVisualizer
from jumper_extension.utilities import get_available_levels
from jumper_extension.logo import jumper_colors

_LINESTYLE_MPL = {"solid": "-", "dashed": "--", "dotted": ":"}


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
        if config is None:
            return

        render_fn = RENDERERS.get(config.type)
        if render_fn is None:
            return
        result = render_fn(df, config, level, self._hardware, self._io_window)
        if result is None:
            return

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        for item in result.series:
            ax.plot(
                df["time"],
                item.data,
                linestyle=_LINESTYLE_MPL.get(item.linestyle, "-"),
                linewidth=item.width,
                color=item.color,
                alpha=item.opacity,
                label=item.label,
            )
        if result.series:
            ax.legend()

        ax.set_title(result.title + (" (No Idle)" if not show_idle else ""))
        ax.set_xlabel("Time (seconds)")
        ax.grid(True)
        ylim = result.ylim
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
