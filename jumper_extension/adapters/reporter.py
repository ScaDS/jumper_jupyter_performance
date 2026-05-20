import logging

from jumper_extension.adapters.data import aggregate_node_info
from jumper_extension.monitor.common import MonitorProtocol, UnavailablePerformanceMonitor
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES, EXTENSION_INFO_MESSAGES, ExtensionInfoCode,
)
from typing import List, Tuple, Union, Protocol, runtime_checkable

from pathlib import Path
from IPython.display import display, HTML
from jinja2 import Environment, FileSystemLoader, select_autoescape

from jumper_extension.utilities import filter_perfdata
from .analyzer import PerformanceAnalyzer, PerformanceTag, TagScore
from .cell_history import CellHistory

logger = logging.getLogger("extension")


class ReportBuilder:
    """Base class for report builders"""
    def __init__(
        self,
        monitor: MonitorProtocol,
        cell_history: CellHistory,
        analyzer: PerformanceAnalyzer,
    ):
        self.monitor = monitor
        self.cell_history = cell_history
        self.min_duration = None
        self.analyzer = analyzer

    def _prepare_report_data(self, cell_range, level):
        """Prepare all necessary data for performance reporting.

        Returns:
            dict: Dictionary containing filtered_cells, perfdata, ranked_tags,
                  total_duration, and other data needed for display methods.
                  Returns None if data preparation fails.
        """

        cell_range = self._resolve_cell_range(cell_range)

        if cell_range is None:
            return

        # Filter cell history data first using cell_range
        start_idx, end_idx = cell_range
        filtered_cells = self.cell_history.view(start_idx, end_idx + 1)

        perfdata = self.monitor.nodes.view(level=level)
        perfdata = filter_perfdata(
            filtered_cells, perfdata, compress_idle=False
        )

        # Check if non-empty, otherwise print results
        if perfdata.empty:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.NO_PERFORMANCE_DATA
                ]
            )
            return

        # Analyze cell performance
        hardware = aggregate_node_info(self.monitor.nodes.hardware)
        memory_limit = hardware.memory_limits.get(level, 0.0)
        gpu_memory_limit = hardware.gpu_memory if hardware.num_gpus > 0 else None

        tags_model = self.analyzer.analyze_cell_performance(
            perfdata,
            memory_limit,
            gpu_memory_limit
        )

        # Calculate the total duration of selected cells
        total_duration = filtered_cells["duration"].sum()

        return {
            'cell_range': cell_range,
            'filtered_cells': filtered_cells,
            'perfdata': perfdata,
            'tags_model': tags_model,
            'total_duration': total_duration,
        }

    def _resolve_cell_range(self, cell_range) -> Union[Tuple[int, int], None]:
        """
        Resolve cell range for performance reporting.

        Behavior:
        - If cell_range is None, selects the last cell whose duration is not "short"
         and returns it as a singleton range (idx, idx).
        - Returns None if:
          - no active monitor is attached,
          - the history has no cells,
          - there is no cell with a non-short duration.
        """

        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return None

        if cell_range is None:
            valid_cells = self.cell_history.view()

            if len(valid_cells) > 0:
                # Filter for non-short cells
                min_duration = (
                    self.min_duration if self.min_duration is not None else 0
                )
                non_short_cells = valid_cells[
                    valid_cells["duration"] >= min_duration
                    ]

                if len(non_short_cells) > 0:
                    # Get the last non-short cell index
                    last_valid_cell_idx = int(
                        non_short_cells.iloc[-1]["cell_index"]
                    )
                    cell_range = (last_valid_cell_idx, last_valid_cell_idx)
                    return cell_range
                else:
                    logger.warning(
                        EXTENSION_ERROR_MESSAGES[
                            ExtensionErrorCode.NO_PERFORMANCE_DATA
                        ]
                    )
                    return None
            else:
                return None

        return cell_range

    @staticmethod
    def _format_performance_tags(ranked_tags: List[TagScore]):
        """Format ranked performance tags for display"""
        if not ranked_tags:
            return [{"name": "UNKNOWN", "slug": "unknown"}]

        # If the only classification is NORMAL, do not display any tag
        if len(ranked_tags) == 1 and ranked_tags[0].tag == PerformanceTag.NORMAL:
            return []

        # Format all tags with their scores/ratios
        tag_displays = []
        for tag_score in ranked_tags:
            # Create slug for CSS hooks and uppercase name for display
            tag_slug = str(tag_score.tag)
            tag_name = tag_slug.upper()
            tag_displays.append({
                "name": tag_name,
                "slug": tag_slug,
            })
        return tag_displays


class ReportPrinter(ReportBuilder):
    def __init__(
        self,
        monitor: MonitorProtocol,
        cell_history: CellHistory,
        analyzer: PerformanceAnalyzer,
    ):
        super().__init__(monitor, cell_history, analyzer)

    def print(self, cell_range=None, level="process"):
        """Print performance report"""
        data = self._prepare_report_data(cell_range, level)
        if data is None:
            return

        filtered_cells = data['filtered_cells']
        perfdata = data['perfdata']
        tags_model = data['tags_model']
        total_duration = data['total_duration']

        print("-" * 40)
        print("JUmPER Performance Report")
        print("-" * 40)
        n_cells = len(filtered_cells)
        print(
            f"Duration: {total_duration:.2f}s "
            f"({n_cells} cell{'s' if n_cells != 1 else ''})"
        )
        print("-" * 40)

        # Output performance tags
        tags_display = self._format_performance_tags(tags_model)
        if tags_display:
            print("Signature(s):")
            tags_line = " | ".join(tag["name"] for tag in tags_display)
            print(tags_line)

            print("-" * 40)

        # Report table
        hardware = aggregate_node_info(self.monitor.nodes.hardware)
        metrics = [
            (
                f"CPU Util (Across {hardware.num_cpus} CPUs)",
                "cpu_util_avg",
                "-",
            ),
            (
                "Memory (GB)",
                "memory",
                f"{hardware.memory_limits.get(level, 0.0):.2f}",
            ),
            (
                f"GPU Util (Across {hardware.num_gpus} GPUs)",
                "gpu_util_avg",
                "-",
            ),
            (
                "GPU Memory (GB)",
                "gpu_mem_avg",
                f"{hardware.gpu_memory:.2f}",
            ),
        ]

        print(f"{'Metric':<25} {'AVG':<8} {'MIN':<8} {'MAX':<8} {'TOTAL':<8}")
        print("-" * 65)
        for name, col, total in metrics:
            if col in perfdata.columns:
                print(
                    f"{name:<25} {perfdata[col].mean():<8.2f} "
                    f"{perfdata[col].min():<8.2f} {perfdata[col].max():<8.2f} "
                    f"{total:<8}"
                )


@runtime_checkable
class ReportDisplayerProtocol(Protocol):
    """Structural protocol for HTML/text report displayers."""
    def display(self, cell_range=None, level: str = "process") -> None: ...


class ReportDisplayer(ReportBuilder):
    def __init__(
        self,
        monitor: MonitorProtocol,
        cell_history: CellHistory,
        analyzer: PerformanceAnalyzer,
        templates_dir=None
    ):
        super().__init__(monitor, cell_history, analyzer)
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent.parent / "templates"

    def display(self, cell_range=None, level="process"):
        """Print performance report"""

        data = self._prepare_report_data(cell_range, level)
        if data is None:
            return

        filtered_cells = data['filtered_cells']
        perfdata = data['perfdata']
        tags_model = data['tags_model']
        total_duration = data['total_duration']

        tags_display = self._format_performance_tags(tags_model)

        # Build report
        hardware = aggregate_node_info(self.monitor.nodes.hardware)
        metrics_spec = [
            (f"CPU Util (Across {hardware.num_cpus} CPUs)", "cpu_util_avg", "-"),
            ("Memory (GB)", "memory", f"{hardware.memory_limits.get(level, 0.0):.2f}"),
            (f"GPU Util (Across {hardware.num_gpus} GPUs)", "gpu_util_avg", "-"),
            ("GPU Memory (GB)", "gpu_mem_avg", f"{hardware.gpu_memory:.2f}"),
        ]
        metrics_rows = []
        for name, col, total in metrics_spec:
            if col in perfdata.columns:
                metrics_rows.append({
                    "name": name,
                    "avg": float(perfdata[col].mean()),
                    "min": float(perfdata[col].min()),
                    "max": float(perfdata[col].max()),
                    "total": total,
                })

        # Render Jinja2 HTML from external files
        env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"])
        )
        report_html_path = Path("report") / "report.html"
        template = env.get_template(report_html_path.as_posix())
        # Read external stylesheet and inline it for notebook rendering
        try:
            styles_path = self.templates_dir / "report" / "styles.css"
            inline_styles = styles_path.read_text(encoding="utf-8") if styles_path.exists() else ""
        except Exception:
            inline_styles = ""

        html = template.render(
            duration=total_duration,
            n_cells=len(filtered_cells) if filtered_cells is not None else 1,
            metrics=metrics_rows,
            tags=tags_display,
            inline_styles=inline_styles,
        )
        display(HTML(html))


class UnavailableReportDisplayer:
    def __init__(self, reason="Display not available."):
        self._reason = reason

    def display(self, cell_range=None, level="process"):
        """non-opt display"""
        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.HTML_REPORTS_NOT_AVAILABLE
            ].format(reason=self._reason))


class PerformanceReporter:
    """Adapter class for performance reporting"""
    def __init__(
        self,
        printer: ReportPrinter,
        displayer: ReportDisplayerProtocol
    ):
        self.printer = printer
        self.displayer = displayer

    def attach(
        self,
        monitor: MonitorProtocol,
    ):
        """Attach started PerformanceMonitor"""
        # Attach to printer
        self.printer.monitor = monitor
        self.printer.min_duration = monitor.interval
        # Attach to displayer
        self.displayer.monitor = monitor
        self.displayer.min_duration = monitor.interval

    def print(self, cell_range=None, level="process"):
        """Print performance report"""
        self.printer.print(cell_range, level)

    def display(self, cell_range=None, level="process"):
        """Display performance report"""
        self.displayer.display(cell_range, level)


def build_performance_reporter(
    cell_history: CellHistory,
    templates_dir=None,
    display_disabled: bool = False,
    display_disabled_reason="Display not available.",
    thresholds=None,
):
    """
    Build PerformanceReporter object.
    Allows building a reporter without displaying.
    """
    monitor = UnavailablePerformanceMonitor(
        reason="Monitor has not been started yet."
    )
    analyzer = PerformanceAnalyzer(thresholds=thresholds)
    printer = ReportPrinter(monitor, cell_history, analyzer)
    if display_disabled:
        displayer = UnavailableReportDisplayer(
            reason=display_disabled_reason
        )
    else:
        displayer = ReportDisplayer(
            monitor,
            cell_history,
            analyzer,
            templates_dir
        )
    return PerformanceReporter(printer, displayer)


