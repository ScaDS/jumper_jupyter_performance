import logging
import shlex
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Iterator

import pandas as pd

from jumper_extension.adapters.script_writer import NotebookScriptWriter
from jumper_extension.core.parsers import (
    parse_cell_range,
    parse_arguments,
    build_perfmonitor_start_parser,
    build_perfreport_parser,
    build_auto_perfreports_parser,
    build_perfmonitor_plot_parser,
    build_export_perfdata_parser,
    build_export_cell_history_parser,
    build_import_perfdata_parser,
    build_import_cell_history_parser,
    build_export_session_parser,
    build_import_session_parser,
    ArgParsers,
)
from jumper_extension.core.state import Settings
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.monitor.common import MonitorProtocol
from jumper_extension.monitor.backends.thread import PerformanceMonitor
from jumper_extension.adapters.session import SessionExporter, SessionImporter
from jumper_extension.adapters.visualizer.visualizer import build_performance_visualizer, \
    VisualizerProtocol
from jumper_extension.adapters.reporter import PerformanceReporter, build_performance_reporter
from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.utilities import get_available_levels


logger = logging.getLogger("extension")


class PerfmonitorService:
    """High-level performance monitoring service.

    This service wires together monitoring, visualization, reporting,
    cell history, and script recording. It is the main entry point for
    using JUmPER from pure Python code.

    Examples:
        Build a default service::

            from jumper_extension.core.service import (
                build_perfmonitor_service,
            )

            service = build_perfmonitor_service()
    """
    def __init__(
        self,
        settings: Settings,
        monitor: MonitorProtocol,
        visualizer: VisualizerProtocol,
        reporter: PerformanceReporter,
        cell_history: CellHistory,
        script_writer: NotebookScriptWriter,
    ):
        """Initialize a PerfmonitorService instance.

        Args:
            settings: Extension settings to use for this service.
            monitor: Performance monitor that will collect metrics.
            visualizer: Visualizer attached to the monitor.
            reporter: Reporter responsible for performance reports.
            cell_history: Cell history tracker for executed cells.
            script_writer: Script writer used for code recording.
        """
        self.settings = settings
        self.monitor = monitor
        self.visualizer = visualizer
        self.reporter = reporter
        self.cell_history = cell_history
        self.script_writer = script_writer
        self._skip_report = False

    @staticmethod
    def _create_monitor(monitor_type: str = "default") -> MonitorProtocol:
        """Create a monitor instance based on the requested type.

        Args:
            monitor_type: ``"default"`` for the best available backend
                (native_c if compiled and healthy, else subprocess
                Python), ``"thread"`` for the in-process threaded
                monitor, ``"slurm_multinode"`` for the multi-node
                SLURM monitor.

        Returns:
            A monitor satisfying :class:`MonitorProtocol`.
        """
        if monitor_type == "thread":
            return PerformanceMonitor()
        if monitor_type == "native_c":
            from jumper_extension.monitor.backends.native_c import CSubprocessPerformanceMonitor
            return CSubprocessPerformanceMonitor()
        if monitor_type == "slurm_multinode":
            from jumper_extension.monitor.backends.slurm_multinode import SlurmMultinodeMonitor
            return SlurmMultinodeMonitor()
        # default: try native_c first, fall back to subprocess Python
        try:
            from jumper_extension.monitor.backends.native_c.build import ensure_native_c
            if ensure_native_c():
                from jumper_extension.monitor.backends.native_c import CSubprocessPerformanceMonitor
                logger.info("[JUmPER] Using native_c monitor (compiled C collector).")
                return CSubprocessPerformanceMonitor()
        except Exception:
            pass
        from jumper_extension.monitor.backends.subprocess_python import SubprocessPerformanceMonitor
        logger.info("[JUmPER] Using subprocess_python monitor (Python collector).")
        return SubprocessPerformanceMonitor()

    def on_pre_run_cell(
        self,
        raw_cell: str,
        cell_magics: List[str],
        should_skip_report: bool,
    ):
        """Prepare internal state before executing a cell.

        Args:
            raw_cell: Source code of the cell being executed.
            cell_magics: List of magic commands detected in the cell.
            should_skip_report: Whether automatic reporting should be
                skipped for this cell.
        """
        self.cell_history.start_cell(raw_cell, cell_magics)
        self._skip_report = should_skip_report

    def on_post_run_cell(self, result):
        """Handle post-cell execution, including automatic reports.

        If automatic reports are enabled and monitoring is running,
        this will emit either a text or HTML report for the last cell.

        Args:
            result: Execution result object returned by IPython.
        """
        self.cell_history.end_cell(result)
        if (
                not self._skip_report
                and self.monitor.running
                and self.settings.perfreports.enabled
        ):
            if self.settings.perfreports.text:
                self.reporter.print(
                    cell_range=None, level=self.settings.perfreports.level
                )
            else:
                self.reporter.display(
                    cell_range=None, level=self.settings.perfreports.level
                )

    def show_resources(self) -> None:
        """Display available hardware resources.

        Prints information about CPUs, memory, and GPUs available to the
        current or imported session.

        Returns:
            None

        Examples:
            >>> service.show_resources()
        """
        if not self.monitor.running and not self.monitor.is_imported:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        if self.monitor.is_imported:
            logger.info(
                EXTENSION_INFO_MESSAGES[ExtensionInfoCode.IMPORTED_SESSION_RESOURCES].format(
                    source=self.monitor.session_source
                )
            )
        print("[JUmPER]:")
        cpu_info = (
            f"  CPUs: {self.monitor.num_cpus}\n    "
            f"CPU affinity: {self.monitor.cpu_handles}"
        )
        print(cpu_info)
        mem_gpu_info = (
            f"  Memory: {self.monitor.memory_limits['system']} GB\n  "
            f"GPUs: {self.monitor.num_gpus}"
        )
        print(mem_gpu_info)
        if self.monitor.num_gpus:
            print(f"    {self.monitor.gpu_name}, {self.monitor.gpu_memory} GB")

    def show_cell_history(self) -> None:
        """Show an interactive table of executed cells.

        Displays the tracked cell history using an interactive table
        widget, if available.

        Returns:
            None

        Examples:
            >>> service.show_cell_history()
        """
        self.cell_history.show_itable()

    def start_monitoring(
        self,
        interval: Optional[float] = None,
        monitor_type: str = "default",
        check_sanity: bool = False,
        monitor: Optional[MonitorProtocol] = None,
    ) -> Optional[ExtensionErrorCode]:
        """Start performance monitoring.

        This method configures and starts the underlying performance
        monitor. If an offline (imported) session is currently
        attached, it is replaced with a new live monitor instance.

        Args:
            interval: Sampling interval in seconds. If ``None``, the
                value from ``settings.monitoring.default_interval`` is
                used.
            monitor_type: Monitor backend to use. ``"default"`` uses
                the standard single-node :class:`PerformanceMonitor`.
                ``"slurm_multinode"`` uses the multi-node SLURM
                monitor that connects to all allocated nodes via SSH.

        Returns:
            Optional[ExtensionErrorCode]: An error code if monitoring
            was already running, otherwise ``None``.

        Examples:
            Start monitoring with the default interval::

                service.start_monitoring()

            Start monitoring with a custom interval::

                service.start_monitoring(interval=0.5)

            Start multi-node SLURM monitoring::

                service.start_monitoring(monitor_type="slurm_multinode")
        """
        # If an imported (offline) session is currently attached, or the
        # monitor has not been started yet, install the requested monitor.
        # A user-supplied instance takes precedence over the built-in
        # factory so custom backends can be plugged in.
        if not self.monitor.running:
            if monitor is not None:
                self.monitor = monitor
            else:
                self.monitor = self._create_monitor(monitor_type)

        if self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.MONITOR_ALREADY_RUNNING]
            )
            return ExtensionErrorCode.MONITOR_ALREADY_RUNNING

        if interval is None:
            interval = self.settings.monitoring.default_interval
        else:
            self.settings.monitoring.user_interval = interval

        if check_sanity:
            from jumper_extension.monitor.sanity import (
                is_supported_monitor,
                run_sanity_check,
            )
            if is_supported_monitor(self.monitor) and monitor is None:
                # Use a throw-away instance so the real monitor starts
                # with a clean state.
                sanity_monitor = self._create_monitor(monitor_type)
                run_sanity_check(sanity_monitor)
            else:
                msg = (
                    f"[JUmPER] --check-sanity was tailored for the "
                    f"'thread', 'subprocess_python' and 'native_c' monitors. "
                    f"'{type(self.monitor).__name__}' is not supported by "
                    f"the tailored check; skipping sanity check."
                )
                logger.warning(msg)
                print(msg)

        self.monitor.start(interval)
        self.settings.monitoring.running = self.monitor.running
        self.visualizer.attach(self.monitor)
        self.reporter.attach(self.monitor)
        return None

    def stop_monitoring(self) -> None:
        """Stop the active performance monitoring session.

        Returns:
            None

        Examples:
            >>> service.stop_monitoring()
        """
        if not self.monitor:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        self.monitor.stop()
        self.settings.monitoring.running = False

    def plot_performance(
        self,
        metrics: Optional[List[str]] = None,
        cell_range: Optional[Tuple[int, int]] = None,
        level: Optional[str] = None,
        save_jpeg: Optional[str] = None,
        pickle_file: Optional[str] = None,
        backend: Optional[str] = None,
        live: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Open an interactive performance plot.

        Works for both live and imported sessions. Uses the attached
        visualizer to display metrics and interactive widgets. When
        ``level`` is provided (or inferred for exports), the plot is
        rendered directly without ipywidgets, which also enables JPEG
        and pickle exports.

        Args:
            metrics: Optional list of metric subset names to plot
            cell_range: Optional tuple of (start_idx, end_idx) for cell range
            level: Optional performance level for direct plotting
            save_jpeg: Optional path to save plot as JPEG
            pickle_file: Optional path to serialize plot data
            backend: Optional visualizer backend ("matplotlib" or "plotly")
            live: If set, tuple of (update_interval, window_seconds) for live plotting

        Returns:
            None

        Examples:
            >>> service.plot_performance()
            >>> service.plot_performance(
            ...     metrics=["cpu_summary", "memory"],
            ...     level="process",
            ...     cell_range=(0, 3),
            ... )
            >>> service.plot_performance(live=(2.0, 120.0))
        """
        if not self.monitor.running and not self.monitor.is_imported:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return
        
        if live is not None and self.monitor.is_imported:
            logger.warning(
                "Live plotting is not available for imported sessions. "
                "Use regular plotting instead."
            )
            return
        
        if self.monitor.is_imported:
            logger.info(
                EXTENSION_INFO_MESSAGES[ExtensionInfoCode.IMPORTED_SESSION_PLOT].format(
                    source=self.monitor.session_source
                )
            )

        selected_backend = (
            (backend or self.settings.visualizer_backend) or "matplotlib"
        )
        selected_backend = selected_backend.strip().lower()
        if backend:
            self.settings.visualizer_backend = selected_backend

        current_backend = (
            "plotly"
            if self.visualizer.__class__.__name__
            == "PlotlyPerformanceVisualizer"
            else "matplotlib"
        )
        if selected_backend != current_backend:
            self.visualizer = build_performance_visualizer(
                self.cell_history,
                plots_disabled=False,
                plots_disabled_reason="Plotting not available.",
                backend=selected_backend,
            )
            self.visualizer.attach(self.monitor)

        effective_level = level

        if effective_level is None and (
            metrics or save_jpeg or pickle_file
        ):
            # Default to configured level for direct plotting/export paths
            effective_level = self.settings.perfreports.level

        if effective_level is not None:
            available_levels = get_available_levels()
            if effective_level not in available_levels:
                logger.warning(
                    EXTENSION_ERROR_MESSAGES[
                        ExtensionErrorCode.INVALID_LEVEL
                    ].format(level=effective_level, levels=available_levels)
                )
                return

        if live is not None:
            update_interval, window_seconds = live
            self.visualizer.plot_live(
                metric_subsets=metrics,
                cell_range=cell_range,
                level=effective_level,
                update_interval=update_interval,
                window_seconds=window_seconds,
            )
        else:
            self.visualizer.plot(
                metric_subsets=metrics,
                cell_range=cell_range,
                level=effective_level,
                save_jpeg=save_jpeg,
                pickle_file=pickle_file,
            )

    def enable_perfreports(
        self,
        level: str,
        interval: Optional[float] = None,
        text: bool = False
    ) -> None:
        """Enable automatic performance reports after each cell.

        Args:
            level: Monitoring level (``\"process\"``, ``\"user\"``,
                ``\"system\"``, or ``\"slurm\"``).
            interval: Sampling interval in seconds. If provided, this
                value is used when starting monitoring.
            text: If ``True``, use plain-text reports instead of HTML.

        Returns:
            None

        Examples:
            Enable HTML reports at process level::

                service.enable_perfreports(level="process")

            Enable text reports with a custom interval::

                service.enable_perfreports(
                    level="user",
                    interval=0.5,
                    text=True,
                )
        """
        self.settings.perfreports.enabled = True
        self.settings.perfreports.level = level
        self.settings.perfreports.text = text

        format_message = "text" if text else "html"
        options_message = f"level: {level}, interval: {interval}, format: {format_message}"

        error_code = self.start_monitoring(interval)

        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_ENABLED
            ].format(
                options_message=options_message,
            )
        )

    def disable_perfreports(self) -> None:
        """Disable automatic performance reports after cell execution.

        Returns:
            None

        Examples:
            >>> service.disable_perfreports()
        """
        self.settings.perfreports.enabled = False
        logger.info(
            EXTENSION_INFO_MESSAGES[
                ExtensionInfoCode.PERFORMANCE_REPORTS_DISABLED
            ]
        )

    def show_perfreport(
        self,
        cell_range: Optional[Tuple[int, int]] = None,
        level: Optional[str] = None,
        text: bool = False
    ) -> None:
        """Show a performance report for the current session.

        Args:
            cell_range: Optional tuple ``(start_idx, end_idx)`` limiting
                the report to a subset of cells. If ``None``, all cells
                are included.
            level: Optional monitoring level override. If ``None``,
                the default report level is used.
            text: If ``True``, render a text report instead of HTML.

        Returns:
            None

        Examples:
            Show a report for all cells::

                service.show_perfreport()

            Show a report for cells 2 through 5 at system level::

                service.show_perfreport(
                    cell_range=(2, 5),
                    level="system",
                )
        """
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return

        if text:
            self.reporter.print(cell_range=cell_range, level=level)
        else:
            self.reporter.display(cell_range=cell_range, level=level)

    def export_perfdata(
        self,
        file: Optional[str] = None,
        level: Optional[str] = None,
        name: Optional[str] = None
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Export performance data or return it as data frames.

        Args:
            file: Optional target file path. If provided, data is
                written using the monitor's data adapter. If ``None``,
                data is returned as a mapping of variable name to
                ``pandas.DataFrame``.
            level: Optional monitoring level override. If ``None``,
                the default export level is used.

        Returns:
            Optional[Dict[str, pandas.DataFrame]]: If ``file`` is
            ``None``, a mapping from variable name to data frame. If
            ``file`` is set, an empty dictionary.

        Examples:
            Export metrics to a CSV file::

                service.export_perfdata(
                    file="performance.csv",
                    level="process",
                )

            Get a DataFrame in memory::

                frames = service.export_perfdata()
                df = next(iter(frames.values()))
        """
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return {}

        if file:
            self.monitor.data.export(
                file, level=level, cell_history=self.cell_history
            )
            return {}
        else:
            df = self.monitor.data.view(
                level=level, cell_history=self.cell_history
            )
            var_name = name or self.settings.export_vars.perfdata
            logger.info(
                EXTENSION_INFO_MESSAGES[
                    ExtensionInfoCode.PERFORMANCE_DATA_AVAILABLE
                ].format(var_name=var_name)
            )
            return {var_name: df}

    def load_perfdata(self, file: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load performance data from a file.

        Args:
            file: Path to a CSV or JSON file containing performance
                data.

        Returns:
            Optional[Dict[str, pandas.DataFrame]]: Mapping from the
            configured variable name to the loaded data frame.

        Examples:
            >>> frames = service.load_perfdata("performance.csv")
            >>> df = next(iter(frames.values()))
        """
        df = self.monitor.data.load(file)
        var_name = self.settings.loaded_vars.perfdata
        if df is not None:
            logger.info(
                EXTENSION_INFO_MESSAGES[
                    ExtensionInfoCode.PERFORMANCE_DATA_AVAILABLE
                ].format(var_name=var_name)
            )
        return {var_name: df}

    def export_cell_history(
        self,
        file: Optional[str] = None,
        name: Optional[str] = None
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Export cell history or return it as a data frame.

        Args:
            file: Optional target file path. If provided, the cell
                history is written to disk. If ``None``, data is
                returned as a mapping of variable name to
                ``pandas.DataFrame``.

        Returns:
            Optional[Dict[str, pandas.DataFrame]]: If ``file`` is
            ``None``, a mapping from variable name to data frame. If
            ``file`` is set, an empty dictionary.

        Examples:
            Export cell history to CSV::

                service.export_cell_history(file="cells.csv")

            Get the history as a DataFrame::

                frames = service.export_cell_history()
                df = next(iter(frames.values()))
        """
        if file:
            self.cell_history.export(file)
            return {}
        else:
            df = self.cell_history.view()
            var_name = name or self.settings.export_vars.cell_history
            logger.info(
                f"[JUmPER]: Cell history data available as '{var_name}'"
            )
            return {var_name: df}

    def load_cell_history(self, file: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load cell history from a file.

        Args:
            file: Path to a CSV or JSON file containing cell history.

        Returns:
            Optional[Dict[str, pandas.DataFrame]]: Mapping from the
            configured variable name to the loaded data frame.

        Examples:
            >>> frames = service.load_cell_history("cells.csv")
            >>> df = next(iter(frames.values()))
        """
        df = self.cell_history.load(file)
        var_name = self.settings.loaded_vars.cell_history
        if df is not None:
            logger.info(
                f"[JUmPER]: Cell history data available as '{var_name}'"
            )
        return {var_name: df}

    def export_session(self, path: Optional[str] = None) -> None:
        """Export the full monitoring session.

        This uses :class:`SessionExporter` to write performance data
        and cell history to a directory or zip archive.

        Args:
            path: Optional target directory or ``.zip`` file. If the
                path ends with ``.zip``, a temporary directory is used
                and then compressed into that archive. If ``None``, a
                timestamped directory is created.

        Returns:
            None

        Examples:
            Export to a directory::

                service.export_session("session-dir")

            Export to a zip archive::

                service.export_session("session.zip")
        """
        if not self.monitor.running and not self.monitor.is_imported:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
        exporter = SessionExporter(self.monitor, self.cell_history, self.visualizer, self.reporter, logger)
        exporter.export(path)

    def import_session(self, path: str) -> None:
        """Import a monitoring session from disk.

        Uses :class:`SessionImporter` to attach performance data and
        cell history from the given directory or zip archive.

        Args:
            path: Directory or ``.zip`` archive previously created by
                :meth:`export_session`.

        Returns:
            None

        Examples:
            >>> service.import_session("session.zip")
        """
        importer = SessionImporter(logger)
        ok = importer.import_(path, self)
        if ok:
            logger.info(
                EXTENSION_INFO_MESSAGES[ExtensionInfoCode.SESSION_IMPORTED].format(
                    source=self.monitor.session_source
                )
            )

    def fast_setup(self) -> None:
        """Quickly start monitoring with per-cell reports enabled.

        This convenience helper starts monitoring with a one-second
        interval and enables HTML performance reports at the ``process``
        level.

        Returns:
            None

        Examples:
            >>> service.fast_setup()
        """
        self.start_monitoring(1.0)
        self.enable_perfreports(level="process", interval=1.0, text=False)
        logger.info("[JUmPER]: Fast setup complete! Ready for interactive analysis.")

    def start_script_recording(self, output_path: Optional[str] = None) -> None:
        """Start recording code from cells to a Python script.

        Args:
            output_path: Optional path to the output script file. If
                ``None``, a filename is generated automatically.

        Returns:
            None

        Examples:
            Start recording to an auto-generated file::

                service.start_script_recording()

            Record to a specific script path::

                service.start_script_recording("analysis_script.py")
        """
        self.script_writer.start_recording(self.settings.snapshot(), output_path)

        if output_path:
            logger.info(f"[JUmPER]: Started script recording to '{output_path}'")
        else:
            logger.info("[JUmPER]: Started script recording (filename will be auto-generated)")

    def stop_script_recording(self) -> Optional[str]:
        """Stop recording and save accumulated code to a script file.

        Returns:
            Optional[str]: Path to the saved script file, or ``None``
            if recording was not active or no cells were captured.

        Examples:
            >>> path = service.stop_script_recording()
            >>> print(path)
        """
        if not self.script_writer:
            print("No script recording in progress.")
            return None

        output_path = self.script_writer.stop_recording()
        logger.info(f"Script saved to: {output_path}")
        return output_path

    @contextmanager
    def monitored(self) -> "Iterator[PerfmonitorService]":
        """Context manager for monitoring a code block.

        This helper simulates a virtual cell: it registers a synthetic
        cell before the block and finalizes it afterwards so that the
        enclosed code is tracked like any other cell.

        Yields:
            PerfmonitorService: The current service instance, for
            optional use inside the context.

        Examples:
            Use the service as a monitoring context::

                with service.monitored():
                    do_expensive_work()
        """
        unavailable_message = "unavailable on monitored context"
        self.on_pre_run_cell(
            raw_cell=f"# <Code {unavailable_message}>",
            cell_magics=[f"<Magics {unavailable_message}>"],
            should_skip_report=False
        )
        try:
            yield self
        finally:
            self.on_post_run_cell(None)

    def close(self) -> None:
        """Stop monitoring and release resources held by the service.

        Returns:
            None

        Examples:
            >>> service.close()
        """
        if self.monitor:
            self.monitor.stop()


class PerfmonitorMagicAdapter:
    """
    String-based adapter for IPython magic commands.
    Parses string arguments and delegates to PerfmonitorService.
    All methods must have the same names as magic commands to be recognized by script writer.
    """
    def __init__(
        self,
        service: PerfmonitorService,
        parsers: ArgParsers
    ):
        self.service = service
        self.parsers = parsers

    def on_pre_run_cell(self, raw_cell: str, cell_magics: List[str], should_skip_report: bool):
        """Delegate to magic_adapter."""
        self.service.on_pre_run_cell(raw_cell, cell_magics, should_skip_report)

    def on_post_run_cell(self, result):
        """Delegate to magic_adapter."""
        self.service.on_post_run_cell(result)

    def perfmonitor_resources(self, line: str):
        """Display available hardware resources (CPUs, memory, GPUs)."""
        self.service.show_resources()

    def show_cell_history(self, line: str):
        """Show interactive table of all executed cells with timestamps and durations."""
        self.service.show_cell_history()

    def perfmonitor_start(self, line: str):
        """Start performance monitoring with specified interval (default: 1 second)."""
        if line:
            try:
                tokens = shlex.split(line)
            except ValueError:
                logger.warning(
                    EXTENSION_ERROR_MESSAGES[
                        ExtensionErrorCode.INVALID_INTERVAL_VALUE
                    ].format(interval=line)
                )
                return

            if tokens and tokens[0] != "--monitor":
                try:
                    float(tokens[0])
                except ValueError:
                    logger.warning(
                        EXTENSION_ERROR_MESSAGES[
                            ExtensionErrorCode.INVALID_INTERVAL_VALUE
                        ].format(interval=tokens[0])
                    )
                    return

        args = parse_arguments(self.parsers.perfmonitor_start, line)
        if args is None:
            return
        self.service.start_monitoring(
            interval=args.interval,
            monitor_type=args.monitor,
            check_sanity=args.check_sanity,
        )

    def perfmonitor_stop(self, line: str):
        """Stop the active performance monitoring session."""
        self.service.stop_monitoring()

    @staticmethod
    def _parse_live_args(live_list):
        """Parse --live argument list into (interval, window) tuple.

        ``--live``           → (2.0, 120.0)
        ``--live 1.0``       → (1.0, 120.0)
        ``--live 2.0 60``    → (2.0, 60.0)
        """
        defaults = (2.0, 120.0)
        if not live_list:
            return defaults
        interval = live_list[0] if len(live_list) >= 1 else defaults[0]
        window = live_list[1] if len(live_list) >= 2 else defaults[1]
        return (interval, window)

    def perfmonitor_plot(self, line: str):
        """Open interactive plot or direct plot/export of performance data."""
        args = parse_arguments(self.parsers.perfmonitor_plot, line)
        if args is None:
            return

        cell_range = None
        if args.cell:
            cell_range = self._parse_cell_range(args.cell)
            if cell_range is None:
                return

        metrics = None
        if args.metrics:
            metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]

        self.service.plot_performance(
            metrics=metrics,
            cell_range=cell_range,
            level=args.level,
            save_jpeg=args.save_jpeg,
            pickle_file=args.pickle_file,
            backend=args.backend,
            live=self._parse_live_args(args.live) if hasattr(args, 'live') and args.live is not None else None,
        )

    def perfmonitor_enable_perfreports(self, line: str):
        """Enable automatic performance reports after each cell execution."""
        args = parse_arguments(self.parsers.auto_perfreports, line)
        if args is None:
            return
        self.service.enable_perfreports(
            level=args.level,
            interval=float(args.interval) if args.interval else None,
            text=args.text
        )

    def perfmonitor_disable_perfreports(self, line: str):
        """Disable automatic performance reports after cell execution."""
        self.service.disable_perfreports()

    def perfmonitor_perfreport(self, line: str):
        """Show performance report with optional cell range and level filters."""
        args = parse_arguments(self.parsers.perfreport, line)
        if not args:
            return

        cell_range = None
        if args.cell:
            cell_range = self._parse_cell_range(args.cell)
            if cell_range is None:
                return

        self.service.show_perfreport(
            cell_range=cell_range,
            level=args.level,
            text=args.text
        )

    def perfmonitor_export_perfdata(self, line: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Export performance data or push as DataFrame."""
        args = parse_arguments(self.parsers.export_perfdata, line)
        return self.service.export_perfdata(
            file=args.file if args else None,
            level=args.level if args else None,
            name=args.name if args else None
        )

    def perfmonitor_load_perfdata(self, line: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Import performance data from file."""
        args = parse_arguments(self.parsers.import_perfdata, line)
        if not args:
            return {}
        return self.service.load_perfdata(args.file)

    def perfmonitor_export_cell_history(self, line: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Export cell history or push as DataFrame."""
        args = parse_arguments(self.parsers.export_cell_history, line)
        return self.service.export_cell_history(
            file=args.file if args else None,
            name=args.name if args else None
        )

    def perfmonitor_load_cell_history(self, line: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Import cell history from file."""
        args = parse_arguments(self.parsers.import_cell_history, line)
        if not args:
            return {}
        return self.service.load_cell_history(args.file)

    def perfmonitor_fast_setup(self, line: str):
        """Quick setup: start perfmonitor and enable perfreports."""
        self.service.fast_setup()

    def perfmonitor_help(self, line: str):
        """Show comprehensive help information for all available commands."""
        commands = [
            "perfmonitor_fast_setup -- quick setup: enable ipympl plots, start monitor, enable reports",
            "perfmonitor_help -- show this comprehensive help",
            "perfmonitor_resources -- show available hardware resources",
            "show_cell_history -- show interactive table of cell execution history",
            "perfmonitor_start [interval] [--monitor TYPE] -- start monitoring (default: 1s, monitor=default)",
            "perfmonitor_stop -- stop monitoring",
            "perfmonitor_perfreport [--cell RANGE] [--level LEVEL] -- show report",
            "perfmonitor_plot -- interactive plot with widgets for data exploration",
            "perfmonitor_enable_perfreports [--level LEVEL] [--interval INTERVAL] [--text] -- enable auto-reports",
            "perfmonitor_disable_perfreports -- disable auto-reports",
            "perfmonitor_export_perfdata [--file FILE] [--level LEVEL] [--name NAME] -- export CSV;"
            " without --file pushes DataFrame (default 'perfdata_df')",
            "perfmonitor_export_cell_history [--file FILE] [--name NAME] -- export history to JSON/CSV;"
            " without --file pushes DataFrame (default 'cell_history_df')",
            "export_session [target|target.zip] -- export full session",
            "import_session <dir-or-zip> -- import full session for offline analysis",
            "start_write_script [output_path] -- record subsequent cells to a Python script",
            "end_write_script -- stop recording and save the script",
        ]
        print("Available commands:")
        for cmd in commands:
            print(f"  {cmd}")

        print("\nMonitoring Levels:")
        print("  process -- current Python process only (default, most focused)")
        print("  user    -- all processes belonging to current user")
        print("  system  -- system-wide metrics across all processes")
        available_levels = get_available_levels()
        if "slurm" in available_levels:
            print("  slurm   -- processes within current SLURM job (HPC environments)")

        print("\nMonitor Types:")
        print("  default           -- standard single-node monitoring (default)")
        print("  slurm_multinode  -- multi-node SLURM monitoring via SSH")
        print("                      (writes results to jumper_multinode.jsonl)")

        print("\nCell Range Formats:")
        print("  5       -- single cell (cell #5)")
        print("  2:8     -- range of cells (cells #2 through #8)")
        print("  :5      -- from start to cell #5")
        print("  3:      -- from cell #3 to end")

        print("\nMetric Categories:")
        print("  cpu, gpu, mem, io (default: all available)")
        print("  cpu_all, gpu_all for detailed per-core/per-GPU metrics")

    def export_session(self, line: str):
        """Export full session into a directory or zip.

        Usage:
          %export_session
          %export_session my_dir
          %export_session my.zip
        """
        args = parse_arguments(self.parsers.export_session, line)
        if args is None:
            return
        self.service.export_session(path=args.path)

    def import_session(self, line: str):
        """Import full session from a directory or zip.

        Usage:
          %import_session path/to/dir-or-zip
        """
        args = parse_arguments(self.parsers.import_session, line)
        if not args:
            return
        self.service.import_session(args.path)

    def start_write_script(self, line: str):
        """
        Start recording code from cells to a Python script.

        Usage:
          %start_write_script [output_path]

        Examples:
          %start_write_script
          %start_write_script my_script.py
        """
        output_path = line.strip() if line else None
        self.service.start_script_recording(output_path)

    def end_write_script(self, line: str):
        """Stop recording and save accumulated code to file."""
        self.service.stop_script_recording()

    def _parse_cell_range(self, cell_str: str) -> Optional[Tuple[int, int]]:
        """Parse a cell range string into start and end indices."""
        result = parse_cell_range(cell_str, len(self.service.cell_history))
        if result is None and cell_str:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_CELL_RANGE
                ].format(cell_range=cell_str)
            )
        return result

    @contextmanager
    def monitored(self):
        """Code performance monitoring context manager."""
        with self.service.monitored():
            yield self

    def close(self):
        """Close the magic_adapter and release any resources."""
        self.service.close()


def build_perfmonitor_service(
        plots_disabled: bool = False,
        plots_disabled_reason: str = "Plotting not available.",
        display_disabled: bool = False,
        display_disabled_reason: str = "Display not available.",
        visualizer_backend: str = "matplotlib",
) -> PerfmonitorService:
    """Build a new :class:`PerfmonitorService` instance.

    This factory configures the default monitor, visualizer, reporter,
    cell history, and script writer for use in Python code.

    Args:
        plots_disabled: If ``True``, disable plotting in the visualizer.
        plots_disabled_reason: Human-readable reason shown when plots
            are disabled.
        display_disabled: If ``True``, disable rich display for reports.
        display_disabled_reason: Human-readable reason shown when rich
            display is disabled.
        visualizer_backend: Visualizer backend to use. Supported values:
            ``"matplotlib"`` (default) and ``"plotly"``.

    Returns:
        PerfmonitorService: A fully initialized service instance.

    Examples:
        >>> from jumper_extension.core.service import build_perfmonitor_service
        >>> service = build_perfmonitor_service()
    """
    settings = Settings()
    settings.visualizer_backend = (
        visualizer_backend.strip().lower()
        if visualizer_backend
        else "matplotlib"
    )
    monitor = PerformanceMonitor()
    cell_history = CellHistory()
    visualizer = build_performance_visualizer(
        cell_history,
        plots_disabled=plots_disabled,
        plots_disabled_reason=plots_disabled_reason,
        backend=visualizer_backend,
    )
    reporter = build_performance_reporter(
        cell_history,
        display_disabled=display_disabled,
        display_disabled_reason=display_disabled_reason,
    )
    script_writer = NotebookScriptWriter(cell_history)

    return PerfmonitorService(
        settings=settings,
        monitor=monitor,
        visualizer=visualizer,
        reporter=reporter,
        cell_history=cell_history,
        script_writer=script_writer,
    )


def build_perfmonitor_magic_adapter(
        plots_disabled: bool = False,
        plots_disabled_reason: str = "Plotting not available.",
        display_disabled: bool = False,
        display_disabled_reason: str = "Display not available.",
        visualizer_backend: str = "matplotlib",
) -> PerfmonitorMagicAdapter:
    """Build a new :class:`PerfmonitorMagicAdapter` instance.

    This factory constructs a :class:`PerfmonitorService` and wraps it
    with a string-based adapter suitable for IPython magics or other
    command-style interfaces.

    Args:
        plots_disabled: If ``True``, disable plotting in the visualizer.
        plots_disabled_reason: Human-readable reason shown when plots
            are disabled.
        display_disabled: If ``True``, disable rich display for reports.
        display_disabled_reason: Human-readable reason shown when rich
            display is disabled.
        visualizer_backend: Visualizer backend to use. Supported values:
            ``"matplotlib"`` (default) and ``"plotly"``.

    Returns:
        PerfmonitorMagicAdapter: Adapter instance wrapping the service.

    Examples:
        >>> from jumper_extension.core.service import (
        ...     build_perfmonitor_magic_adapter,
        ... )
        >>> adapter = build_perfmonitor_magic_adapter()
    """
    service = build_perfmonitor_service(
        plots_disabled=plots_disabled,
        plots_disabled_reason=plots_disabled_reason,
        display_disabled=display_disabled,
        display_disabled_reason=display_disabled_reason,
        visualizer_backend=visualizer_backend,
    )

    parsers = ArgParsers(
        perfmonitor_start=build_perfmonitor_start_parser(),
        perfreport=build_perfreport_parser(),
        auto_perfreports=build_auto_perfreports_parser(),
        perfmonitor_plot=build_perfmonitor_plot_parser(),
        export_perfdata=build_export_perfdata_parser(),
        export_cell_history=build_export_cell_history_parser(),
        import_perfdata=build_import_perfdata_parser(),
        import_cell_history=build_import_cell_history_parser(),
        export_session=build_export_session_parser(),
        import_session=build_import_session_parser(),
    )

    return PerfmonitorMagicAdapter(
        service=service,
        parsers=parsers,
    )
