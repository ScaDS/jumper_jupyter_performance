import logging
from typing import Any

from IPython.core.magic import Magics, line_magic, magics_class

from jumper_extension.ipython.utilities import (
    is_pure_line_magic_cell,
    get_called_line_magics,
)
from jumper_extension.core.service import PerfmonitorMagicAdapter


logger = logging.getLogger("extension")


@magics_class
class PerfmonitorMagics(Magics):
    """IPython line magics for the JUmPER extension.

    This class defines the ``%perfmonitor_*`` family of magics and a
    few helpers. Each magic forwards its work to a
    :class:`PerfmonitorMagicAdapter` instance, which in turn delegates
    to :class:`PerfmonitorService`.

    Args:
        shell: The current IPython shell instance.
        magic_adapter: Adapter that implements the string-based public
            API used by these magics.

    Examples:
        Load the extension in a notebook::

            %load_ext jumper_extension
    """

    def __init__(
        self,
        shell: Any,
        magic_adapter: PerfmonitorMagicAdapter,
    ) -> None:
        """Initialize the magics wrapper.

        Args:
            shell: IPython shell the magics are registered on.
            magic_adapter: Adapter used to execute the underlying
                commands.
        """
        super().__init__(shell)
        self.magic_adapter = magic_adapter

    def pre_run_cell(self, info: Any) -> None:
        """Hook executed before each cell.

        This inspects the raw cell source, extracts any magic commands,
        and informs the underlying adapter so that monitoring and
        reporting state can be updated.

        Args:
            info: IPython pre-run information object that contains
                ``raw_cell``.
        Returns:
            None
        """
        raw_cell = info.raw_cell
        called_line_magics = get_called_line_magics(raw_cell)
        should_skip_report = is_pure_line_magic_cell(raw_cell)
        self.magic_adapter.on_pre_run_cell(
            raw_cell,
            called_line_magics,
            should_skip_report,
        )

    def post_run_cell(self, result: Any) -> None:
        """Hook executed after each cell has run.

        Delegates to the magic adapter so that post-cell reporting and
        bookkeeping can be performed.

        Args:
            result: IPython execution result object.
        Returns:
            None
        """
        self.magic_adapter.on_post_run_cell(result.result)

    @line_magic
    def perfmonitor_resources(self, line: str) -> None:
        """Show hardware resources available to the current session.

        This magic prints CPUs, memory, and GPU information for either
        a live or imported monitoring session.

        Args:
            line: Unused argument string.
        Returns:
            None

        Examples:
            Show resources for the current session::

                %perfmonitor_resources
        """
        self.magic_adapter.perfmonitor_resources(line)

    @line_magic
    def perfmonitor_start(self, line: str) -> None:
        """Start performance monitoring.

        If an interval is provided as a single numeric argument, it is
        interpreted as the sampling interval in seconds; otherwise the
        default interval is used.

        Args:
            line: Optional interval argument, for example ``"1.0"``.
        Returns:
            None

        Examples:
            Start monitoring with the default interval::

                %perfmonitor_start

            Start monitoring with a 0.5 second interval::

                %perfmonitor_start 0.5
        """
        self.magic_adapter.perfmonitor_start(line)

    @line_magic
    def perfmonitor_stop(self, line: str) -> None:
        """Stop the active performance monitoring session.

        Args:
            line: Unused argument string.
        Returns:
            None

        Examples:
            %perfmonitor_stop
        """
        self.magic_adapter.perfmonitor_stop(line)

    @line_magic
    def perfmonitor_plot(self, line: str) -> None:
        """Open an interactive performance plot.

        This magic opens interactive widgets for exploring collected
        performance data. Supports live-updating plots that continuously
        refresh with new data while allowing other cells to execute.

        Args:
            line: Raw argument string forwarded to the adapter.
                Supports --live flag for continuous updates.
        Returns:
            None

        Examples:
            Open an interactive plot for the current session::

                %perfmonitor_plot

            Open a live-updating plot (non-blocking, 2s updates)::

                %perfmonitor_plot --live

            Open a live plot with custom update interval (1 second)::

                %perfmonitor_plot --live 1.0

            Open a live plot with specific metrics and fast updates::

                %perfmonitor_plot --live 0.5 --metrics cpu,mem
        """
        self.magic_adapter.perfmonitor_plot(line)

    @line_magic
    def perfmonitor_enable_perfreports(self, line: str) -> None:
        """Enable automatic performance reports after each cell.

        The line string is parsed for options such as monitoring level,
        interval, and whether to use text or HTML output.

        Args:
            line: Raw argument string, for example
                ``"--level process --interval 1.0"``.
        Returns:
            None

        Examples:
            Enable HTML reports at process level::

                %perfmonitor_enable_perfreports --level process

            Enable text reports for user level with custom interval::

                %perfmonitor_enable_perfreports --level user --interval 0.5 --text
        """
        self.magic_adapter.perfmonitor_enable_perfreports(line)


    @line_magic
    def perfmonitor_disable_perfreports(self, line: str) -> None:
        """Disable automatic performance reports.

        Args:
            line: Unused argument string.
        Returns:
            None

        Examples:
            %perfmonitor_disable_perfreports
        """
        self.magic_adapter.perfmonitor_disable_perfreports(line)

    @line_magic
    def perfmonitor_perfreport(self, line: str) -> None:
        """Show a performance report for the current session.

        The line string may include cell range and monitoring level
        options to restrict the report.

        Args:
            line: Raw argument string, for example
                ``"--cell 2:5 --level system"``.
        Returns:
            None

        Examples:
            Show a report for all cells::

                %perfmonitor_perfreport

            Show a report for cells 2–5 at system level::

                %perfmonitor_perfreport --cell 2:5 --level system
        """
        self.magic_adapter.perfmonitor_perfreport(line)

    @line_magic
    def perfmonitor_export_perfdata(self, line: str) -> None:
        """Export performance data or push it into the notebook.

        If ``--file`` is provided, data is written to disk. Otherwise,
        the resulting data frames are pushed into the user namespace.

        Args:
            line: Raw argument string, such as
                ``"--file perf.csv --level process"``.
        Returns:
            None

        Examples:
            Export process-level data to CSV::

                %perfmonitor_export_perfdata --file perf.csv --level process

            Push a DataFrame into the notebook::

                %perfmonitor_export_perfdata --level user
        """
        perfdata = self.magic_adapter.perfmonitor_export_perfdata(line)
        self.shell.push(perfdata)

    @line_magic
    def perfmonitor_export_cell_history(self, line: str) -> None:
        """Export cell history or push it into the notebook.

        If ``--file`` is provided, the cell history is written to disk.
        Otherwise, a data frame is pushed into the user namespace.

        Args:
            line: Raw argument string, for example
                ``"--file cells.csv"``.
        Returns:
            None

        Examples:
            Export cell history to CSV::

                %perfmonitor_export_cell_history --file cells.csv

            Push the cell history DataFrame::

                %perfmonitor_export_cell_history
        """
        cell_history_data = self.magic_adapter.perfmonitor_export_cell_history(line)
        self.shell.push(cell_history_data)

    @line_magic
    def perfmonitor_load_perfdata(self, line: str) -> None:
        """Load performance data from disk and push it to the notebook.

        Args:
            line: Raw argument string containing ``--file``.
        Returns:
            None

        Examples:
            %perfmonitor_load_perfdata --file perf.csv
        """
        perfdata = self.magic_adapter.perfmonitor_load_perfdata(line)
        self.shell.push(perfdata)

    @line_magic
    def perfmonitor_load_cell_history(self, line: str) -> None:
        """Load cell history from disk and push it to the notebook.

        Args:
            line: Raw argument string containing ``--file``.
        Returns:
            None

        Examples:
            %perfmonitor_load_cell_history --file cells.csv
        """
        cell_history_data = self.magic_adapter.perfmonitor_load_cell_history(line)
        self.shell.push(cell_history_data)

    @line_magic
    def export_session(self, line: str) -> None:
        """Export the full monitoring session to a directory or zip.

        When the target ends with ``.zip``, a temporary directory is
        created and compressed into that archive.

        Args:
            line: Raw argument string containing an optional target
                path.
        Returns:
            None

        Examples:
            Export into a directory::

                %export_session my_dir

            Export into a zip archive::

                %export_session my_session.zip
        """
        self.magic_adapter.export_session(line)

    @line_magic
    def import_session(self, line: str) -> None:
        """Import a monitoring session from a directory or zip.

        Args:
            line: Raw argument string with the source path.
        Returns:
            None

        Examples:
            %import_session my_session.zip
        """
        self.magic_adapter.import_session(line)

    @line_magic
    def perfmonitor_fast_setup(self, line: str) -> None:
        """Run a quick setup for interactive monitoring.

        This helper enables ``ipympl`` interactive plots (if available),
        starts monitoring, and turns on automatic performance reports.

        Args:
            line: Unused argument string.
        Returns:
            None

        Examples:
            Quickly prepare interactive monitoring in a notebook::

                %perfmonitor_fast_setup
        """
        # Enable ipympl interactive plots
        try:
            self.shell.run_line_magic('matplotlib', 'ipympl')
            print("[JUmPER]: Enabled ipympl interactive plots")
        except Exception as e:
            logger.warning(f"Failed to enable ipympl interactive plots: {e}")
        self.magic_adapter.perfmonitor_fast_setup(line)

    @line_magic
    def show_cell_history(self, line: str) -> None:
        """Show an interactive table of executed cells.

        Args:
            line: Unused argument string.
        Returns:
            None

        Examples:
            %show_cell_history
        """
        self.magic_adapter.show_cell_history(line)

    @line_magic
    def perfmonitor_help(self, line: str) -> None:
        """Show comprehensive help for all available magics.

        Args:
            line: Unused argument string.
        Returns:
            None

        Examples:
            %perfmonitor_help
        """
        self.magic_adapter.perfmonitor_help(line)

    @line_magic
    def start_write_script(self, line: str) -> None:
        """Start recording code from subsequent cells to a Python script.

        If no path is provided, the script writer chooses a default
        filename based on the current time.

        Args:
            line: Optional output path, for example
                ``\"my_script.py\"``.

        Examples:
            Start recording to a generated filename::

                %start_write_script

            Record to a specific file::

                %start_write_script my_script.py
        """
        self.magic_adapter.start_write_script(line)

    @line_magic
    def end_write_script(self, line: str) -> None:
        """Stop recording and save accumulated code to a script file.

        Args:
            line: Unused argument string.

        Examples:
            %end_write_script
        """
        self.magic_adapter.end_write_script(line)
