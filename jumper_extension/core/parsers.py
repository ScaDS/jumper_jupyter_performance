"""Module containing parser utilities for the JUmPER extension."""
import argparse
import shlex
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.config.loader import load_config
from jumper_extension.utilities import get_available_levels


@dataclass
class ArgParsers:
    """Configuration for command-line argument parsers."""
    perfmonitor_start: argparse.ArgumentParser
    perfreport: argparse.ArgumentParser
    ai_review: argparse.ArgumentParser
    auto_perfreports: argparse.ArgumentParser
    perfmonitor_plot: argparse.ArgumentParser
    export_perfdata: argparse.ArgumentParser
    export_cell_history: argparse.ArgumentParser
    import_perfdata: argparse.ArgumentParser
    import_cell_history: argparse.ArgumentParser
    export_session: argparse.ArgumentParser
    import_session: argparse.ArgumentParser


def build_perfmonitor_start_parser() -> argparse.ArgumentParser:
    """Build an ArgumentParser for the perfmonitor_start command."""
    default_interval = load_config().settings.monitoring.default_interval
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "interval",
        nargs="?",
        type=float,
        default=None,
        help=f"Sampling interval in seconds (default: {default_interval})",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="default",
        choices=["default", "native_c", "subprocess_python", "thread", "slurm_multinode"],
        help="Monitor backend to use (default: subprocess-based Python collector, "
             "'native_c' for native C collector, "
             "'subprocess_python' for explicit Python subprocess collector, "
             "'thread' for in-process threaded monitor)",
    )
    parser.add_argument(
        "--check-sanity",
        dest="check_sanity",
        action="store_true",
        help="Run a short sanity check of the selected monitor before "
             "starting real monitoring. Tailored for thread, "
             "subprocess_python and native_c monitors; other monitors "
             "are expected to fail this check.",
    )
    return parser


def build_perfreport_parser() -> argparse.ArgumentParser:
    """Build an ArgumentParser instance for JUmPER commands."""
    perfreports = load_config().settings.perfreports
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cells",
        type=str,
        help="Cell index or range (e.g., 5, 2:8, :5)"
    )
    parser.add_argument(
        "--level",
        default=perfreports.level,
        choices=get_available_levels(),
        help="Performance level",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        default=perfreports.text,
        help="Show report in text format"
    )
    return parser

def build_ai_review_parser() -> argparse.ArgumentParser:
    """Build an ArgumentParser for the perfmonitor_ai_review command."""
    from jumper_extension.adapters.ai_reviewer.strategy import strategy_ids

    ai_context = load_config().ai.context
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cells",
        type=str,
        help="Cell index or range to analyze (e.g., 5, 2:8, :5)"
    )
    parser.add_argument(
        "--strategy",
        default=ai_context.strategy,
        choices=strategy_ids(),
        help="Review strategy steering context sources and prompt rules",
    )
    parser.add_argument(
        "--level",
        default=load_config().settings.perfreports.level,
        choices=get_available_levels(),
        help="Performance level",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Run every suggestion, repair what fails and report a measured verdict. "
            "Replays the whole session per run - expect it to be slow"
        ),
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=None,
        metavar="N",
        help="Timed replays per suggestion; the first is dropped as warm-up",
    )
    parser.add_argument(
        "--fix-attempts",
        type=int,
        default=None,
        metavar="N",
        help="Repair rounds before a failing suggestion is reported as failed",
    )
    parser.add_argument(
        "--replay-mode",
        default=None,
        choices=("full", "fork", "dill"),
        help=(
            "How the state a timed cell needs is rebuilt between measurements. "
            "'full' re-runs every preceding cell each time; the others rebuild it "
            "once and reuse it. A mode that cannot serve this cell falls back to full"
        ),
    )
    parser.add_argument(
        "--result-names",
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated top-level names holding the cell's result (e.g. "
            "answers). Correctness is then decided by these alone, and a "
            "suggestion that stops binding them is reported as failed rather "
            "than timed. Default: every binding of the original cell"
        ),
    )
    _CHECK_NAMES = ("validate_syntax", "run")
    parser.add_argument(
        "--check",
        action="append",
        choices=_CHECK_NAMES,
        metavar="STEP",
        help=(
            "Run only these benchmark steps (repeatable); the rest are turned off. "
            f"Steps: {', '.join(_CHECK_NAMES)}"
        ),
    )
    parser.add_argument(
        "--skip-check",
        action="append",
        choices=_CHECK_NAMES,
        metavar="STEP",
        help="Turn off a benchmark step (repeatable); overrides config and --check",
    )
    parser.add_argument(
        "--resume",
        type=str,
        metavar="RUN_ID",
        help="Resume a previous review and apply one of its suggestions",
    )
    parser.add_argument(
        "--select",
        type=int,
        metavar="N",
        help="1-based index of the suggestion to apply (required with --resume)",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Free-text instruction; steers the suggestions, or rewrites the chosen one on --resume",
    )
    return parser

def build_perfmonitor_plot_parser() -> argparse.ArgumentParser:
    monitoring = load_config().settings.monitoring
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--metrics",
        type=str,
        help="Comma-separated list of metrics to plot directly"
    )
    parser.add_argument(
        "--cells",
        type=str,
        help="Cell index or range (e.g., 5, 2:8, :5)"
    )
    parser.add_argument(
        "--level",
        choices=get_available_levels(),
        help="Performance level for direct plotting",
    )
    parser.add_argument(
        "--save-jpeg",
        dest="save_jpeg",
        type=str,
        help="Save plot to a JPEG file"
    )
    parser.add_argument(
        "--pickle",
        dest="pickle_file",
        type=str,
        help="Serialize plot data to a pickle file"
    )
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "plotly"],
        help="Visualizer backend for this plot command",
    )
    parser.add_argument(
        "--live",
        nargs="*",
        type=float,
        metavar=("INTERVAL", "WINDOW"),
        help="Enable live-updating plots. Optional args: INTERVAL (update rate "
             f"in seconds, default {monitoring.live_update_interval}) and WINDOW "
             f"(sliding window in seconds, default {monitoring.live_window_seconds}). "
             "E.g. --live, --live 1.0, --live 2.0 60"
    )
    return parser

def build_auto_perfreports_parser() -> argparse.ArgumentParser:
    parser = build_perfreport_parser()
    default_interval = load_config().settings.monitoring.default_interval
    parser.add_argument(
        "--interval",
        type=float,
        default=default_interval,
        help=f"Interval between automatic reports (default: {default_interval})",
    )
    return parser

def build_export_perfdata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file", type=str, help="Output filename")
    parser.add_argument("--name", type=str, help="Custom DataFrame variable name")
    parser.add_argument(
        "--level",
        default=load_config().settings.perfreports.level,
        choices=get_available_levels(),
        help="Performance level",
    )
    return parser

def build_export_cell_history_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file", type=str, help="Output filename")
    parser.add_argument("--name", type=str, help="Custom DataFrame variable name")
    return parser

def build_import_perfdata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # Positional filename, no --file or --level required
    parser.add_argument("file", type=str, help="Input performance data filename")
    return parser

def build_import_cell_history_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # Positional filename, no --file required
    parser.add_argument("file", type=str, help="Input cell history filename")
    return parser

def build_export_session_parser() -> argparse.ArgumentParser:
    """Build parser for exporting a full session package.

    Usage examples in magics:
      %export_session                        # uses default directory name
      %export_session my_dir                 # export into directory
      %export_session my_session.zip         # export and zip (auto-detected by .zip extension)
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Target directory or .zip path (defaults to jumper-session-<timestamp>)",
    )
    return parser

def build_import_session_parser() -> argparse.ArgumentParser:
    """Build parser for importing a full session package from directory or zip."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "path",
        type=str,
        help="Path to exported session directory or .zip archive",
    )
    return parser

def parse_arguments(parser: argparse.ArgumentParser, line: str) -> Optional[argparse.Namespace]:
    """Parse common command line arguments for JUmPER commands.
    
    Args:
        line: The command line string to parse
        parser: Optional existing ArgumentParser instance
        
    Returns:
        Parsed arguments or None if parsing failed
    """
    try:
        args = (
            parser.parse_args(shlex.split(line))
            if line
            else parser.parse_args([])
        )
    except (SystemExit, Exception):
        args = None
    return args


def parse_cell_range(cell_str: str, cell_history_length: int) -> Optional[Tuple[int, int]]:
    """Parse a cell range string into start and end indices.
    
    Args:
        cell_str: String representing cell range (e.g., "1:3", "5", ":10")
        cell_history_length: Length of cell history
        
    Returns:
        Tuple of (start_idx, end_idx) or None if invalid
    """
    if not cell_str:
        return None
        
    try:
        max_idx = cell_history_length - 1
        if ":" in cell_str:
            start_str, end_str = cell_str.split(":", 1)
            start_idx = 0 if not start_str else int(start_str)
            end_idx = max_idx if not end_str else int(end_str)
        else:
            start_idx = end_idx = int(cell_str)
            
        if 0 <= start_idx <= end_idx <= max_idx:
            return start_idx, end_idx
    except (ValueError, IndexError, AttributeError):
        pass
        
    return None
