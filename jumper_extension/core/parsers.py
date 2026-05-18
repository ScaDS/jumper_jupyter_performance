"""Module containing parser utilities for the JUmPER extension."""
import argparse
import shlex
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any

from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.utilities import get_available_levels


@dataclass
class ArgParsers:
    """Configuration for command-line argument parsers."""
    perfreport: argparse.ArgumentParser
    auto_perfreports: argparse.ArgumentParser
    perfmonitor_plot: argparse.ArgumentParser
    export_perfdata: argparse.ArgumentParser
    export_cell_history: argparse.ArgumentParser
    import_perfdata: argparse.ArgumentParser
    import_cell_history: argparse.ArgumentParser
    export_session: argparse.ArgumentParser
    import_session: argparse.ArgumentParser


def build_perfreport_parser() -> argparse.ArgumentParser:
    """Build an ArgumentParser instance for JUmPER commands."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cell",
        type=str,
        help="Cell index or range (e.g., 5, 2:8, :5)"
    )
    parser.add_argument(
        "--level",
        default="process",
        choices=get_available_levels(),
        help="Performance level",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Show report in text format"
    )
    return parser

def build_perfmonitor_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--metrics",
        type=str,
        help="Comma-separated list of metrics to plot directly"
    )
    parser.add_argument(
        "--cell",
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
             "in seconds, default 2.0) and WINDOW (sliding window in seconds, "
             "default 120). E.g. --live, --live 1.0, --live 2.0 60"
    )
    return parser

def build_auto_perfreports_parser() -> argparse.ArgumentParser:
    parser = build_perfreport_parser()
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval between automatic reports (default: 1 second)",
    )
    return parser

def build_export_perfdata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file", type=str, help="Output filename")
    parser.add_argument("--name", type=str, help="Custom DataFrame variable name")
    parser.add_argument(
        "--level",
        default="process",
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
    except Exception:
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
