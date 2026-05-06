import logging
from typing import Callable

from jumper_extension.core.messages import (
    ExtensionInfoCode,
    EXTENSION_INFO_MESSAGES,
)
from jumper_extension.core.service import build_perfmonitor_magic_adapter
from jumper_extension.ipython.magics import PerfmonitorMagics
from jumper_extension.ipython.utilities import get_called_line_magics

logger = logging.getLogger("extension")
_perfmonitor_magics = None


class DropCellTransformer:
    """
    Drop the entire cell if it is being recorded.
    """
    def __init__(
        self,
        is_control_cell: Callable,
        is_recording_active: Callable
    ):
        self.is_control_cell = is_control_cell
        self.is_recording_active = is_recording_active

    def __call__(self, lines: list[str]) -> list[str]:
        """
        IPython cleanup_transforms expects a callable: (lines) -> lines.
        """
        cell = "".join(lines)
        new_cell = self.transform_cell(cell)

        # Keep IPython expectations: return list[str] with line endings preserved.
        if new_cell == "":
            return []
        return new_cell.splitlines(keepends=True)

    def transform_cell(self, cell: str) -> str:
        """
        Return an empty string to drop the whole cell.
        """
        if not self.is_recording_active():
            return cell

        called_line_magics = get_called_line_magics(cell)
        if self.is_control_cell(called_line_magics):
            return cell  # Allow control magics cell to execute

        return "print('[JUmPER]: Cell execution skipped during script recording')\n"


def load_ipython_extension(ipython):
    global _perfmonitor_magics
    magic_adapter = build_perfmonitor_magic_adapter(visualizer_backend='plotly')

    tm = ipython.input_transformer_manager
    drop_transformer = DropCellTransformer(
        is_control_cell=magic_adapter.service.script_writer.is_control_cell,
        is_recording_active=magic_adapter.service.script_writer.is_recording_active
    )
    ipython._drop_cell_transformer = drop_transformer
    tm.cleanup_transforms.append(drop_transformer)

    _perfmonitor_magics = PerfmonitorMagics(ipython, magic_adapter)
    ipython.events.register("pre_run_cell", _perfmonitor_magics.pre_run_cell)
    ipython.events.register("post_run_cell", _perfmonitor_magics.post_run_cell)
    ipython.register_magics(_perfmonitor_magics)
    logger.info(EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXTENSION_LOADED])


def unload_ipython_extension(ipython):
    tm = ipython.input_transformer_manager
    drop_transformer = getattr(ipython, "_drop_cell_transformer", None)
    if drop_transformer:
        if drop_transformer in tm.cleanup_transforms:
            tm.cleanup_transforms.remove(drop_transformer)
        del ipython._drop_cell_transformer

    global _perfmonitor_magics
    if _perfmonitor_magics:
        ipython.events.unregister(
            "pre_run_cell", _perfmonitor_magics.pre_run_cell
        )
        ipython.events.unregister(
            "post_run_cell", _perfmonitor_magics.post_run_cell
        )
        _perfmonitor_magics.magic_adapter.close()
        _perfmonitor_magics = None
