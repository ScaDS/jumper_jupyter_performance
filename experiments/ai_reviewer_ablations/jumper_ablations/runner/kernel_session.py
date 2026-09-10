"""A kernel to run one pass of a usecase notebook in, one cell at a time.

Executing the notebook as a document would be simpler, but the plan is not
known in advance: how many reviews to ask for depends on the protocol, and the
``--resume`` invocations depend on run ids the kernel only produces once the
first review has finished. So the harness drives the kernel cell by cell and
reads each cell's output before deciding the next one.

Every pass gets its own kernel. A usecase prefix builds a dask cluster and
gigabytes of intermediate state; carrying that between ablations would make
the second one a measurement of the first one's leftovers.
"""
from __future__ import annotations

import dataclasses
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, DeadKernelError


@dataclasses.dataclass
class CellResult:
    """What one executed cell produced."""

    source: str
    stdout: str
    error: str
    outputs: list

    @property
    def ok(self) -> bool:
        return not self.error

    def markers(self, marker: str) -> list[dict]:
        """Every ``<marker>{...}`` line the cell printed, decoded.

        The in-kernel half talks back over stdout because that is the one
        channel that survives whatever the notebook itself is printing.
        """
        found = []
        for line in self.stdout.splitlines():
            if line.startswith(marker):
                try:
                    found.append(json.loads(line[len(marker):]))
                except json.JSONDecodeError:
                    continue
        return found


@contextmanager
def _environment(overrides: dict) -> Iterator[None]:
    """Apply *overrides* to os.environ for the lifetime of the block.

    The kernel inherits the parent's environment, and this is where the
    reviewer is told which strategies file to read and where to log.
    """
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update({key: str(value) for key, value in overrides.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class KernelSession:
    """One live kernel, fed cells in order."""

    def __init__(
        self,
        kernel_name: str = "python3",
        cell_timeout: int = 14400,
        startup_timeout: int = 180,
        working_directory: Path | None = None,
        environment: dict | None = None,
    ):
        self.kernel_name = kernel_name
        self.cell_timeout = cell_timeout
        self.startup_timeout = startup_timeout
        self.working_directory = Path(working_directory or Path.cwd())
        self.environment = dict(environment or {})
        self.notebook = nbformat.v4.new_notebook(cells=[])
        self._client: NotebookClient | None = None
        self._kernel = None

    def __enter__(self) -> "KernelSession":
        self.working_directory.mkdir(parents=True, exist_ok=True)
        self._client = NotebookClient(
            self.notebook,
            kernel_name=self.kernel_name,
            timeout=self.cell_timeout,
            startup_timeout=self.startup_timeout,
            allow_errors=True,
            resources={"metadata": {"path": str(self.working_directory)}},
        )
        self._environment_guard = _environment(self.environment)
        self._environment_guard.__enter__()
        self._kernel = self._client.setup_kernel()
        self._kernel.__enter__()
        return self

    def __exit__(self, *exception) -> None:
        try:
            if self._kernel is not None:
                self._kernel.__exit__(*exception)
        finally:
            self._kernel = None
            self._client = None
            self._environment_guard.__exit__(*exception)

    def run(self, source: str) -> CellResult:
        """Execute *source* and return everything it produced."""
        if self._client is None:
            raise RuntimeError("KernelSession used outside its context")

        cell = nbformat.v4.new_code_cell(source)
        self.notebook.cells.append(cell)
        index = len(self.notebook.cells) - 1
        error = ""
        try:
            self._client.execute_cell(cell, index)
        except (CellExecutionError, DeadKernelError) as failure:
            error = str(failure)
        return CellResult(
            source=source,
            stdout=_stdout_of(cell),
            error=error or _error_of(cell),
            outputs=list(cell.get("outputs", [])),
        )


def _stdout_of(cell) -> str:
    parts = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            parts.append(output.get("text", ""))
    return "".join(parts)


def _error_of(cell) -> str:
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            name = output.get("ename", "Error")
            value = output.get("evalue", "")
            return f"{name}: {value}"
    return ""
