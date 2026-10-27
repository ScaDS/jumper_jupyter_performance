"""Running child.py once per measurement, and turning its answer into a row."""

from __future__ import annotations

import concurrent.futures
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from jumper_ablations.usecases.notebook import cell_source, read_notebook
from jumper_ablations.usecases.registry import Usecase
from jumper_ablations.verification.schema import (
    EAGER,
    HUMAN_FAST,
    ORIGINAL,
    STATUS_FAILED,
    STATUS_TIMEOUT,
    TORCH_COMPILE,
    Measurement,
    References,
)

CHILD = Path(__file__).with_name("child.py")
SEED = 42

# IPython syntax has no meaning in a plain interpreter: line and cell magics,
# shell escapes and the extension's own setup are dropped from the prefix.
_IPYTHON_LINE = re.compile(r"^\s*[%!]")


def plain_python(source: str) -> str:
    return "\n".join(
        line for line in source.splitlines() if not _IPYTHON_LINE.match(line)
    )


def notebook_parts(usecase: Usecase) -> tuple[str, str]:
    """The prefix as plain Python, and the original payload cell."""
    notebook = read_notebook(usecase.notebook_path)
    prefix = "\n\n".join(
        plain_python(cell_source(notebook, index))
        for index in usecase.layout.prefix_indices
        if notebook.cells[index].get("cell_type") == "code"
    )
    return prefix, cell_source(notebook, usecase.layout.payload_index)


def run_child(label: str, job: dict, timeout_s: float) -> tuple[Measurement, dict]:
    """One fresh process; the measurement, plus the raw result for answers."""
    with tempfile.TemporaryDirectory(prefix="verify-") as directory:
        job_path = Path(directory) / "job.json"
        result_path = Path(directory) / "result.json"
        job_path.write_text(json.dumps(job), encoding="utf-8")
        try:
            subprocess.run(
                [sys.executable, str(CHILD), str(job_path), str(result_path)],
                cwd=directory,
                timeout=timeout_s,
                capture_output=True,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Measurement(label=label, status=STATUS_TIMEOUT), {}
        if not result_path.is_file():
            return (
                Measurement(
                    label=label,
                    status=STATUS_FAILED,
                    error="the process died",
                ),
                {},
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))

    return (
        Measurement(
            label=label,
            status=result.get("status", STATUS_FAILED),
            time_s=result.get("time_s"),
            peak_memory_bytes=result.get("peak_memory_bytes"),
            correct=result.get("correct"),
            error=result.get("error", ""),
        ),
        result,
    )


def _kernelbench_job(
    usecase: Usecase,
    candidate: str | None,
    compiled: bool = False,
) -> dict:
    spec = usecase.manifest.verification
    prefix, original = notebook_parts(usecase)
    return {
        "kind": "kernelbench",
        "prefix": prefix,
        "original": original,
        "candidate": candidate,
        "compile": compiled,
        "seed": SEED,
        "trials": spec.trials,
        "warmup": spec.warmup,
        "correctness_trials": spec.correctness_trials,
        "atol": spec.atol,
        "rtol": spec.rtol,
    }


def _ecco_job(usecase: Usecase, program: str, program_kind: str) -> dict:
    prefix, _original = notebook_parts(usecase)
    return {
        "kind": "ecco",
        "prefix": prefix,
        "program": program,
        "program_kind": program_kind,
    }


def _solution_path(usecase: Usecase, relative: str) -> Path:
    return usecase.manifest_path.parent / relative


def measure_references(usecase: Usecase) -> References:
    """Everything the suggestions of one usecase are compared against.

    For ECCO the expected answers are what the human fast solution prints:
    the scaled inputs have no published outputs, and the fast solution is the
    one the dataset vouches for.
    """
    spec = usecase.manifest.verification
    references = References(usecase=usecase.id, kind=spec.kind)
    if spec.kind == "kernelbench":
        for label, compiled in ((EAGER, False), (TORCH_COMPILE, True)):
            measurement, _ = run_child(
                label,
                _kernelbench_job(usecase, None, compiled),
                spec.timeout_s,
            )
            references.entries[label] = measurement
        return references

    _prefix, original = notebook_parts(usecase)
    original_measurement, original_result = run_child(
        ORIGINAL,
        _ecco_job(usecase, original, "cell"),
        spec.timeout_s,
    )
    references.entries[ORIGINAL] = original_measurement
    human_code = _solution_path(usecase, spec.reference_solution).read_text(encoding="utf-8")
    measurement, result = run_child(
        HUMAN_FAST,
        _ecco_job(usecase, human_code, "script"),
        spec.timeout_s,
    )
    references.entries[HUMAN_FAST] = measurement
    expected = result.get("answers") or []
    references.expected_answers = expected

    submissions = sorted(_solution_path(usecase, spec.submissions).glob("*.py"))
    with concurrent.futures.ThreadPoolExecutor(spec.parallel_submissions) as pool:
        futures = [
            pool.submit(
                run_child,
                path.stem,
                _ecco_job(usecase, path.read_text(encoding="utf-8", errors="replace"), "script"),
                spec.timeout_s,
            )
            for path in submissions
        ]
        for future in futures:
            measurement, result = future.result()
            if measurement.ok:
                measurement.correct = result.get("answers") == expected
            references.population.append(measurement)
    # The slow and the fast solution have to agree, or the usecase's
    # generator produced inputs outside the problem's constraints.
    if original_measurement.ok:
        original_measurement.correct = original_result.get("answers") == expected
    if references.entries[HUMAN_FAST].ok:
        references.entries[HUMAN_FAST].correct = True
    return references


def measure_suggestion(
    usecase: Usecase,
    label: str,
    code: str,
    references: References,
) -> Measurement:
    spec = usecase.manifest.verification
    if spec.kind == "kernelbench":
        measurement, _ = run_child(
            label,
            _kernelbench_job(usecase, code),
            spec.timeout_s,
        )
        return measurement
    measurement, result = run_child(
        label,
        _ecco_job(usecase, code, "cell"),
        spec.timeout_s,
    )
    if measurement.ok:
        measurement.correct = result.get("answers") == references.expected_answers
    return measurement
