"""cli.verify's ECCO path, end to end on a usecase small enough to run here.

KernelBench needs a GPU and is exercised on capella; the ECCO path is plain
Python, so its whole chain - prefix extraction, one fresh process per
program, expected answers from the human solution, the population - runs in
the test.
"""

from __future__ import annotations

import json

import yaml

from jumper_ablations.usecases.registry import get_usecase
from jumper_ablations.verification.measure import (
    measure_references,
    measure_suggestion,
    plain_python,
)
from jumper_ablations.verification.schema import HUMAN_FAST, ORIGINAL

PREFIX = "scaled_inputs = [f'{n}\\n' for n in (10, 20, 30)]"
SLOW = (
    "def solve(text):\n"
    "    n = int(text)\n"
    "    return str(sum(i for i in range(n + 1)))\n"
    "\n"
    "answers = [solve(text) for text in scaled_inputs]"
)
HUMAN = "n = int(input())\nprint(n * (n + 1) // 2)"


def _cell(kind: str, source: str, index: int) -> dict:
    entry = {"cell_type": kind, "id": f"cell-{index}", "metadata": {}, "source": source}
    if kind == "code":
        entry.update(execution_count=None, outputs=[])
    return entry


def _usecase(root):
    family = root / "ecco"
    family.mkdir()
    notebook = {
        "cells": [
            _cell("markdown", "# Triangular numbers\n#### Payload type: loop", 0),
            _cell("code", "%load_ext jumper_extension\n%perfmonitor_fast_setup", 1),
            _cell("code", PREFIX, 2),
            _cell("code", SLOW, 3),
            _cell("code", "%perfmonitor_ai_review --benchmark --replay-mode full", 4),
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (family / "triangle.ipynb").write_text(json.dumps(notebook))
    (family / "triangle").mkdir()
    (family / "triangle" / "human_fast.py").write_text(HUMAN)
    submissions = family / "triangle" / "submissions"
    submissions.mkdir()
    (submissions / "s1.py").write_text(HUMAN)
    (submissions / "s2.py").write_text("print(0)")
    (submissions / "s3.py").write_text("raise ValueError")
    (family / "triangle.yaml").write_text(
        yaml.safe_dump(
            {
                "verification": {
                    "kind": "ecco",
                    "timeout_s": 60,
                    "reference_solution": "triangle/human_fast.py",
                    "submissions": "triangle/submissions",
                    "parallel_submissions": 2,
                }
            }
        )
    )
    return get_usecase("ecco/triangle", root=root)


def test_plain_python_drops_ipython_syntax():
    assert plain_python("%%time\nx = 1\n!ls\ny = 2") == "x = 1\ny = 2"


def test_ecco_references_and_suggestion(tmp_path):
    usecase = _usecase(tmp_path)

    references = measure_references(usecase)

    assert references.expected_answers == [["55"], ["210"], ["465"]]
    assert references.entries[ORIGINAL].correct is True
    assert references.entries[HUMAN_FAST].ok
    assert references.entries[HUMAN_FAST].peak_memory_bytes > 0
    by_label = {measurement.label: measurement for measurement in references.population}
    assert by_label["s1"].correct is True
    assert by_label["s2"].correct is False
    assert by_label["s3"].status == "failed"

    right = measure_suggestion(
        usecase,
        "1",
        "def solve(text):\n    n = int(text)\n    return str(n * (n + 1) // 2)\n"
        "answers = [solve(text) for text in scaled_inputs]",
        references,
    )
    wrong = measure_suggestion(
        usecase,
        "2",
        "answers = ['0' for text in scaled_inputs]",
        references,
    )
    incomplete = measure_suggestion(usecase, "3", "result = 1", references)

    assert right.ok and right.correct is True
    assert wrong.ok and wrong.correct is False
    assert incomplete.status == "failed" and incomplete.correct is False
