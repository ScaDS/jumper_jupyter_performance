"""The program cli.verify runs in a fresh process for every measurement.

    python child.py <job.json> <result.json>

Standalone on purpose - it imports nothing from the harness - so that what it
measures is the job and an interpreter, and a peak resident set read at the
end belongs to that one program the way Judge0's memory figure does.

Two job kinds, matching the two papers:

``kernelbench`` executes the notebook prefix and the original payload cell,
then the candidate cell, with ``iterations = 1`` so the payload's own loop is
cheap. It then times ``Model(*get_init_inputs())`` per call with CUDA events
and compares its outputs with the original's using ``allclose`` on fresh
random inputs - KernelBench's protocol.

``ecco`` executes the notebook prefix (which builds ``scaled_inputs``) and
then either a cell, read back through its ``answers``, or a stdin/stdout
script run once per input.
"""

from __future__ import annotations

import io
import itertools
import json
import os
import resource
import sys
import time
import traceback


def _peak_memory_bytes() -> int:
    # ru_maxrss is in kilobytes on Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


_SOURCE_NUMBERS = itertools.count()


def _compile(code: str, stem: str):
    """Compile *code* as if it came from a file, and write that file.

    Code compiled from a string has no source file, and libraries that look
    for one fail where a notebook cell would not: numba's ``cache=True``,
    ``inspect.getsource``. The file goes into the working directory, which
    run_child makes a temporary one per measurement.
    """
    path = os.path.abspath(f"{stem}_{next(_SOURCE_NUMBERS)}.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(code)
    return compile(code, path, "exec")


def _execute(code: str, namespace: dict) -> None:
    exec(_compile(code, "cell"), namespace)


def _kernelbench(job: dict) -> dict:
    import torch

    device = torch.device("cuda")
    namespace: dict = {}
    _execute(job["prefix"], namespace)
    namespace["iterations"] = 1

    reference_namespace = dict(namespace)
    _execute(job["original"], reference_namespace)
    reference_class = reference_namespace["Model"]

    if job.get("candidate") is None:
        candidate_class = reference_class
    else:
        candidate_namespace = dict(namespace)
        _execute(job["candidate"], candidate_namespace)
        candidate_class = candidate_namespace.get("ModelNew") or candidate_namespace.get("Model")
        if candidate_class is None:
            return {"status": "unmeasurable", "error": "the cell defines no Model class"}

    get_inputs = namespace["get_inputs"]
    get_init_inputs = namespace["get_init_inputs"]
    seed = job["seed"]

    def build(model_class):
        torch.manual_seed(seed)
        model = model_class(*get_init_inputs()).to(device)
        if job.get("compile"):
            model = torch.compile(model)
        return model

    with torch.no_grad():
        correct = None
        if job.get("candidate") is not None:
            reference_model = build(reference_class)
            candidate_model = build(candidate_class)
            correct = True
            for trial in range(job["correctness_trials"]):
                torch.manual_seed(seed + 1 + trial)
                inputs = [tensor.to(device) for tensor in get_inputs()]
                expected = reference_model(*inputs)
                produced = candidate_model(*inputs)
                torch.cuda.synchronize()
                if produced.shape != expected.shape or not torch.allclose(
                    produced,
                    expected,
                    atol=job["atol"],
                    rtol=job["rtol"],
                ):
                    correct = False
                    break
                del inputs, expected, produced
            del reference_model, candidate_model
            torch.cuda.empty_cache()

        torch.manual_seed(seed)
        inputs = [tensor.to(device) for tensor in get_inputs()]
        model = build(candidate_class)
        for _ in range(job["warmup"]):
            model(*inputs)
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(job["trials"]):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            torch.cuda.synchronize()
            elapsed.append(start.elapsed_time(end) / 1000.0)

    return {
        "status": "ok",
        "time_s": sum(elapsed) / len(elapsed),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "correct": correct,
    }


def _run_script(code: str, text: str) -> str:
    compiled = _compile(code, "solution")
    real_stdin, real_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(text)
    sys.stdout = captured = io.StringIO()
    try:
        exec(compiled, {"__name__": "__main__"})
    except SystemExit:
        pass
    finally:
        sys.stdin, sys.stdout = real_stdin, real_stdout
    return captured.getvalue()


def _ecco(job: dict) -> dict:
    namespace: dict = {}
    _execute(job["prefix"], namespace)
    scaled_inputs = namespace["scaled_inputs"]

    start = time.perf_counter()
    if job["program_kind"] == "cell":
        _execute(job["program"], namespace)
        answers = namespace.get("answers")
        # The reviewer is asked for the complete cell. One that rewrote the
        # solution and dropped the lines running it is an incomplete
        # suggestion, and fails like any other broken one.
        if answers is None:
            return {
                "status": "failed",
                "correct": False,
                "error": "incomplete cell: it defines no answers",
            }
    else:
        answers = [_run_script(job["program"], text) for text in scaled_inputs]
    elapsed = time.perf_counter() - start

    return {
        "status": "ok",
        "time_s": elapsed,
        "peak_memory_bytes": _peak_memory_bytes(),
        "answers": [str(answer).split() for answer in answers],
    }


def main(job_path: str, result_path: str) -> None:
    with open(job_path) as handle:
        job = json.load(handle)
    try:
        result = _kernelbench(job) if job["kind"] == "kernelbench" else _ecco(job)
    except BaseException:
        result = {"status": "failed", "error": traceback.format_exc(limit=5)[-2000:]}
    with open(result_path, "w") as handle:
        json.dump(result, handle)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
