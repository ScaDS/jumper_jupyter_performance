"""The cells a pass executes, and the magic lines inside them.

The notebook is not rewritten. Its own cells run verbatim, its own review line
is the one that runs, and the harness only appends the flags the experiment is
varying. That is the whole fidelity claim: what gets measured is the command a
user types, with a preset selected.

The one flag the harness insists on is ``--cells``. Left off, the magic picks
"the last cell that was not short", and the harness's own injected cells are
sitting right after the payload - whether one of them is short enough to be
skipped is a timing accident, not a design.
"""
from __future__ import annotations

import pprint
import shlex

from jumper_ablations.config.schema import ProtocolConfig

REVIEW_MAGIC = "%perfmonitor_ai_review"

# Flags the experiment owns. Whatever the notebook said about them is dropped
# and re-stated from the config, so a run can never be half-configured.
_MANAGED_WITH_VALUE = (
    "--strategy",
    "--cells",
    "--benchmark-runs",
    "--fix-attempts",
    "--replay-mode",
    "--resume",
)
_MANAGED_BOOLEAN = ("--benchmark",)


def _strip_managed(tokens: list[str]) -> list[str]:
    """Everything the notebook asked for except what the harness sets."""
    kept: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        name = token.split("=", 1)[0]
        if name in _MANAGED_WITH_VALUE:
            skip_next = "=" not in token
            continue
        if name in _MANAGED_BOOLEAN:
            continue
        kept.append(token)
    return kept


def review_line(
    base_line: str,
    ablation_id: str,
    replay_mode: str,
    protocol: ProtocolConfig,
    target_cell_index: int | None = None,
    with_benchmark: bool = True,
) -> str:
    """The notebook's review invocation, with this ablation selected."""
    tokens = shlex.split(base_line)
    if not tokens or tokens[0] != REVIEW_MAGIC:
        raise ValueError(f"not a review invocation: {base_line!r}")

    rebuilt = [REVIEW_MAGIC, *_strip_managed(tokens[1:])]
    rebuilt += ["--strategy", ablation_id]
    if target_cell_index is not None:
        rebuilt += ["--cells", str(target_cell_index)]
    if with_benchmark and protocol.benchmark.enabled:
        rebuilt += [
            "--benchmark",
            "--benchmark-runs",
            str(protocol.benchmark.runs),
            "--fix-attempts",
            str(protocol.benchmark.fix_attempts),
            "--replay-mode",
            replay_mode,
        ]
    return " ".join(rebuilt)


def resume_benchmark_line(
    reviewer_run_id: str,
    replay_mode: str,
    protocol: ProtocolConfig,
) -> str:
    """Re-score suggestions that already exist, under another replay mode.

    The model is not asked again: the plan calls for one stored set of
    suggestions evaluated under every applicable mode, so that a difference
    between modes is the instrument and not a new sample.
    """
    return " ".join(
        [
            REVIEW_MAGIC,
            "--resume",
            reviewer_run_id,
            "--benchmark",
            "--benchmark-runs",
            str(protocol.benchmark.runs),
            "--fix-attempts",
            str(protocol.benchmark.fix_attempts),
            "--replay-mode",
            replay_mode,
        ]
    )


def _python_literal(mapping: dict) -> str:
    """Render *mapping* as Python source for an injected cell.

    Python source, not JSON: an injected cell is executed, and JSON writes
    None as `null`, which is a NameError three cells later rather than an
    error where the mistake was made.
    """
    return pprint.pformat(mapping, indent=4, width=72, sort_dicts=False)


def bootstrap_cell(
    run_directory: str,
    usecase: str,
    ablation: str,
    ablation_family: str,
    repetition: int,
    suite: str,
    run_id: str,
) -> str:
    arguments = _python_literal(
        {
            "run_directory": run_directory,
            "usecase": usecase,
            "ablation": ablation,
            "ablation_family": ablation_family,
            "repetition": repetition,
            "suite": suite,
            "run_id": run_id,
        }
    )
    return (
        "from jumper_ablations import runtime\n"
        f"runtime.bootstrap(**{arguments})\n"
    )


def resolve_target_cell(payload_path: str) -> str:
    return (
        "from jumper_ablations import runtime\n"
        f"runtime.resolve_target({payload_path!r})\n"
    )


def begin_cell(
    generation: int,
    phase: str,
    replay_mode: str,
    reviewer_run_id: str | None = None,
) -> str:
    arguments = _python_literal(
        {
            "generation": generation,
            "phase": phase,
            "requested_replay_mode": replay_mode,
            "reviewer_run_id": reviewer_run_id,
        }
    )
    return (
        "from jumper_ablations import runtime\n"
        f"runtime.begin(**{arguments})\n"
    )


def capture_cell() -> str:
    return "from jumper_ablations import runtime\nruntime.capture()\n"
