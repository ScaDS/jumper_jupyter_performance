"""One pass of one usecase under one ablation, and the grid of them.

A pass is a kernel: the notebook's prefix runs once, the payload runs once,
and then the review magic is asked for as many independent generations as the
protocol wants. That ordering is the reason ten generations are affordable -
the prefix, which is the expensive part, is paid for once - and it is also
what makes generations comparable: every generation of a pass saw the same
machine state.

Ablations are crossed with usecases in the outer loops, generations paired by
index, so `base` generation 3 and `no_timing` generation 3 differ in the
preset and in nothing the harness controls.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
from pathlib import Path

from jumper_ablations.config.schema import AblationConfig, ExperimentConfig
from jumper_ablations.paths import STRATEGIES_FILE
from jumper_ablations.records.schema import PHASE_REBENCHMARK, PHASE_REVIEW
from jumper_ablations.runner import cell_plan
from jumper_ablations.runner.kernel_session import KernelSession
from jumper_ablations.runner.run_directory import RunDirectory
from jumper_ablations.runtime.session import CAPTURE_MARKER, TARGET_MARKER
from jumper_ablations.usecases.notebook import read_notebook
from jumper_ablations.usecases.registry import Usecase

logger = logging.getLogger("jumper_ablations")

# The reviewer needs this to talk to the model at all; it is the operator's to
# provide, and the harness only passes it through to the kernel.
API_KEY_ENV = "JUMPER_AI_API_KEY"


@dataclasses.dataclass
class PassOutcome:
    usecase: str
    ablation: str
    repetition: int
    status: str
    generations: int = 0
    captures: int = 0
    # What the protocol asked this pass to record. A pass short of it has
    # measured a different experiment from the one the report will describe,
    # and the difference has to be visible without re-reading the records.
    expected_captures: int = 0
    empty_context: int = 0
    target_cell_index: int | None = None
    error: str = ""

    @property
    def complete(self) -> bool:
        return (
            self.captures >= self.expected_captures and not self.empty_context
        )

    def as_entry(self) -> dict:
        return dataclasses.asdict(self)


def workspace_for(config: ExperimentConfig, usecase: Usecase) -> Path:
    """Where this usecase's notebook is executed.

    One directory per usecase family, because a family shares state on
    purpose: minian/cell_77 reads the intermediate store minian/cell_40's
    pipeline wrote, and giving them separate directories would mean
    recomputing it.
    """
    family = usecase.id.split("/", 1)[0]
    return Path(config.workspace_root) / family


def _kernel_environment(
    pass_directory: Path,
    usecase: Usecase | None = None,
    workspace: Path | None = None,
    strategies_path: Path = STRATEGIES_FILE,
) -> dict:
    """What the kernel is told before it starts.

    The strategies path is how an ablation reaches the reviewer at all: the
    presets live in this experiment's folder and the reviewer merges them over
    its built-ins, so `--strategy no_timing` resolves without the extension
    knowing this experiment exists. What is handed over is the run's own
    snapshot rather than the editable file it was copied from, so a preset
    edited mid-sweep cannot redefine the passes that have not run yet.
    """
    environment = {
        "JUMPER_AI_STRATEGIES_PATH": str(strategies_path),
        # Per pass, so ai_prompts.log can be checked against the messages the
        # record claims were sent.
        "JUMPER_LOG_DIR": str(pass_directory / "logs"),
        # The benchmark mkdtemps a session export per measurement and never
        # cleans up; keeping them here keeps them findable and prunable.
        "TMPDIR": str(pass_directory / "tmp"),
        # Dask puts its worker scratch in the working directory when nothing
        # says otherwise, and the working directory is shared by every pass
        # of a usecase family on purpose - that is how cell_77 reads what the
        # pipeline wrote. Two passes running at once then build their
        # clusters on top of each other, and the prefix dies deserialising
        # its own task graph. Shared data, private scratch.
        "DASK_TEMPORARY_DIRECTORY": str(pass_directory / "tmp"),
    }
    api_key = os.environ.get(API_KEY_ENV)
    if api_key:
        environment[API_KEY_ENV] = api_key
    if usecase is not None and workspace is not None:
        environment.update(usecase.manifest.resolved_environment(workspace))
    return environment


def _code_cells_up_to_payload(usecase: Usecase) -> list[str]:
    """The notebook's own cells, verbatim, up to and including the target.

    Nothing is added. A usecase notebook loads the extension and starts the
    monitor itself - that is what makes it runnable by hand - so the harness
    executing it is the same sequence a person would step through.
    """
    notebook = read_notebook(usecase.notebook_path)
    sources: list[str] = []
    for index in usecase.layout.executable_indices:
        cell = notebook.cells[index]
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if source.strip():
            sources.append(source)
    return sources


def run_pass(
    config: ExperimentConfig,
    run_directory: RunDirectory,
    usecase: Usecase,
    ablation: AblationConfig,
    repetition: int,
) -> PassOutcome:
    """Execute one kernel pass and return what it managed to record."""
    outcome = PassOutcome(
        usecase=usecase.id,
        ablation=ablation.id,
        repetition=repetition,
        status="ok",
    )
    protocol = config.protocol
    pass_directory = run_directory.pass_directory(
        usecase.id,
        ablation.id,
        repetition,
    )

    notebook = read_notebook(usecase.notebook_path)
    payload_source = notebook.cells[usecase.layout.payload_index].get(
        "source", ""
    )
    payload_path = pass_directory / "payload.py"
    payload_path.write_text(payload_source, encoding="utf-8")

    workspace = workspace_for(config, usecase)
    session = KernelSession(
        kernel_name=protocol.kernel.name,
        cell_timeout=protocol.kernel.cell_timeout,
        startup_timeout=protocol.kernel.startup_timeout,
        working_directory=workspace,
        environment=_kernel_environment(
            pass_directory,
            usecase,
            workspace,
            run_directory.strategies_snapshot,
        ),
    )

    with session:
        for source in _code_cells_up_to_payload(usecase):
            result = session.run(source)
            if not result.ok:
                outcome.status = "prefix_failed"
                outcome.error = result.error
                return outcome

        result = session.run(
            cell_plan.bootstrap_cell(
                run_directory=str(run_directory.path),
                usecase=usecase.id,
                ablation=ablation.id,
                ablation_family=ablation.family,
                repetition=repetition,
                suite=config.suite.id,
                run_id=config.run_id,
                sampling=protocol.sampling.as_applied(0, repetition),
            )
        )
        if not result.ok:
            outcome.status = "bootstrap_failed"
            outcome.error = result.error
            return outcome

        target_index = None
        if protocol.pin_target_cell:
            result = session.run(
                cell_plan.resolve_target_cell(str(payload_path))
            )
            markers = result.markers(TARGET_MARKER)
            if not result.ok or not markers:
                outcome.status = "target_unresolved"
                outcome.error = result.error or "no target marker in output"
                return outcome
            target_index = int(markers[-1]["cell_index"])
            outcome.target_cell_index = target_index

        replay_mode = usecase.manifest.benchmark.replay_mode
        inline = protocol.benchmark.mode == "inline"
        line = cell_plan.review_line(
            base_line=usecase.layout.review_line,
            ablation_id=ablation.id,
            replay_mode=replay_mode,
            protocol=protocol,
            target_cell_index=target_index,
            with_benchmark=inline,
        )
        logger.info(f"{usecase.id} | {ablation.id} | {line}")

        expected = _expected_captures(protocol, usecase, inline)
        for generation in range(1, protocol.generations_per_target + 1):
            captured, empty = _one_generation(
                session=session,
                protocol=protocol,
                usecase=usecase,
                generation=generation,
                repetition=repetition,
                review_command=line,
                replay_mode=replay_mode,
                inline_benchmark=inline,
            )
            outcome.generations += 1
            outcome.captures += captured
            outcome.expected_captures += expected
            outcome.empty_context += empty
            if captured == 0:
                outcome.status = "capture_failed"

    if outcome.status == "ok" and not outcome.complete:
        # Not a failure of the harness, but not a pass either: the records
        # that are here would be scored as though the missing ones had never
        # been asked for, and every rate would quietly be a rate over the
        # generations that happened to work.
        outcome.status = "incomplete"
        outcome.error = (
            f"{outcome.captures}/{outcome.expected_captures} record(s)"
            + (
                f", {outcome.empty_context} with no context"
                if outcome.empty_context
                else ""
            )
        )
        logger.error(
            f"{usecase.id} | {ablation.id} | r{repetition} | incomplete: "
            f"{outcome.error}"
        )

    return outcome


def prune_workdirs(pass_directory: Path) -> int:
    """Remove what the measurements left behind, and say how much that was.

    Every replay exports a session archive into the pass's temporary
    directory and nothing removes it, so a usecase whose prefix reads a
    multi-gigabyte store writes that store out once per measurement. Over a
    sweep of the minian usecases that is hundreds of gigabytes, none of it
    read again: the timings and the correctness verdicts are already in the
    records. Kept only when the protocol asks, for debugging a measurement.
    """
    scratch = pass_directory / "tmp"
    if not scratch.is_dir():
        return 0
    freed = sum(
        path.stat().st_size
        for path in scratch.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    shutil.rmtree(scratch, ignore_errors=True)
    if freed:
        logger.info(
            f"pruned {freed / 1e9:.1f} GB of benchmark scratch from "
            f"{pass_directory.name}"
        )
    return freed


def _expected_captures(protocol, usecase: Usecase, inline: bool) -> int:
    """Records one generation owes: the review, and each measurement."""
    if not protocol.benchmark.enabled:
        return 1
    modes = {
        *usecase.manifest.benchmark.extra_replay_modes,
        *protocol.benchmark.extra_replay_modes,
    }
    if not inline:
        modes.add(usecase.manifest.benchmark.replay_mode)
    return 1 + len(modes)


def _one_generation(
    session: KernelSession,
    protocol,
    usecase: Usecase,
    generation: int,
    repetition: int,
    review_command: str,
    replay_mode: str,
    inline_benchmark: bool,
) -> tuple[int, int]:
    """One review, then a measurement of it under each applicable mode.

    With the two-command shape the review is captured on its own first, so the
    suggestions are on record as the model wrote them; the benchmark then
    re-opens the same run id. With the inline shape the notebook's own
    ``--benchmark`` review does both at once and there is nothing to re-open
    unless another replay mode was asked for.
    """
    captures = 0
    empty = 0
    session.run(
        cell_plan.begin_cell(
            generation=generation,
            phase=PHASE_REVIEW,
            replay_mode=replay_mode if inline_benchmark else "",
            # Generation N of every preset draws under the same seed, so two
            # presets differ in their context and not in their sampling.
            sampling=protocol.sampling.as_applied(generation, repetition),
        )
    )
    review = session.run(review_command)
    if not review.ok:
        logger.warning(
            f"{usecase.id} generation {generation} failed: {review.error}"
        )
    result = session.run(cell_plan.capture_cell())
    summaries = result.markers(CAPTURE_MARKER)
    if not summaries:
        logger.error(
            f"{usecase.id} generation {generation}: nothing was captured "
            f"({result.error or 'no marker in output'})"
        )
        return captures, empty
    captures += 1

    if not summaries[-1].get("populated_sources"):
        # Not a measurement. The reviewer collects nothing when the target
        # cell has no performance data, warns, and then asks the model anyway
        # - which answers, plausibly and about nothing. Saying so here is the
        # difference between an experiment and a pile of confabulations.
        empty = 1
        logger.error(
            f"{usecase.id} generation {generation}: the reviewer received an "
            "empty context; check that the payload runs long enough to be "
            "sampled and that the monitor is running"
        )

    reviewer_run_id = summaries[-1].get("reviewer_run_id") or ""
    if not reviewer_run_id or not protocol.benchmark.enabled:
        return captures, empty

    # The usecase knows which fast modes its prefix survives; the protocol
    # knows which ones this sweep is asking about. A mode named in either is
    # measured, and neither silently overrules the other - a manifest whose
    # extra modes were ignored would make replay_mode_agreement report the
    # single-mode nulls while the yaml says two modes were compared.
    modes = [
        mode
        for mode in (
            *usecase.manifest.benchmark.extra_replay_modes,
            *protocol.benchmark.extra_replay_modes,
        )
        if mode != replay_mode
    ]
    modes = list(dict.fromkeys(modes))
    if not inline_benchmark:
        # The mode the usecase asked for is measured here rather than on the
        # review line, so it leads the list.
        modes = [replay_mode, *modes]

    for mode in modes:
        session.run(
            cell_plan.begin_cell(
                generation=generation,
                phase=PHASE_REBENCHMARK,
                replay_mode=mode,
                reviewer_run_id=reviewer_run_id,
                sampling=protocol.sampling.as_applied(generation, repetition),
            )
        )
        session.run(
            cell_plan.resume_benchmark_line(
                reviewer_run_id=reviewer_run_id,
                replay_mode=mode,
                protocol=protocol,
            )
        )
        again = session.run(cell_plan.capture_cell())
        if again.markers(CAPTURE_MARKER):
            captures += 1
    return captures, empty
