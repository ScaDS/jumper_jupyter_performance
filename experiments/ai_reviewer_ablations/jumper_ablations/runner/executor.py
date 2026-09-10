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
    target_cell_index: int | None = None
    error: str = ""

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


def _kernel_environment(pass_directory: Path) -> dict:
    """What the kernel is told before it starts.

    The strategies path is how an ablation reaches the reviewer at all: the
    presets live in this experiment's folder and the reviewer merges them over
    its built-ins, so `--strategy no_timing` resolves without the extension
    knowing this experiment exists.
    """
    environment = {
        "JUMPER_AI_STRATEGIES_PATH": str(STRATEGIES_FILE),
        # Per pass, so ai_prompts.log can be checked against the messages the
        # record claims were sent.
        "JUMPER_LOG_DIR": str(pass_directory / "logs"),
        # The benchmark mkdtemps a session export per measurement and never
        # cleans up; keeping them here keeps them findable and prunable.
        "TMPDIR": str(pass_directory / "tmp"),
    }
    api_key = os.environ.get(API_KEY_ENV)
    if api_key:
        environment[API_KEY_ENV] = api_key
    return environment


def _code_cells_up_to_payload(usecase: Usecase) -> list[str]:
    """The notebook's own cells, verbatim, up to and including the target."""
    notebook = read_notebook(usecase.notebook_path)
    sources = []
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
    payload_source = notebook.cells[usecase.layout.payload_index].get("source", "")
    payload_path = pass_directory / "payload.py"
    payload_path.write_text(payload_source, encoding="utf-8")

    session = KernelSession(
        kernel_name=protocol.kernel.name,
        cell_timeout=protocol.kernel.cell_timeout,
        startup_timeout=protocol.kernel.startup_timeout,
        working_directory=workspace_for(config, usecase),
        environment=_kernel_environment(pass_directory),
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
            )
        )
        if not result.ok:
            outcome.status = "bootstrap_failed"
            outcome.error = result.error
            return outcome

        target_index = None
        if protocol.pin_target_cell:
            result = session.run(cell_plan.resolve_target_cell(str(payload_path)))
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

        for generation in range(1, protocol.generations_per_target + 1):
            captured = _one_generation(
                session=session,
                protocol=protocol,
                usecase=usecase,
                generation=generation,
                review_command=line,
                replay_mode=replay_mode,
                inline_benchmark=inline,
            )
            outcome.generations += 1
            outcome.captures += captured
            if captured == 0:
                outcome.status = "capture_failed"

    return outcome


def _one_generation(
    session: KernelSession,
    protocol,
    usecase: Usecase,
    generation: int,
    review_command: str,
    replay_mode: str,
    inline_benchmark: bool,
) -> int:
    """One review, then a measurement of it under each applicable mode.

    With the two-command shape the review is captured on its own first, so the
    suggestions are on record as the model wrote them; the benchmark then
    re-opens the same run id. With the inline shape the notebook's own
    ``--benchmark`` review does both at once and there is nothing to re-open
    unless another replay mode was asked for.
    """
    captures = 0
    session.run(
        cell_plan.begin_cell(
            generation=generation,
            phase=PHASE_REVIEW,
            replay_mode=replay_mode if inline_benchmark else "",
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
            f"{'(' + result.error + ')' if result.error else '(no marker in output)'}"
        )
        return captures
    captures += 1

    if not summaries[-1].get("populated_sources"):
        # Not a measurement. The reviewer collects nothing when the target
        # cell has no performance data, warns, and then asks the model anyway
        # - which answers, plausibly and about nothing. Saying so here is the
        # difference between an experiment and a pile of confabulations.
        logger.error(
            f"{usecase.id} generation {generation}: the reviewer received an "
            "empty context; check that the payload runs long enough to be "
            "sampled and that the monitor is running"
        )

    reviewer_run_id = summaries[-1].get("reviewer_run_id") or ""
    if not reviewer_run_id or not protocol.benchmark.enabled:
        return captures

    modes = list(protocol.benchmark.extra_replay_modes)
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
    return captures
