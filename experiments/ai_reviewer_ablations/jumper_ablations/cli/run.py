"""Run the experiment: the real magic, in a real kernel, recorded.

The suite says which usecases are crossed with which ablations; the protocol
says how hard each cell of that grid is hit. Hydra composes both and writes
the composed config into the run directory, so a result and the setup that
produced it stay together.

    python -m jumper_ablations.cli.run suite=context_sources
    python -m jumper_ablations.cli.run suite=smoke protocol=pilot
    python -m jumper_ablations.cli.run suite=smoke \
        protocol.generations_per_target=1
"""

from __future__ import annotations

import json
import logging
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from jumper_ablations.config.schema import load_experiment_config
from jumper_ablations.paths import CLI_CONFIG_PATH, register_resolvers
from jumper_ablations.records.store import RecordStore
from jumper_ablations.runner.executor import run_pass
from jumper_ablations.runner.run_directory import RunDirectory, machine
from jumper_ablations.strategies import build_strategies_file, compose_ablation
from jumper_ablations.usecases.registry import get_usecase

logger = logging.getLogger("jumper_ablations")

register_resolvers()


# What a resume has to agree with the stored records about. The machine and
# the timestamp are deliberately absent: continuing a sweep on another node
# is a normal thing to do and is recorded rather than refused.
_RESUME_INVARIANTS = ("suite", "protocol", "usecases", "ablations")


def _refuse_incompatible_resume(
    existing: dict,
    snapshot: dict,
    run_directory: RunDirectory,
) -> None:
    """Stop a resume that would mix two different experiments in one run.

    Reusing a run id picks up the records already there. If the suite, the
    protocol, a usecase manifest or an ablation has changed since, the result
    is one directory holding measurements of two different things, with
    metadata describing only the later one - and nothing in the records says
    which half is which. Refusing costs a new run id.
    """
    changed = [
        field
        for field in _RESUME_INVARIANTS
        if existing.get(field) != snapshot.get(field)
    ]
    if not changed:
        return
    raise SystemExit(
        f"{run_directory.path.name} already holds records produced under a "
        f"different {', '.join(changed)}. Resuming would mix two experiments "
        "in one run directory. Use a new run_id, or delete this one."
    )


def _already_recorded(run_directory: RunDirectory) -> set:
    """Passes this run directory already holds, as (usecase, ablation, rep).

    A sweep is hours to days of machine time and a kernel can die for reasons
    that have nothing to do with the experiment. Re-running with the same
    run_id then continues where it stopped instead of paying for the finished
    passes again. Only successful passes count: a failed one is worth another
    try, and its records were never written.
    """
    if not run_directory.passes_index.is_file():
        return set()

    done = set()
    for line in run_directory.passes_index.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("status") == "ok":
            done.add(
                (
                    entry.get("usecase"),
                    entry.get("ablation"),
                    entry.get("repetition"),
                )
            )
    return done


@hydra.main(
    version_base=None, config_path=CLI_CONFIG_PATH, config_name="config"
)
def main(raw_config: DictConfig) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_experiment_config(raw_config)
    run_directory = RunDirectory.create(HydraConfig.get().runtime.output_dir)

    # Regenerated every run: a preset edited since the last one must not be
    # measured under its old meaning.
    build_strategies_file()

    usecases = [
        get_usecase(one, config.usecases_root) for one in config.suite.usecases
    ]
    ablations = [compose_ablation(one) for one in config.suite.ablations]

    snapshot = {
        "run_id": config.run_id,
        "suite": config.suite.model_dump(),
        "protocol": config.protocol.model_dump(),
        "usecases": {
            usecase.id: usecase.manifest.model_dump() for usecase in usecases
        },
        "ablations": {
            ablation.id: ablation.model_dump() for ablation in ablations
        },
        "machine": machine(),
        "config": OmegaConf.to_container(raw_config, resolve=True),
    }
    existing = run_directory.meta()
    if existing:
        _refuse_incompatible_resume(existing, snapshot, run_directory)
        run_directory.record_invocation(machine())
    else:
        run_directory.write_meta(snapshot)
    run_directory.snapshot_strategies()

    total = len(usecases) * len(ablations) * config.protocol.repetitions
    protocol = config.protocol
    logger.info(
        f"suite '{config.suite.id}': {len(usecases)} usecase(s) x "
        f"{len(ablations)} ablation(s) x {protocol.repetitions} "
        f"repetition(s) = {total} kernel pass(es) into {run_directory.path}"
    )
    # Stated before anything runs, because the difference between a
    # context-collection pass and a full sweep is two config tokens and about
    # three orders of magnitude of machine time.
    logger.info(
        f"protocol: {protocol.generations_per_target} generation(s) per pass; "
        + (
            f"benchmark on, {protocol.benchmark.runs} run(s) per suggestion, "
            f"mode={protocol.benchmark.mode}"
            if protocol.benchmark.enabled
            else "benchmark OFF - no suggestion will be measured"
        )
    )

    done = _already_recorded(run_directory)
    if done:
        logger.info(f"resuming: {len(done)} pass(es) already recorded")

    failures = 0
    for usecase in usecases:
        for ablation in ablations:
            for repetition in range(config.protocol.repetitions):
                if (usecase.id, ablation.id, repetition) in done:
                    logger.info(
                        f"{usecase.id} | {ablation.id} | r{repetition} | "
                        "already recorded, skipping"
                    )
                    continue
                outcome = run_pass(
                    config=config,
                    run_directory=run_directory,
                    usecase=usecase,
                    ablation=ablation,
                    repetition=repetition,
                )
                run_directory.append_pass(outcome.as_entry())
                level = (
                    logging.INFO if outcome.status == "ok" else logging.ERROR
                )
                logger.log(
                    level,
                    f"{outcome.usecase} | {outcome.ablation} | "
                    f"r{outcome.repetition} | {outcome.status} | "
                    f"{outcome.captures} record(s)",
                )
                failures += outcome.status != "ok"

    store = RecordStore(run_directory.path)
    store.write_flat_view()
    logger.info(f"records in {run_directory.records}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
