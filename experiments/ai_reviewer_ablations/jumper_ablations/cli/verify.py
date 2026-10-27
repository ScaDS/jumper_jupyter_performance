"""Re-measure a run's suggestions the way KernelBench and ECCO measure.

    python -m jumper_ablations.cli.verify
    python -m jumper_ablations.cli.verify target_run=20261027_143000

Runs after a sweep and before ``cli.evaluate``; it never calls the model.
Every usecase whose manifest has a ``verification`` block gets its references
measured once - the original Model eager and under torch.compile, or the
slow program, the human fast one and every CodeNet submission - and then
every suggestion of every review is measured the same way, each in a fresh
process. Everything lands in ``results/<run>/verification/``.

Resumable: a references file or a review file already on disk is kept, so a
job that ran out of time is simply started again.

It has to run where the usecase runs - KernelBench needs the GPU - and it
reads the notebooks on disk, because the prefix it executes is the notebook's.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from jumper_ablations.cli.common import configure_logging, resolve_run
from jumper_ablations.config.schema import load_experiment_config
from jumper_ablations.metrics.context import build_run_contexts
from jumper_ablations.paths import CLI_CONFIG_PATH, register_resolvers
from jumper_ablations.records.store import load_records
from jumper_ablations.usecases.registry import discover_usecases
from jumper_ablations.verification.measure import (
    measure_references,
    measure_suggestion,
)
from jumper_ablations.verification.schema import (
    REFERENCES_NAME,
    References,
    Verification,
    usecase_directory,
    write_model,
)

logger = logging.getLogger("jumper_ablations")

register_resolvers()


def _references(usecase, run_path: Path) -> References:
    path = usecase_directory(run_path, usecase.id) / REFERENCES_NAME
    if path.is_file():
        return References.model_validate_json(path.read_text(encoding="utf-8"))
    logger.info(f"{usecase.id}: measuring references")
    references = measure_references(usecase)
    write_model(references, path)
    for label, measurement in references.entries.items():
        logger.info(
            f"{usecase.id}: {label} {measurement.status} "
            f"time={measurement.time_s} correct={measurement.correct}"
        )
    if references.population:
        measured = sum(1 for measurement in references.population if measurement.ok)
        logger.info(
            f"{usecase.id}: {measured}/{len(references.population)} "
            "submissions measured"
        )
    return references


def _measured_suggestions(context) -> list:
    """The code the benchmark ran: after repair, when there was one."""
    if context.measurement is not None and context.measurement.outputs.suggestions:
        return context.measurement.outputs.suggestions
    return context.record.outputs.suggestions


@hydra.main(
    version_base=None, config_path=CLI_CONFIG_PATH, config_name="offline"
)
def main(raw_config: DictConfig) -> int:
    configure_logging()
    config = load_experiment_config(raw_config)
    run = resolve_run(config)
    usecases = {
        usecase_id: usecase
        for usecase_id, usecase in discover_usecases(Path(config.usecases_root)).items()
        if usecase.manifest.verification is not None
    }
    contexts = [
        context
        for context in build_run_contexts(list(load_records(run.path)), usecases)
        if context.usecase_id in usecases
    ]
    if not contexts:
        logger.info(f"{run.path.name}: no review of a usecase with a verification block")
        return 0

    for usecase_id in sorted({context.usecase_id for context in contexts}):
        usecase = usecases[usecase_id]
        references = _references(usecase, run.path)
        for context in contexts:
            if context.usecase_id != usecase_id:
                continue
            reviewer_run_id = context.record.identity.reviewer_run_id
            path = usecase_directory(run.path, usecase_id) / f"{reviewer_run_id}.json"
            if path.is_file():
                continue
            verification = Verification(
                usecase=usecase_id,
                ablation=context.ablation_id,
                reviewer_run_id=reviewer_run_id,
            )
            for suggestion in _measured_suggestions(context):
                label = str(suggestion.index)
                verification.suggestions[label] = measure_suggestion(
                    usecase,
                    label,
                    suggestion.code,
                    references,
                )
            write_model(verification, path)
            logger.info(
                f"{usecase_id} {context.ablation_id} {reviewer_run_id}: "
                f"{len(verification.suggestions)} suggestion(s) verified"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
