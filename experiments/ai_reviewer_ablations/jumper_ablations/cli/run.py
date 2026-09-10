"""Run the experiment: the real magic, in a real kernel, recorded.

The suite says which usecases are crossed with which ablations; the protocol
says how hard each cell of that grid is hit. Hydra composes both and writes
the composed config into the run directory, so a result and the setup that
produced it stay together.

    python -m jumper_ablations.cli.run suite=context_sources
    python -m jumper_ablations.cli.run suite=smoke protocol=pilot
    python -m jumper_ablations.cli.run suite=smoke protocol.generations_per_target=1
"""
from __future__ import annotations

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


@hydra.main(version_base=None, config_path=CLI_CONFIG_PATH, config_name="config")
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
    run_directory.snapshot_strategies()

    usecases = [
        get_usecase(one, config.usecases_root) for one in config.suite.usecases
    ]
    ablations = [compose_ablation(one) for one in config.suite.ablations]

    run_directory.write_meta(
        {
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
    )

    total = len(usecases) * len(ablations) * config.protocol.repetitions
    logger.info(
        f"suite '{config.suite.id}': {len(usecases)} usecase(s) x "
        f"{len(ablations)} ablation(s) x {config.protocol.repetitions} "
        f"repetition(s) = {total} kernel pass(es) into {run_directory.path}"
    )

    failures = 0
    for usecase in usecases:
        for ablation in ablations:
            for repetition in range(config.protocol.repetitions):
                outcome = run_pass(
                    config=config,
                    run_directory=run_directory,
                    usecase=usecase,
                    ablation=ablation,
                    repetition=repetition,
                )
                run_directory.append_pass(outcome.as_entry())
                level = logging.INFO if outcome.status == "ok" else logging.ERROR
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
