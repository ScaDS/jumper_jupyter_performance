"""Where things live, resolved from this file rather than from the cwd.

An experiment is run from wherever the caller happens to be standing, and
Hydra deliberately does not change that, so every path in the harness is
anchored here instead.
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parent

CONFIGS_DIR = EXPERIMENT_ROOT / "configs"
USECASES_DIR = EXPERIMENT_ROOT / "usecases"
STRATEGIES_DIR = EXPERIMENT_ROOT / "strategies"
RESULTS_DIR = EXPERIMENT_ROOT / "results"

# The file the AI reviewer is pointed at through JUMPER_AI_STRATEGIES_PATH.
# Generated from configs/ablation/*.yaml; never edited by hand.
STRATEGIES_FILE = STRATEGIES_DIR / "strategies.yaml"

# Hydra's config_path is resolved relative to the module holding @hydra.main,
# and every entry point sits one level down in cli/.
CLI_CONFIG_PATH = "../../configs"


def register_resolvers() -> None:
    """Teach OmegaConf where the experiment lives.

    ``${experiment_root:}`` is what config.yaml uses to place results next to
    the configs that produced them, whatever directory the command was typed
    in. Registering twice is normal - the CLI entry points each do it - so the
    call is idempotent.
    """
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver(
        "experiment_root",
        lambda: str(EXPERIMENT_ROOT),
        replace=True,
    )
