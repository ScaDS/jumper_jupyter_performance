from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from jumper_extension.adapters.ai_reviewer.strategy.models import Strategy
from jumper_extension.config.loader import load_config

logger = logging.getLogger("extension")

_STRATEGIES_PATH = Path(__file__).parent / "strategies.yaml"

# An external caller - an ablation experiment, a deployment with house rules -
# can add strategies without editing this package: the file named by this
# variable (or by ``ai.context.strategies_path``) is parsed exactly like the
# built-in one and merged over it. Everything downstream inherits that for
# free, including the ``--strategy`` argparse choices, which are built from
# ``strategy_ids()`` when the extension loads.
_STRATEGIES_PATH_ENV = "JUMPER_AI_STRATEGIES_PATH"


def _parse(path: Path) -> dict[str, Strategy]:
    """Parse one strategies file into ``{id: Strategy}``."""
    data = yaml.safe_load(path.read_text()) or {}
    strategies = {}
    for entry in data.get("strategies") or []:
        effect = entry.get("effect") or {}
        overrides = {
            **(effect.get("context") or {}),
            **(effect.get("overrides") or {}),
        }
        strategies[entry["id"]] = Strategy(
            id=entry["id"],
            name=entry["name"],
            description=entry.get("description", ""),
            overrides=overrides,
            require_note=bool(entry.get("require_note", False)),
        )
    return strategies


def _extra_path() -> Path | None:
    """The external strategies file, or None when none is configured.

    A path that does not exist is a misconfiguration worth saying out loud -
    silently falling back to the built-ins would let a whole experiment run
    against the wrong presets.
    """
    configured = (
        os.environ.get(_STRATEGIES_PATH_ENV)
        or load_config().ai.context.strategies_path
    )
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_file():
        logger.warning(
            f"[JUmPER]: strategies file '{path}' does not exist; "
            "using the built-in strategies only"
        )
        return None
    return path


def load_strategies(path: Path = _STRATEGIES_PATH) -> dict[str, Strategy]:
    """Parse ``strategies.yaml`` into ``{id: Strategy}``.

    Each strategy's ``effect.overrides`` (prompt items) and ``effect.context``
    (context sources) are merged into one flat ``id -> enabled`` map.

    External strategies are merged over the built-ins, so an external file can
    both add presets and replace one of the built-ins by reusing its id.
    """
    strategies = _parse(path)
    extra = _extra_path()
    if extra is not None:
        strategies.update(_parse(extra))
    return strategies


def strategy_ids(path: Path = _STRATEGIES_PATH) -> list[str]:
    return list(load_strategies(path))


def get_strategy(strategy_id: str, path: Path = _STRATEGIES_PATH) -> Strategy:
    return load_strategies(path)[strategy_id]
