from pathlib import Path

import pytest

from jumper_extension.adapters.ai_reviewer.strategy import (
    get_strategy,
    load_strategies,
    strategy_ids,
)
from jumper_extension.adapters.ai_reviewer.strategy import loader

_EXTERNAL = """
strategies:
  - id: probe_ablation
    name: Probe ablation
    description: Added from outside the package.
    effect:
      context: {timing: false, raw_perf: true}
      overrides: {pep8_multiline: false}
  - id: faster
    name: Replaced built-in
    effect:
      context: {packages: false}
"""


@pytest.fixture
def external_strategies(tmp_path: Path) -> Path:
    path = tmp_path / "strategies.yaml"
    path.write_text(_EXTERNAL)
    return path


def test_builtin_strategies_are_loaded():
    strategies = load_strategies()
    assert set(strategies) >= {"faster", "parallelization", "custom", "deep"}
    assert strategies["custom"].require_note is True
    assert strategies["deep"].overrides == {"packages": True, "raw_perf": True}


def test_external_file_adds_ids(monkeypatch, external_strategies):
    monkeypatch.setenv(loader._STRATEGIES_PATH_ENV, str(external_strategies))

    assert "probe_ablation" in strategy_ids()
    assert get_strategy("probe_ablation").overrides == {
        "timing": False,
        "raw_perf": True,
        "pep8_multiline": False,
    }


def test_external_file_replaces_a_builtin(monkeypatch, external_strategies):
    monkeypatch.setenv(loader._STRATEGIES_PATH_ENV, str(external_strategies))

    replaced = get_strategy("faster")
    assert replaced.name == "Replaced built-in"
    assert replaced.overrides == {"packages": False}


def test_missing_external_file_falls_back_to_builtins(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(loader._STRATEGIES_PATH_ENV, str(tmp_path / "absent.yaml"))

    with caplog.at_level("WARNING", logger="extension"):
        ids = strategy_ids()

    assert "faster" in ids
    assert "probe_ablation" not in ids
    assert "does not exist" in caplog.text


def test_config_value_is_used_when_no_environment_variable(
    monkeypatch,
    external_strategies,
):
    monkeypatch.delenv(loader._STRATEGIES_PATH_ENV, raising=False)
    config = loader.load_config()
    monkeypatch.setattr(
        config.ai.context,
        "strategies_path",
        str(external_strategies),
    )

    assert "probe_ablation" in strategy_ids()
