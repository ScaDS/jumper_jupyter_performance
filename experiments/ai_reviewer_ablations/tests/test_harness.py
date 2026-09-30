"""The wiring: how a review line is built, and what the presets say."""

from __future__ import annotations

import pytest
import yaml

from jumper_ablations.config.schema import ProtocolConfig
from jumper_ablations.runner.cell_plan import (
    begin_cell,
    review_line,
    resume_benchmark_line,
)
from jumper_ablations.strategies import (
    available_ablation_ids,
    build_strategies_file,
    compose_ablation,
)

NOTEBOOK_LINE = "%perfmonitor_ai_review --benchmark --replay-mode full"


def test_review_line_keeps_the_notebook_and_appends_the_preset():
    line = review_line(
        base_line=NOTEBOOK_LINE + " --level user",
        ablation_id="no_timing",
        replay_mode="fork",
        protocol=ProtocolConfig(),
        target_cell_index=7,
    )

    # The notebook's own flags survive; the managed ones are re-stated.
    # The level is managed: the protocol declares it, the run records it and
    # the report describes it, so a notebook asking for another one would
    # leave the config describing measurements it did not produce.
    assert "--level process" in line
    assert "--level user" not in line
    assert "--strategy no_timing" in line
    assert "--cells 7" in line
    assert "--replay-mode fork" in line
    assert line.count("--replay-mode") == 1
    assert line.count("--benchmark ") == 1


def test_review_line_drops_the_benchmark_for_the_two_command_shape():
    line = review_line(
        base_line=NOTEBOOK_LINE,
        ablation_id="base",
        replay_mode="full",
        protocol=ProtocolConfig(),
        target_cell_index=1,
        with_benchmark=False,
    )

    assert "--benchmark" not in line
    assert "--replay-mode" not in line
    assert line == (
        "%perfmonitor_ai_review --strategy base --level process --cells 1"
    )


def test_the_protocol_decides_the_monitoring_level():
    line = review_line(
        base_line=NOTEBOOK_LINE,
        ablation_id="base",
        replay_mode="full",
        protocol=ProtocolConfig(level="system"),
        with_benchmark=False,
    )

    assert "--level system" in line


def test_review_line_rejects_a_cell_that_is_not_a_review():
    with pytest.raises(ValueError):
        review_line(
            base_line="print('hello')",
            ablation_id="base",
            replay_mode="full",
            protocol=ProtocolConfig(),
        )


def test_resume_line_measures_stored_suggestions():
    line = resume_benchmark_line("abc123", "dill", ProtocolConfig())

    assert line.startswith(
        "%perfmonitor_ai_review --resume abc123 --benchmark"
    )
    assert "--replay-mode dill" in line
    assert "--strategy" not in line


def test_injected_cells_are_python_not_json():
    # json.dumps would write None as `null`, which is a NameError two cells
    # later rather than an error where the mistake was made.
    source = begin_cell(1, "review", "", reviewer_run_id=None)

    assert "null" not in source
    assert "None" in source
    compile(source, "<begin_cell>", "exec")


def test_every_ablation_composes_into_a_strategy(tmp_path):
    destination = build_strategies_file(
        destination=tmp_path / "strategies.yaml"
    )
    entries = yaml.safe_load(destination.read_text())["strategies"]

    assert {entry["id"] for entry in entries} == set(available_ablation_ids())
    for entry in entries:
        assert entry["name"]
        assert set(entry["effect"]) == {"context", "overrides"}


def test_a_delta_inherits_the_base_it_declares():
    base = compose_ablation("base")
    no_timing = compose_ablation("no_timing")

    assert base.flat_overrides()["timing"] is True
    assert no_timing.flat_overrides()["timing"] is False
    # Everything else is untouched: an ablation removes one thing.
    differing = {
        source
        for source, enabled in no_timing.flat_overrides().items()
        if base.flat_overrides().get(source) != enabled
    }
    assert differing == {"timing"}


def test_prompt_ablations_leave_every_context_source_on():
    prompt = compose_ablation("prompt_parallel_hint")

    assert prompt.family == "prompt"
    assert all(
        prompt.effect.context[source] for source in prompt.effect.context
    )
    assert prompt.effect.overrides["parallel_analysis_hint"] is True


def test_a_usecase_notebook_has_to_stand_on_its_own(tmp_path):
    # A notebook the harness would have to patch is not a usecase: it could
    # not be opened and run by hand, so what it does and what the experiment
    # measured would be two different things.
    import nbformat

    from jumper_ablations.cli.usecases import prepare
    from jumper_ablations.usecases.notebook import read_layout

    path = tmp_path / "bare.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# Bare"),
            nbformat.v4.new_code_cell("total = sum(range(10))"),
        ]
    )
    nbformat.write(notebook, str(path))

    with pytest.raises(ValueError, match="never loads the extension"):
        read_layout(path)

    assert prepare(path) == ["setup", "review"]

    layout = read_layout(path)
    assert layout.review_index == 3
    assert layout.payload_index == 2
    assert prepare(path) == []


def test_generations_are_paired_on_the_seed():
    # Two presets at the same generation must draw under the same seed: the
    # report's paired delta assumes they differ in context and nothing else.
    from jumper_ablations.config.schema import SamplingProtocol

    sampling = SamplingProtocol(temperature=0.7, seed_base=1000)

    assert sampling.seed_for(1) == 1001
    assert sampling.seed_for(2) == 1002
    assert sampling.as_applied(3) == {
        "temperature": 0.7,
        "top_p": None,
        "seed": 1003,
    }


def test_an_unset_seed_base_leaves_the_seed_alone():
    from jumper_ablations.config.schema import SamplingProtocol

    sampling = SamplingProtocol(seed_base=None)

    assert sampling.seed_for(4) is None
    assert sampling.as_applied(4)["seed"] is None


def test_sampling_reaches_the_kernel_as_python():
    source = begin_cell(
        2, "review", "", sampling={"temperature": 0.7, "seed": 1002}
    )

    assert "null" not in source
    assert "'seed': 1002" in source
    compile(source, "<begin_cell>", "exec")


def test_a_rerun_skips_the_passes_that_already_succeeded(tmp_path):
    # A sweep is hours of machine time; a kernel dying at pass 15 of 18 must
    # not cost the fourteen that worked.
    import json

    from jumper_ablations.cli.run import _already_recorded
    from jumper_ablations.runner.run_directory import RunDirectory

    run = RunDirectory.create(tmp_path / "run")
    for entry in (
        {
            "usecase": "a/one",
            "ablation": "base",
            "repetition": 0,
            "status": "ok",
        },
        {
            "usecase": "a/one",
            "ablation": "no_perf",
            "repetition": 0,
            "status": "ok",
        },
        # Failed passes are worth retrying, and wrote no records.
        {
            "usecase": "a/two",
            "ablation": "base",
            "repetition": 0,
            "status": "prefix_failed",
        },
    ):
        run.append_pass(entry)

    done = _already_recorded(run)

    assert done == {("a/one", "base", 0), ("a/one", "no_perf", 0)}
    assert ("a/two", "base", 0) not in done
    assert (
        json.loads(run.passes_index.read_text().splitlines()[0])["status"]
        == "ok"
    )


def test_a_usecase_can_pin_absolute_paths_for_its_data(tmp_path):
    # A replay runs in a fresh temporary directory, so a path the notebook
    # resolves against the working directory is not there when it is measured.
    from jumper_ablations.usecases.registry import UsecaseManifest

    manifest = UsecaseManifest(
        id="minian/cell_77",
        environment={
            "MINIAN_INTERMEDIATE": "${workspace}/minian_intermediate"
        },
    )

    resolved = manifest.resolved_environment(tmp_path / "minian")

    assert resolved["MINIAN_INTERMEDIATE"] == str(
        tmp_path / "minian" / "minian_intermediate"
    )
