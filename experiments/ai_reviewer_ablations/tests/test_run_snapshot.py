"""A finished run describes itself, and keeps describing itself.

Scoring reads the manifests the run recorded, not the ones on disk now, and
a resume refuses to mix a second experiment into the same directory.
"""

from __future__ import annotations

import json

import pytest

from jumper_ablations.cli.common import snapshot_usecases
from jumper_ablations.cli.run import _refuse_incompatible_resume
from jumper_ablations.config.schema import (
    ExperimentConfig,
    JudgeConfig,
    MetricsConfig,
    ProtocolConfig,
    ReportingConfig,
    SuiteConfig,
)
from jumper_ablations.runner.run_directory import RunDirectory

MANIFEST = {
    "id": "synthetic/loop",
    "title": "Pure-python accumulation loop",
    "payload_type": "cpu_bound_python_loop",
    "reference_facts": [
        {
            "id": "sequential_loop",
            "source": "code",
            "weight": 2,
            "fact": "The work is a sequential loop.",
        }
    ],
}

SNAPSHOT = {
    "run_id": "r1",
    "suite": {"id": "smoke"},
    "protocol": {"generations_per_target": 5},
    "usecases": {"synthetic/loop": MANIFEST},
    "ablations": {"base": {"id": "base"}},
}


def _config(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        run_id="r1",
        results_root=str(tmp_path),
        workspace_root=str(tmp_path / "workspace"),
        usecases_root=str(tmp_path / "usecases"),
        suite=SuiteConfig(id="smoke", usecases=[], ablations=[]),
        protocol=ProtocolConfig(),
        judge=JudgeConfig(),
        metrics=MetricsConfig(),
        reporting=ReportingConfig(),
    )


def _run(tmp_path, meta: dict | None) -> RunDirectory:
    directory = RunDirectory.create(tmp_path / "r1")
    if meta is not None:
        directory.write_meta(meta)
    return directory


def test_scoring_reads_the_manifest_the_run_recorded(tmp_path):
    # The live directory is empty - the usecase has been renamed away, which
    # is what happened in practice. Scoring must still work, because the run
    # carries its own copy.
    run = _run(tmp_path, SNAPSHOT)

    usecases = snapshot_usecases(_config(tmp_path), run)

    assert set(usecases) == {"synthetic/loop"}
    facts = usecases["synthetic/loop"].manifest.reference_facts
    assert [fact.id for fact in facts] == ["sequential_loop"]


def test_a_renamed_usecase_cannot_change_a_finished_score(tmp_path):
    run = _run(tmp_path, SNAPSHOT)
    before = snapshot_usecases(_config(tmp_path), run)

    # Someone adds a fact to the manifest on disk after the run.
    live = tmp_path / "usecases" / "synthetic"
    live.mkdir(parents=True)
    (live / "loop.yaml").write_text(
        json.dumps({**MANIFEST, "reference_facts": []}), encoding="utf-8"
    )

    after = snapshot_usecases(_config(tmp_path), run)

    assert len(before["synthetic/loop"].manifest.reference_facts) == 1
    assert len(after["synthetic/loop"].manifest.reference_facts) == 1


def test_a_run_without_a_snapshot_says_so(tmp_path, caplog):
    run = _run(tmp_path, {"run_id": "r1"})
    (tmp_path / "usecases").mkdir()

    with caplog.at_level("WARNING"):
        snapshot_usecases(_config(tmp_path), run)

    assert "no usecase snapshot" in caplog.text


@pytest.mark.parametrize(
    "field, value",
    [
        ("suite", {"id": "robotics"}),
        ("protocol", {"generations_per_target": 10}),
        ("usecases", {"minian/cell_77": MANIFEST}),
        ("ablations", {"base": {"id": "base"}, "no_perf": {"id": "no_perf"}}),
    ],
)
def test_a_changed_experiment_cannot_resume_into_the_same_run(
    tmp_path, field, value
):
    run = _run(tmp_path, SNAPSHOT)

    with pytest.raises(SystemExit, match=field):
        _refuse_incompatible_resume(SNAPSHOT, {**SNAPSHOT, field: value}, run)


def test_the_same_experiment_resumes(tmp_path):
    run = _run(tmp_path, SNAPSHOT)

    # A different machine is a normal way to continue a sweep, not a reason
    # to refuse one.
    _refuse_incompatible_resume(
        SNAPSHOT, {**SNAPSHOT, "machine": {"node": "elsewhere"}}, run
    )


def test_resuming_records_itself_without_touching_the_definition(tmp_path):
    run = _run(tmp_path, SNAPSHOT)

    run.record_invocation({"node": "second-node"}, "00-of-02")
    run.record_invocation({"node": "third-node"}, "01-of-02")

    assert run.meta()["suite"] == SNAPSHOT["suite"]
    assert run.meta()["usecases"] == SNAPSHOT["usecases"]
    assert {entry["node"] for entry in run.invocation_entries()} == {
        "second-node",
        "third-node",
    }


def test_the_definition_is_claimed_by_exactly_one_invocation(tmp_path):
    # Shards start together, so all of them find the directory empty. The
    # exclusive create decides; everyone else is handed what stands and
    # checks itself against that rather than against its own copy.
    run = RunDirectory.create(tmp_path / "r1")

    first = run.claim_meta(SNAPSHOT)
    second = run.claim_meta({**SNAPSHOT, "machine": {"node": "other"}})

    assert first == SNAPSHOT
    assert second == SNAPSHOT
    assert run.meta() == SNAPSHOT


def test_the_strategies_snapshot_is_written_once(tmp_path):
    run = _run(tmp_path, SNAPSHOT)
    source = tmp_path / "strategies.yaml"
    source.write_text("strategies: [{id: base}]\n", encoding="utf-8")

    run.snapshot_strategies(source)
    run.snapshot_strategies(source)

    assert "changed" not in run.strategies_snapshot.read_text()


def test_presets_edited_mid_sweep_cannot_join_the_same_run(tmp_path):
    # Half the records would have been produced under definitions the other
    # half never saw, and nothing in either half would say so.
    run = _run(tmp_path, SNAPSHOT)
    source = tmp_path / "strategies.yaml"
    source.write_text("strategies: [{id: base}]\n", encoding="utf-8")
    run.snapshot_strategies(source)

    source.write_text("strategies: [{id: base, changed: true}]\n", "utf-8")

    with pytest.raises(SystemExit, match="different strategy"):
        run.snapshot_strategies(source)


def test_a_reader_never_sees_half_a_file(tmp_path):
    # This is the failure that killed two shards thirteen seconds into a
    # sweep: every shard regenerates the shared strategies file on startup
    # and then checks the run's snapshot against it, so a plain write leaves
    # a window in which the file is a prefix of itself and the check fails.
    import threading

    from jumper_ablations.files import write_atomically

    target = tmp_path / "shared.yaml"
    payload = "strategies:\n" + "".join(
        f"  - id: preset_{index}\n" for index in range(400)
    )
    write_atomically(target, payload)

    torn = []
    stop = threading.Event()

    def write() -> None:
        for _ in range(40):
            write_atomically(target, payload)

    def read() -> None:
        while not stop.is_set():
            try:
                seen = target.read_text(encoding="utf-8")
            except FileNotFoundError:
                torn.append("missing")
                continue
            if seen != payload:
                torn.append(f"{len(seen)} of {len(payload)} bytes")

    writers = [threading.Thread(target=write) for _ in range(4)]
    readers = [threading.Thread(target=read) for _ in range(2)]
    for thread in readers + writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    for thread in readers:
        thread.join()

    assert torn == []
