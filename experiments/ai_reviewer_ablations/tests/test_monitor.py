"""The monitor must describe any run directory, and never refuse to draw.

A monitor is looked at when something has gone wrong, which is exactly when
the run directory is least likely to be well formed: half-written files, a
sweep from an older harness, a definition that never got written because the
job died in its first second. A stack trace instead of a page at that moment
is worse than useless, so these tests feed the reader the shapes it will
actually meet and assert only that it answers.

They also fix the contract in monitor/PROTOCOL.md: the harness and the
monitor share a directory layout and nothing else, neither imports the other,
and either may gain fields without the other being touched.
"""

from __future__ import annotations

import json

import pytest

from monitor import model

DEFINITION = {
    "schema_version": 1,
    "run_id": "r1",
    "suite": {"id": "demo", "usecases": ["a/one"], "ablations": ["base"]},
    "protocol": {
        "generations_per_target": 2,
        "repetitions": 1,
        "benchmark": {"enabled": True, "mode": "separate"},
    },
    "usecases": {
        "a/one": {"id": "a/one", "benchmark": {"replay_mode": "full"}}
    },
    "ablations": {
        "base": {
            "id": "base",
            "family": "context",
            "effect": {
                "context": {source: True for source in model.SOURCE_IDS}
            },
        }
    },
    "config": {"shard": {"index": 0, "count": 1}},
}


def _run(tmp_path, meta=DEFINITION, name="r1"):
    run = tmp_path / name
    (run / "records").mkdir(parents=True)
    if meta is not None:
        (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run


def _record(run, usecase="a/one", ablation="base", generation=1, **extra):
    identity = {
        "record_id": f"{model.slug(usecase)}__{ablation}__r00"
        f"__g{generation:02d}__review__none",
        "usecase": usecase,
        "ablation": ablation,
        "repetition": 0,
        "generation": generation,
        "phase": "review",
    }
    body = {
        "identity": identity,
        "inputs": {},
        "outputs": {"analysis": "x", "suggestions": [{}, {}]},
        "cost": {"llm_latency_s": 1.0, "total_tokens": 10},
        "environment": {"warnings": [], "degraded": False},
    }
    body.update(extra)
    path = run / "records" / f"{identity['record_id']}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# -- the happy shape ----------------------------------------------------


def test_a_run_is_described_by_its_own_definition(tmp_path):
    run = _run(tmp_path)
    _record(run, generation=1)

    snapshot = model.snapshot(run)

    assert snapshot["run"]["run_id"] == "r1"
    assert snapshot["totals"]["passes"] == 1
    # Two generations, each owing a review and one measurement.
    assert snapshot["totals"]["records_expected"] == 4
    assert snapshot["totals"]["records"] == 1
    assert snapshot["grid"][0]["state"] == model.STATE_RUNNING


def test_progress_is_counted_from_records_not_from_the_index(tmp_path):
    # The index line lands when a pass ends, which under a full replay is
    # hours away. Without the records a running sweep would look idle.
    run = _run(tmp_path)
    for generation in (1, 2):
        _record(run, generation=generation)

    entry = model.snapshot(run)["grid"][0]

    assert entry["state"] == model.STATE_RUNNING
    assert entry["generations_seen"] == [1, 2]


def test_a_finished_pass_takes_its_verdict_from_the_index(tmp_path):
    run = _run(tmp_path)
    _record(run)
    (run / "passes-00-of-01.jsonl").write_text(
        json.dumps(
            {
                "usecase": "a/one",
                "ablation": "base",
                "repetition": 0,
                "status": "incomplete",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert model.snapshot(run)["grid"][0]["state"] == "incomplete"


def test_a_shard_that_ended_without_writing_is_lost_not_pending(tmp_path):
    # The one distinction the directory cannot make on its own, and the
    # reason the job id is recorded at all.
    run = _run(tmp_path)
    (run / "invocations").mkdir()
    (run / "invocations" / "00-of-01-x.json").write_text(
        json.dumps({"shard": "00-of-01", "slurm_job_id": "42"}),
        encoding="utf-8",
    )

    lost = model.snapshot(run, {"42": {"state": "failed"}})
    pending = model.snapshot(run, {"42": {"state": "running"}})

    assert lost["grid"][0]["state"] == model.STATE_LOST
    assert pending["grid"][0]["state"] == model.STATE_PENDING


# -- the shapes that must not crash it ----------------------------------


def test_an_empty_directory_is_described_rather_than_refused(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()

    snapshot = model.snapshot(run)

    assert snapshot["totals"]["passes"] == 0
    assert any("meta.json" in problem for problem in snapshot["problems"])


def test_a_missing_directory_is_described_rather_than_refused(tmp_path):
    snapshot = model.snapshot(tmp_path / "never-existed")

    assert snapshot["problems"]
    assert snapshot["totals"]["passes"] == 0


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"suite": None, "protocol": None},
        {"suite": {"usecases": None, "ablations": None}},
        {"protocol": {"repetitions": "many", "generations_per_target": None}},
        {"ablations": {"base": None}},
        {"ablations": "not a mapping"},
        {"usecases": [1, 2, 3]},
        {"config": {"shard": {"count": "four"}}},
        {
            "suite": {"usecases": ["a/one"], "ablations": ["base"]},
            "protocol": {"benchmark": {"extra_replay_modes": "fork"}},
        },
    ],
)
def test_any_definition_can_be_drawn(tmp_path, meta):
    # The harness is free to grow, and a field the monitor guessed the type
    # of must not be able to take the page down with it.
    run = _run(tmp_path, meta)

    snapshot = model.snapshot(run)

    assert isinstance(snapshot["grid"], list)
    assert json.dumps(snapshot)


def test_a_half_written_file_is_skipped_and_counted(tmp_path):
    run = _run(tmp_path)
    _record(run)
    (run / "records" / "torn.json").write_text(
        '{"identity":', encoding="utf-8"
    )
    (run / "passes-00-of-01.jsonl").write_text(
        '{"usecase": "a/one", "status": "ok"}\n{"usecase": "a/tw',
        encoding="utf-8",
    )

    snapshot = model.snapshot(run)

    assert snapshot["records"]["total"] == 1
    assert any("could not be parsed" in one for one in snapshot["problems"])


def test_a_record_with_an_unexpected_body_still_counts(tmp_path):
    run = _run(tmp_path)
    path = _record(run)
    path.write_text(
        json.dumps({"identity": {"record_id": "x", "generation": "two"}}),
        encoding="utf-8",
    )

    snapshot = model.snapshot(run)

    assert snapshot["records"]["total"] == 1
    assert snapshot["records"]["rows"][0]["generation"] == 0


def test_a_newer_layout_is_shown_with_a_warning(tmp_path):
    run = _run(tmp_path, {**DEFINITION, "schema_version": 99})

    snapshot = model.snapshot(run)

    assert snapshot["schema"]["written"] == 99
    assert any("newer harness" in one for one in snapshot["problems"])
    assert snapshot["grid"], "a newer run is still drawn"


def test_fields_the_monitor_does_not_know_are_ignored(tmp_path):
    # The forward half of the contract: the harness may add anything.
    run = _run(
        tmp_path,
        {**DEFINITION, "a_field_from_the_future": {"nested": [1, 2, 3]}},
    )

    assert model.snapshot(run)["totals"]["passes"] == 1


# -- several runs -------------------------------------------------------


def test_runs_are_listed_newest_first(tmp_path):
    _run(tmp_path, DEFINITION, "2026-01-01")
    _run(tmp_path, DEFINITION, "2026-06-01")

    names = [one["name"] for one in model.list_runs(tmp_path)]

    assert names == ["2026-06-01", "2026-01-01"]


def test_runs_asking_the_same_question_may_be_put_in_one_table(tmp_path):
    for name in ("first", "second"):
        run = _run(tmp_path, DEFINITION, name)
        (run / "summary.csv").write_text(
            "usecase,ablation,metric,reported_value,estimate,paired_delta,"
            "paired_delta_ci_low,paired_delta_ci_high,n,gaps\n"
            "a/one,base,best_speedup,mean,1.5,0.1,0.05,0.2,5,0\n",
            encoding="utf-8",
        )

    merged = model.aggregate(tmp_path, ["first", "second"])

    assert merged["compatibility"]["comparable"] is True
    assert len(merged["rows"]) == 1
    assert merged["rows"][0]["first"]["estimate"] == "1.5"
    assert merged["rows"][0]["second"]["estimate"] == "1.5"


def test_runs_asking_different_questions_are_not_pooled_silently(tmp_path):
    _run(tmp_path, DEFINITION, "first")
    _run(
        tmp_path,
        {**DEFINITION, "protocol": {"generations_per_target": 10}},
        "second",
    )

    merged = model.aggregate(tmp_path, ["first", "second"])

    assert merged["compatibility"]["comparable"] is False
    assert "protocol" in merged["compatibility"]["differs"]


def test_aggregating_a_run_without_metrics_yields_no_rows(tmp_path):
    _run(tmp_path, DEFINITION, "first")

    merged = model.aggregate(tmp_path, ["first", "absent"])

    assert merged["rows"] == []
    assert merged["runs"] == ["first", "absent"]


def _two_shards() -> dict:
    """A definition with a pass for each of two shards."""
    return {
        **DEFINITION,
        "protocol": {**DEFINITION["protocol"], "repetitions": 2},
        "config": {"shard": {"count": 2}},
    }


def _invocation(run, shard, job, stamp):
    (run / "invocations").mkdir(exist_ok=True)
    (run / "invocations" / f"{shard}-{stamp}.json").write_text(
        json.dumps(
            {"shard": shard, "slurm_job_id": job, "recorded_at": stamp}
        ),
        encoding="utf-8",
    )


def test_a_queued_shard_is_not_called_lost_after_a_resubmission(tmp_path):
    # A job records itself when it starts, so a shard queued right now still
    # shows the job of the attempt before - which has ended. Reading that as
    # "this shard died" turns a waiting pass red.
    run = _run(tmp_path, _two_shards())
    _invocation(run, "00-of-02", "111", "2026-09-30T20:38:45")
    _invocation(run, "01-of-02", "222", "2026-09-30T20:38:45")
    _invocation(run, "00-of-02", "333", "2026-09-30T22:54:27")

    snapshot = model.snapshot(
        run, {"333": {"state": "running"}, "222": {"state": "failed"}}
    )
    states = {entry["shard"]: entry["state"] for entry in snapshot["grid"]}

    # Shard 1 has not joined the newer attempt yet; its silence is not a
    # verdict, and the ended job belongs to the attempt before.
    assert states.get(1) == model.STATE_PENDING


def test_a_shard_that_died_in_the_current_attempt_is_still_lost(tmp_path):
    run = _run(tmp_path, _two_shards())
    _invocation(run, "00-of-02", "111", "2026-09-30T22:54:27")
    _invocation(run, "01-of-02", "222", "2026-09-30T22:54:27")

    snapshot = model.snapshot(
        run, {"111": {"state": "running"}, "222": {"state": "failed"}}
    )
    states = {entry["shard"]: entry["state"] for entry in snapshot["grid"]}

    assert states.get(1) == model.STATE_LOST


# -- averaging over usecases --------------------------------------------


def _across(values, combine="arithmetic", metric="m", reported="v"):
    """One summary row per usecase, for the same reported value."""
    return [
        {
            "usecase": usecase,
            "ablation": "base",
            "metric": metric,
            "reported_value": reported,
            "metric_name": "Name",
            "value_kind": "result",
            "combine": combine,
            "category": "suggestions",
            "evaluation_method": "deterministic",
            "estimate": str(value),
            "paired_delta": "",
        }
        for usecase, value in values.items()
    ]


def test_a_ratio_is_averaged_geometrically(tmp_path):
    # Twice as fast on one payload and twice as slow on another is no change
    # overall; the arithmetic mean would call it a 25% win.
    combined = model.per_usecase(
        _across({"a": 2.0, "b": 0.5}, combine="geometric")
    )["rows"]

    assert combined[0]["estimate"] == pytest.approx(1.0)


def test_everything_else_is_averaged_arithmetically(tmp_path):
    combined = model.per_usecase(_across({"a": 2.0, "b": 1.0}))["rows"]

    assert combined[0]["estimate"] == pytest.approx(1.5)


def test_each_usecase_weighs_the_same(tmp_path):
    # The whole point of this mode: a payload measured many times does not
    # outvote one measured twice, because the rows are already per usecase.
    rows = _across({"a": 2.0, "b": 1.0, "c": 1.0})

    combined = model.per_usecase(rows)["rows"][0]

    assert combined["estimate"] == pytest.approx(4.0 / 3.0)
    assert combined["k"] == 3


def test_only_the_selected_usecases_are_averaged(tmp_path):
    rows = _across({"a": 2.0, "b": 1.0, "c": 10.0})

    combined = model.per_usecase(rows, usecases=["a", "b"])["rows"][0]

    assert combined["estimate"] == pytest.approx(1.5)
    assert combined["usecases"] == ["a", "b"]


def test_two_usecases_get_a_value_but_no_interval(tmp_path):
    # The mean of two numbers is a mean. A bootstrap over two numbers has
    # three distinct samples, so its percentiles restate the two values.
    combined = model.per_usecase(_across({"a": 2.0, "b": 1.0}))["rows"][0]

    assert combined["estimate"] == pytest.approx(1.5)
    assert combined["k"] == 2
    assert combined["ci_low"] is None and combined["ci_high"] is None


def test_three_usecases_get_an_interval(tmp_path):
    combined = model.per_usecase(_across({"a": 1.0, "b": 2.0, "c": 3.0}))[
        "rows"
    ][0]

    assert combined["ci_low"] is not None
    assert combined["ci_low"] <= combined["estimate"] <= combined["ci_high"]


def test_a_geometric_mean_refuses_a_non_positive_value(tmp_path):
    # log(0) does not exist, and dropping the usecase silently would change
    # which payloads the number is about.
    combined = model.per_usecase(
        _across({"a": 2.0, "b": 0.0}, combine="geometric")
    )["rows"][0]

    assert combined["estimate"] is None


def test_the_paired_delta_is_averaged_as_a_difference(tmp_path):
    # Even for a ratio: a mean of differences is what the rest of the report
    # calls paired_delta, and a geometric mean of them would be a ratio of
    # ratios instead.
    rows = _across({"a": 2.0, "b": 0.5}, combine="geometric")
    for row, delta in zip(rows, (1.0, -0.5)):
        row["paired_delta"] = str(delta)

    combined = model.per_usecase(rows)["rows"][0]

    assert combined["paired_delta"] == pytest.approx(0.25)


def test_a_value_no_usecase_measured_survives_as_a_gap(tmp_path):
    rows = _across({"a": 1.0})
    rows[0]["estimate"] = ""

    combined = model.per_usecase(rows)["rows"][0]

    assert combined["estimate"] is None
    assert combined["k"] == 0
    assert combined["metric_name"] == "Name", "the row still describes itself"
