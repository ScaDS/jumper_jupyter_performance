"""What has to hold for two runs to be joined into one table.

Joining runs is the one aggregation that can be wrong without looking wrong,
in two ways. The grain: a cell-scope metric has already reduced five
repetitions to one number, so concatenating those rows weights the runs by
nothing in particular. And the question: two runs that measured different
presets produce rows with identical keys, so a join succeeds and means
nothing. These tests pin the grain and the fingerprint that stop both.
"""

from __future__ import annotations

import csv
import json

from jumper_ablations.aggregate.runs_index import (
    INDEX_NAME,
    build,
    describe,
    fingerprint_of,
    groups,
    write,
)
from jumper_ablations.aggregate.units import unit_values
from jumper_ablations.metrics.context import CellContext, unit_key_of
from jumper_ablations.metrics.registry import all_metrics

META = {
    "schema_version": 1,
    "run_id": "first",
    "suite": {"id": "demo", "usecases": ["a/one"], "ablations": ["base"]},
    "protocol": {
        "generations_per_target": 2,
        "repetitions": 1,
        "benchmark": {"enabled": True, "mode": "separate"},
    },
    "usecases": {"a/one": {"id": "a/one"}},
    "ablations": {"base": {"id": "base", "family": "context"}},
    "config": {"shard": {"index": 0, "count": 1}},
    "machine": {"recorded_at": "2026-10-08T12:00:00"},
}


def _run(root, name, meta=META, records=0):
    run = root / name
    (run / "records").mkdir(parents=True)
    for index in range(records):
        (run / "records" / f"r{index}.json").write_text("{}", encoding="utf-8")
    if meta is not None:
        payload = {**meta, "run_id": name}
        (run / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    return run


# -- the unit key -------------------------------------------------------


def test_the_unit_is_the_pass_and_the_generation_within_it():
    # Keying on the generation alone made a second repetition overwrite the
    # first, which silently halved every paired comparison.
    assert unit_key_of("a-one__base__r02__g03__review__none") == (2, 3)
    assert unit_key_of("a-one__base__g03") == (0, 3)
    assert unit_key_of("a-one__base") is None


def test_a_cell_restricted_to_a_unit_keeps_only_that_unit():
    class Identity:
        def __init__(self, key):
            self.unit_key = key

    class Record:
        def __init__(self, key):
            self.identity = Identity(key)

    cell = CellContext(
        usecase_id="a/one",
        ablation_id="base",
        records=(Record((0, 1)), Record((0, 2)), Record((1, 1))),
    )

    assert len(cell.restricted_to([(0, 1)]).records) == 1
    # A key listed twice is what a bootstrap resample is.
    assert len(cell.restricted_to([(0, 1), (0, 1)]).records) == 2
    assert len(cell.restricted_to([]).records) == 0


# -- the per-unit table -------------------------------------------------


def test_run_scope_values_are_carried_over_at_unit_grain():
    metrics = all_metrics()
    metric = metrics["precision"]
    rows = metric.rows(
        _Context("a/one", "base", "a-one__base__r00__g01__review__none"),
        {"total_claims": 10},
    )

    frame = unit_values(rows, {metric.id: metric}, [], run_id="first")

    assert set(frame["run_id"]) == {"first"}
    assert set(frame["repetition"]) == {0}
    assert set(frame["generation"]) == {1}
    claims = frame[frame["reported_value"] == "total_claims"]
    assert list(claims["value"]) == [10]


def test_a_row_without_a_unit_in_its_id_is_left_out():
    metrics = all_metrics()
    metric = metrics["precision"]
    rows = metric.rows(_Context("a/one", "base", "a-one__base"), {})

    assert unit_values(rows, {metric.id: metric}, [], run_id="first").empty


class _Context:
    """The least a metric needs to produce rows."""

    def __init__(self, usecase_id, ablation_id, unit_id):
        self.usecase_id = usecase_id
        self.ablation_id = ablation_id
        self.unit_id = unit_id
        self.sample_size = 1


# -- the fingerprint ----------------------------------------------------


def test_the_fingerprint_ignores_key_order():
    reordered = {
        "usecases": META["usecases"],
        "protocol": META["protocol"],
        "ablations": META["ablations"],
    }

    assert fingerprint_of(META) == fingerprint_of(reordered)


def test_the_fingerprint_ignores_everything_but_the_question():
    # When and where a run executed cannot make it incomparable with itself.
    elsewhere = {
        **META,
        "run_id": "second",
        "machine": {"node": "other", "recorded_at": "2027-01-01T00:00:00"},
        "config": {"shard": {"count": 64}},
    }

    assert fingerprint_of(META) == fingerprint_of(elsewhere)


def test_a_different_protocol_is_a_different_question():
    other = {**META, "protocol": {"generations_per_target": 10}}

    assert fingerprint_of(META) != fingerprint_of(other)


def test_a_different_preset_definition_is_a_different_question():
    # The definitions, not the names: two runs can both call a preset "base"
    # and mean different things.
    other = {
        **META,
        "ablations": {"base": {"id": "base", "effect": {"code": False}}},
    }

    assert fingerprint_of(META) != fingerprint_of(other)


# -- the index ----------------------------------------------------------


def test_runs_asking_the_same_question_share_a_fingerprint(tmp_path):
    _run(tmp_path, "first", records=2)
    _run(tmp_path, "second", records=3)
    _run(tmp_path, "other", {**META, "protocol": {"repetitions": 9}})

    pooled = groups(build(tmp_path))

    assert sorted(len(ids) for ids in pooled.values()) == [1, 2]
    together = next(ids for ids in pooled.values() if len(ids) == 2)
    assert sorted(together) == ["first", "second"]


def test_a_run_without_a_definition_still_gets_a_row(tmp_path):
    _run(tmp_path, "headless", meta=None, records=4)

    row = describe(tmp_path / "headless")

    assert row["run_id"] == "headless"
    assert row["records"] == 4
    assert row["fingerprint"], "a run with no meta still hashes to something"


def test_an_unreadable_definition_does_not_stop_the_index(tmp_path):
    run = _run(tmp_path, "torn", records=1)
    (run / "meta.json").write_text('{"run_id":', encoding="utf-8")

    rows = build(tmp_path)

    assert [row["run_id"] for row in rows] == ["torn"]


def test_hydra_output_is_not_mistaken_for_a_run(tmp_path):
    _run(tmp_path, "first")
    (tmp_path / ".hydra" / "20261008_120000").mkdir(parents=True)

    assert [row["run_id"] for row in build(tmp_path)] == ["first"]


def test_the_index_is_written_whole(tmp_path):
    _run(tmp_path, "first", records=1)

    path = write(tmp_path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    assert path.name == INDEX_NAME
    assert rows[0]["run_id"] == "first"
    assert rows[0]["records"] == "1"
    assert rows[0]["has_metrics"] == "False"
