"""Splitting one run across several jobs.

The grid is independent kernel passes, so the cluster is spent by giving each
job a share. What must hold: the shares partition the grid, they write without
treading on each other, and a resume skips whatever any of them finished.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jumper_ablations.cli.run import _already_recorded
from jumper_ablations.config.schema import ShardConfig
from jumper_ablations.runner.run_directory import RunDirectory

GRID = [f"pass-{index:02d}" for index in range(10)]


def test_one_shard_takes_the_whole_grid():
    assert ShardConfig().slice_of(GRID) == GRID


def test_the_shares_partition_the_grid():
    shards = [ShardConfig(index=i, count=4) for i in range(4)]
    shares = [shard.slice_of(GRID) for shard in shards]

    assert sorted(sum(shares, [])) == sorted(GRID)
    for left in range(4):
        for right in range(left + 1, 4):
            assert not set(shares[left]) & set(shares[right])


def test_the_shares_are_striped_rather_than_blocked():
    # Passes differ in cost - a preset returning fewer suggestions benchmarks
    # faster - so contiguous blocks would leave one job running long after
    # the others finished.
    assert ShardConfig(index=0, count=3).slice_of(GRID) == [
        "pass-00",
        "pass-03",
        "pass-06",
        "pass-09",
    ]


def test_more_shards_than_passes_leaves_some_with_nothing():
    shares = [ShardConfig(index=i, count=16).slice_of(GRID) for i in range(16)]

    assert sorted(sum(shares, [])) == sorted(GRID)
    assert shares[12] == []


def test_a_shard_outside_its_count_is_refused():
    with pytest.raises(ValidationError):
        ShardConfig(index=4, count=4)


def test_each_shard_writes_its_own_index(tmp_path):
    # One writer per file: appending to a shared index from several nodes is
    # not atomic, and a torn line decides which passes a resume skips.
    run = RunDirectory.create(tmp_path / "run")

    run.append_pass({"usecase": "a/one", "status": "ok"}, "00-of-02")
    run.append_pass({"usecase": "a/two", "status": "ok"}, "01-of-02")

    assert run.pass_index("00-of-02") != run.pass_index("01-of-02")
    assert len(run.pass_entries()) == 2


def test_a_resume_skips_what_any_shard_finished(tmp_path):
    # The shards of one attempt need not be the shards of the next: a single
    # job with no sharding at all must still skip the finished passes.
    run = RunDirectory.create(tmp_path / "run")
    run.append_pass(
        {
            "usecase": "a/one",
            "ablation": "base",
            "repetition": 0,
            "status": "ok",
        },
        "00-of-03",
    )
    run.append_pass(
        {
            "usecase": "a/one",
            "ablation": "no_perf",
            "repetition": 0,
            "status": "ok",
        },
        "02-of-03",
    )

    assert _already_recorded(run) == {
        ("a/one", "base", 0),
        ("a/one", "no_perf", 0),
    }


def test_an_index_written_before_sharding_is_still_read(tmp_path):
    run = RunDirectory.create(tmp_path / "run")
    run.passes_index.write_text(
        '{"usecase": "a/one", "ablation": "base", '
        '"repetition": 0, "status": "ok"}\n',
        encoding="utf-8",
    )

    assert _already_recorded(run) == {("a/one", "base", 0)}


def test_the_label_names_the_files_it_writes():
    assert ShardConfig(index=2, count=12).label == "02-of-12"
