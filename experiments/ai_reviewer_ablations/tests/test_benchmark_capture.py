"""What a benchmark leaves in the record, and what goes beside it."""

from __future__ import annotations

from jumper_extension.adapters.ai_reviewer.benchmark.models import (
    FAILED,
    OK,
    AttemptRecord,
    BenchmarkResult,
)

from jumper_ablations.config.schema import ProtocolConfig
from jumper_ablations.records.schema import BenchmarkRecord, RunIdentity, RunOutputs, RunRecord
from jumper_ablations.records.store import RecordStore, load_perf_series, load_records
from jumper_ablations.runner.cell_plan import bootstrap_cell
from jumper_ablations.runtime.session import _benchmark_record, _perf_series_from

_SERIES = [
    {"time": [0.0, 0.05], "cpu_util_avg": [10.0, 20.0]},
    {"time": [0.0, 0.05], "cpu_util_avg": [12.0, 22.0]},
]


def _verdict() -> BenchmarkResult:
    return BenchmarkResult(
        label="1",
        status=OK,
        attempts=2,
        duration_s=1.0,
        speedup=4.0,
        correctness="match",
        run_durations_s=[1.5, 1.0],
        run_wall_s=[2.5, 2.0],
        run_perf_series=_SERIES,
        fingerprints={"y": {"kind": "scalar", "value": 42.0}},
        attempt_history=[
            AttemptRecord(
                attempt=1,
                code="broken",
                status=FAILED,
                error="ValueError: boom",
            ),
            AttemptRecord(
                attempt=2,
                code="fixed",
                status=OK,
                duration_s=1.0,
                correctness="match",
            ),
        ],
    )


def test_the_record_keeps_every_replay_and_attempt_but_not_the_series():
    record = _benchmark_record(_verdict())

    assert record.run_durations_s == [1.5, 1.0]
    assert record.run_wall_s == [2.5, 2.0]
    assert record.fingerprints == {"y": {"kind": "scalar", "value": 42.0}}
    assert [(attempt.code, attempt.status) for attempt in record.attempt_history] == [
        ("broken", FAILED),
        ("fixed", OK),
    ]
    assert record.attempt_history[0].error == "ValueError: boom"
    assert "run_perf_series" not in record.model_dump()


def test_perf_series_go_to_a_file_of_their_own(tmp_path):
    state = {
        "benchmarks": {
            "baseline": BenchmarkResult(label="baseline", status=OK),
            "1": _verdict(),
        }
    }
    store = RecordStore(tmp_path)
    identity = RunIdentity(
        record_id="usecase__base__r00__g01__review__none",
        usecase="usecase",
        ablation="base",
    )
    store.write(
        RunRecord(
            identity=identity,
            outputs=RunOutputs(benchmarks={"1": _benchmark_record(_verdict())}),
        )
    )

    series = _perf_series_from(state)
    store.write_perf_series(identity.record_id, series)

    # Only verdicts that kept any are written.
    assert load_perf_series(tmp_path, identity.record_id) == {"1": _SERIES}
    (loaded,) = load_records(tmp_path)
    assert loaded.outputs.benchmarks["1"].run_durations_s == [1.5, 1.0]


def test_a_record_without_any_series_has_no_file(tmp_path):
    assert _perf_series_from({"benchmarks": {"1": BenchmarkResult(label="1", status=OK)}}) == {}
    assert load_perf_series(tmp_path, "missing") == {}


def test_a_record_written_before_these_fields_still_loads():
    record = BenchmarkRecord.model_validate({"label": "1", "status": "ok", "duration_s": 1.0})

    assert record.run_durations_s == []
    assert record.attempt_history == []


def test_the_protocol_keeps_perf_series_and_hands_that_to_the_kernel():
    assert ProtocolConfig().benchmark.keep_perf_series is True

    cell = bootstrap_cell(
        run_directory="/tmp/run",
        usecase="usecase",
        ablation="base",
        ablation_family="context",
        repetition=0,
        suite="suite",
        run_id="run",
        keep_perf_series=True,
    )

    assert "'keep_perf_series': True" in cell
