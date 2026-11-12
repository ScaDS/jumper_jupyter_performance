"""What one real replay leaves behind once the runner has read it back."""

from __future__ import annotations

from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner

# Long enough for a 50 ms sampler to catch it several times.
_SAMPLED_CELL = "import time\ntime.sleep(0.5)\ny = 1"


def _run_once(tmp_path, keep_perf_series: bool):
    runner = BenchmarkRunner(
        prefix_cells=[],
        interval=0.05,
        work_dir=str(tmp_path),
        keep_perf_series=keep_perf_series,
    )
    try:
        return runner.run_once(_SAMPLED_CELL, tag="variant_0")
    finally:
        runner.close()


def test_perf_samples_of_the_timed_cell_are_kept_when_asked(tmp_path):
    outcome = _run_once(tmp_path, keep_perf_series=True)

    assert outcome.ok, outcome.error
    series = outcome.perf_series
    assert series["time"]
    # Column-aligned, like the raw_perf the live review sends the model.
    assert {len(values) for values in series.values()} == {len(series["time"])}
    assert set(outcome.metrics) <= set(series)


def test_perf_samples_are_dropped_by_default(tmp_path):
    outcome = _run_once(tmp_path, keep_perf_series=False)

    assert outcome.ok, outcome.error
    assert outcome.metrics
    assert outcome.perf_series == {}
