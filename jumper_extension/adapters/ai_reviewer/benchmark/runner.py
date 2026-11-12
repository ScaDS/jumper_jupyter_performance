"""Replay one cell, however the chosen strategy gets it there, and read back what it cost.

Rebuilding the state a cell needs is the strategy's job (see ``replay``); this
module owns the half that must stay identical no matter which strategy ran:
results come back as a session export, which is then read through the very same
reporter the live review uses, so a variant's metrics are comparable with the
baseline's by construction rather than by convention.
"""
from __future__ import annotations

import contextlib
import logging
import os
import statistics
import tempfile

import pandas as pd

from jumper_extension.adapters.ai_reviewer.benchmark import fingerprint
from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK, RunOutcome
from jumper_extension.adapters.ai_reviewer.benchmark.replay import (
    FULL,
    FullReplayStrategy,
    ReplayContext,
    StrategyChanged,
    resolve_strategy,
)
from jumper_extension.adapters.ai_reviewer.context.collector import ContextCollector
from jumper_extension.adapters.ai_reviewer.language import LanguageAdapter, get_adapter
from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.adapters.reporter import build_performance_reporter
from jumper_extension.adapters.session import SessionImporter
from jumper_extension.monitor.common import OfflinePerformanceMonitor

logger = logging.getLogger("extension")


class BenchmarkRunner:
    """Drives a replay strategy and turns its session exports into numbers."""

    def __init__(
        self,
        prefix_cells: list[dict],
        interval: float,
        level: str = "process",
        work_dir: str | None = None,
        adapter: LanguageAdapter | None = None,
        replay_mode: str = FULL,
        keep_perf_series: bool = False,
    ):
        self.prefix_cells = prefix_cells
        self.interval = interval
        self.level = level
        # Whether each replay's perf samples are kept beside the summary drawn
        # from them, rather than dropped with the session export.
        self.keep_perf_series = keep_perf_series
        self._work_dir = work_dir or tempfile.mkdtemp(prefix="jumper-benchmark-")
        # The target cell's language decides how a replay is built and launched;
        # defaults to Python so direct constructors keep their old behaviour.
        self.adapter = adapter or get_adapter("python")
        self.strategy = resolve_strategy(
            replay_mode,
            ReplayContext(
                prefix_cells=self.prefix_cells,
                interval=self.interval,
                work_dir=self._work_dir,
                adapter=self.adapter,
            ),
        )
        self._prepared = False

    @property
    def work_dir(self) -> str:
        return self._work_dir

    def run_once(self, code: str, tag: str, timeout: float | None = None) -> RunOutcome:
        """Replay the prefix state plus *code* once, timing the last cell."""
        self._ensure_prepared()
        result = self.strategy.replay(code, tag, timeout)
        if result.strategy_broken:
            # Not this code's fault, so it must not be reported as its failure.
            # Nor can this one measurement simply be retried on the full replay
            # and set beside the ones taken before it: a benchmark's output is a
            # ratio, and the two modes are two instruments. Swap, then unwind and
            # let the whole run start again on one of them.
            self.fall_back(result.error)
            raise StrategyChanged(result.error)
        if not result.ok:
            return RunOutcome(
                status=result.status,
                error=result.error,
                prefix_cell=result.prefix_cell,
            )
        outcome = self._read_outcome(
            result.session_path,
            result.fingerprint_path,
            self.strategy.target_cell_index,
        )
        outcome.wall_s = result.wall_s
        return outcome

    def cross_check_baseline(self, code: str, tag: str = "cross_check") -> RunOutcome | None:
        """Measure *code* once through the full replay, whatever mode is active.

        A fast mode rebuilds state instead of replaying it, and nothing inside
        the benchmark can tell whether it rebuilt the right one: the baseline and
        every variant go through the same rebuild, so they agree with each other
        and the divergence check reports a match. The only thing that can catch
        it is a measurement taken the other way.

        Returns None when the active strategy already *is* the full replay -
        there is nothing to compare - and never disturbs it otherwise: the check
        runs on a throwaway strategy of its own, before the active one prepares,
        so a prefix is never resident twice. A reference that will not run comes
        back as a failed outcome, because a check that silently did not happen is
        indistinguishable from one that passed.
        """
        if isinstance(self.strategy, FullReplayStrategy):
            return None

        reference = FullReplayStrategy(self.strategy.context)
        try:
            reference.prepare()
            result = reference.replay(code, tag, None)
            if not result.ok:
                return RunOutcome(status=result.status, error=result.error)
            # Its own row, not the active strategy's: a full replay puts the cell
            # under test after the prefix it just ran, and reading row zero there
            # would time a prefix cell instead.
            outcome = self._read_outcome(
                result.session_path,
                result.fingerprint_path,
                reference.target_cell_index,
            )
            outcome.wall_s = result.wall_s
            return outcome
        finally:
            reference.close()

    def close(self):
        """Release whatever the strategy is holding. Safe to call twice."""
        self.strategy.close()

    def _ensure_prepared(self):
        """Set the strategy up on first use, falling back to full replay if it cannot.

        Preparing lazily matters: a benchmark whose timed run is turned off never
        reaches a replay, and should not pay for a zygote or a checkpoint it will
        not use. A strategy that cannot start is not an error - the full replay
        is always correct, so we say why and carry on with it.
        """
        if self._prepared:
            return
        self._prepared = True
        outcome = self.strategy.prepare()
        if not outcome.ok:
            self.fall_back(outcome.reason)

    def fall_back(self, reason: str):
        """Give up on the current strategy and finish on the full replay.

        Always possible, and always correct: the full replay needs no setup and
        serves every language, so there is no state in which this cannot be done.
        """
        if isinstance(self.strategy, FullReplayStrategy):
            return
        logger.warning(
            f"[JUmPER]: benchmark replay mode {self.strategy.name!r} is unusable "
            f"({reason}); falling back to the full replay."
        )
        self.strategy.close()
        self.strategy = FullReplayStrategy(self.strategy.context)
        self.strategy.prepare()

    def _read_outcome(
        self,
        session_path: str,
        fingerprint_path: str,
        target_index: int,
    ) -> RunOutcome:
        """Read the exported session through the live review's own analysis path.

        Duration and metrics come from different places on purpose. The hooks
        time every cell exactly, but the sampler only catches cells that outlive
        its interval - and a good optimization is precisely the one that stops
        being sampled. Missing metrics must not read as a failed run.
        """
        # Where the target landed is the strategy's business - a full replay puts
        # it after the prefix it just ran, while one that restores state instead
        # may place it anywhere - so the caller names the row rather than this
        # reading it off whichever strategy happens to be active.
        importer = SessionImporter(logger)
        work_dir, cleanup = importer._prepare_work_directory(session_path)
        try:
            manifest = importer._load_manifest(work_dir)
            perf_dfs = importer._load_performance_data(work_dir)
            cell_history = CellHistory()
            cell_history.data = pd.read_csv(os.path.join(work_dir, "cell_history.csv"))

            duration = _target_duration(cell_history.data, target_index)
            if duration is None:
                return RunOutcome(
                    status=FAILED,
                    error="The run left no history entry for the cell under test.",
                )

            monitor = OfflinePerformanceMonitor(
                manifest=manifest,
                perf_dfs=perf_dfs,
                source=session_path,
            )
            reporter = build_performance_reporter(cell_history, display_disabled=True)
            reporter.attach(monitor)
            with _quiet():
                context = reporter.build_context((target_index, target_index), self.level)
        finally:
            if cleanup:
                _remove_tree(work_dir)

        metrics = (
            ContextCollector._summarize_perfdata(context["perfdata"])["overall"]
            if context is not None
            else {}
        )
        perf_series = (
            context["perfdata"].to_dict(orient="list")
            if self.keep_perf_series and context is not None
            else {}
        )
        prints = fingerprint.load(fingerprint_path) if os.path.exists(fingerprint_path) else {}
        return RunOutcome(
            status=OK,
            duration_s=duration,
            metrics=metrics,
            fingerprints=prints,
            perf_series=perf_series,
        )


def _target_duration(cells: pd.DataFrame, target_index: int) -> float | None:
    """Exact wall-clock of the cell under test, straight from the hooks."""
    if cells.empty or "cell_index" not in cells.columns:
        return None
    rows = cells[cells["cell_index"] == target_index]
    if rows.empty:
        return None
    # Microseconds, not milliseconds: a good optimization can land well under
    # 1ms, and coarser rounding would collapse it to zero and lose the speedup.
    return round(float(rows.iloc[-1]["duration"]), 6)


@contextlib.contextmanager
def _quiet():
    """Mute the reporter's "no performance data" warning for one call.

    A cell too fast to sample is an expected outcome here - we report it as
    missing metrics rather than letting it print once per run.
    """
    extension_logger = logging.getLogger("extension")
    previous = extension_logger.level
    extension_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        extension_logger.setLevel(previous)


def median_of(outcomes: list[RunOutcome]) -> tuple[float, dict]:
    """Median duration and per-metric medians, after dropping the warm-up run.

    The first run pays for imports, JIT and cold caches, which says nothing
    about the code itself; the median of the rest resists the odd GC pause.
    """
    timed = outcomes[1:] if len(outcomes) > 1 else outcomes
    duration = statistics.median(outcome.duration_s for outcome in timed)

    metrics: dict = {}
    for name in timed[0].metrics:
        for statistic in ("mean", "max"):
            values = [
                outcome.metrics[name][statistic]
                for outcome in timed
                if name in outcome.metrics
            ]
            metrics.setdefault(name, {})[statistic] = round(statistics.median(values), 4)
    return round(duration, 6), metrics


def _remove_tree(path: str) -> None:
    import shutil

    try:
        shutil.rmtree(path)
    except Exception:
        pass
