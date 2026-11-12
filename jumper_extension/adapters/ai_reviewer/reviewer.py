import logging
import os
import uuid
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from jumper_extension.adapters.reporter import PerformanceReporter
from jumper_extension.config.loader import load_config
from jumper_extension.core.messages import (
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
    ExtensionErrorCode,
    ExtensionInfoCode,
)
from jumper_extension.monitor.common import MonitorProtocol, UnavailablePerformanceMonitor

logger = logging.getLogger("extension")

_AI_EXTRAS_MISSING_REASON = (
    "optional dependencies (langgraph, langchain-openai) are not installed"
)


def _ai_extras_install_cmd() -> str:
    """Install command for the ``[ai]`` extras.

    Source checkouts must reinstall from their own dir (the wheel lacks the
    reviewer); we detect one via the project's ``pyproject.toml``. Quoted so
    ``[ai]`` survives shells that glob brackets (zsh).
    """
    root = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isfile(os.path.join(root, "pyproject.toml")):
            return f"pip install -e '{root}[ai]'"
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    return "pip install 'jumper-extension[ai]'"


@runtime_checkable
class AIReviewerProtocol(Protocol):
    """Structural protocol for the AI review adapter used by the service."""
    def attach(self, monitor: MonitorProtocol) -> None: ...

    def review(
        self,
        shell: Any,
        cell_range: Optional[Tuple[int, int]] = None,
        level: str = "process",
        strategy: str = "faster",
        note: str = "",
        benchmark: bool = False,
        benchmark_options: Optional[dict] = None,
    ) -> None: ...

    def benchmark(self, run_id: str, benchmark_options: Optional[dict] = None) -> None: ...

    def resume(
        self,
        shell: Any,
        run_id: str,
        select: int,
        note: str = "",
    ) -> None: ...


class AIReviewer:
    """Adapter running the LangGraph-based AI optimization-review workflow.

    Builds (and caches) the fresh-run and resume graphs, generates run
    ids, and keeps track of pending reviews so a follow-up
    ``--resume RUN_ID --select N`` can apply one of the suggestions.
    Mirrors :class:`PerformanceReporter` / :class:`PerformanceVisualizer`:
    constructed with the stable ``reporter`` and bound to the live
    monitor via :meth:`attach`.
    """

    def __init__(self, reporter: PerformanceReporter):
        self.reporter = reporter
        self.monitor: MonitorProtocol = UnavailablePerformanceMonitor(
            reason="Monitor has not been started yet."
        )
        self._pending_reviews: dict[str, dict] = {}
        self._review_graph = None
        self._resume_graph = None
        self._benchmark_graph = None

    def attach(self, monitor: MonitorProtocol) -> None:
        """Attach started PerformanceMonitor."""
        self.monitor = monitor

    def _get_review_graph(self):
        """Lazily build (and cache) the fresh-run AI review graph.

        Imported lazily so the optional ``[ai]`` extras (langgraph,
        langchain-openai) are only required when this magic is used.
        """
        if self._review_graph is None:
            from jumper_extension.adapters.ai_reviewer.agent.graph import build_review_graph
            from jumper_extension.adapters.ai_reviewer.context.collector import ContextCollector
            from jumper_extension.adapters.ai_reviewer.llm.client import LLMClientConfig, build_llm
            from jumper_extension.adapters.ai_reviewer.ui.review_display import build_ai_review_display

            llm = build_llm(LLMClientConfig.from_config())
            self._review_graph = build_review_graph(
                llm,
                ContextCollector(self),
                build_ai_review_display(),
                self.build_orchestrator,
            )
        return self._review_graph

    def _get_benchmark_graph(self):
        """Lazily build (and cache) the graph behind ``--resume --benchmark``."""
        if self._benchmark_graph is None:
            from jumper_extension.adapters.ai_reviewer.agent.graph import build_benchmark_graph
            from jumper_extension.adapters.ai_reviewer.llm.client import LLMClientConfig, build_llm
            from jumper_extension.adapters.ai_reviewer.ui.review_display import build_ai_review_display

            llm = build_llm(LLMClientConfig.from_config())
            self._benchmark_graph = build_benchmark_graph(
                llm,
                build_ai_review_display(),
                self.build_orchestrator,
            )
        return self._benchmark_graph

    def build_orchestrator(self, state, fix_fn, target_index: int):
        """Wire a benchmark for the cell at *target_index*, or None if impossible.

        The prefix is every cell before the target: the cell only means anything
        with the state its predecessors built, and a replay that skipped them
        would push the repair loop into inventing the missing data.
        """
        from jumper_extension.adapters.ai_reviewer.benchmark.checks import resolve_checks
        from jumper_extension.adapters.ai_reviewer.benchmark.orchestrator import BenchmarkOrchestrator
        from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner
        from jumper_extension.adapters.ai_reviewer.language import get_adapter

        prefix_cells = self._prefix_cells(target_index)
        if prefix_cells is None:
            return None

        adapter = get_adapter(self._cell_language(target_index))
        defaults = load_config().ai.benchmark
        options = state["benchmark_options"]
        checks = resolve_checks(adapter, defaults.checks, options.get("checks"))
        runs = options.get("runs") or defaults.runs
        fix_attempts = options.get("fix_attempts") or defaults.fix_attempts
        result_names = options.get("result_names")
        if result_names is None:
            result_names = defaults.result_names
        runner = BenchmarkRunner(
            prefix_cells=prefix_cells,
            interval=defaults.interval,
            level=state["level"],
            adapter=adapter,
            replay_mode=options.get("replay_mode") or defaults.replay.mode,
            keep_perf_series=defaults.keep_perf_series,
        )
        logger.info(
            f"[JUmPER]: benchmarking {len(state['suggestions'])} suggestion(s) against cell "
            f"{target_index}; each replays {len(prefix_cells)} preceding cell(s) "
            f"{runs} time(s). This can take a while."
        )
        return BenchmarkOrchestrator(
            runner=runner,
            fix_fn=fix_fn,
            runs=runs,
            fix_attempts=fix_attempts,
            timeout_factor=defaults.timeout_factor,
            adapter=adapter,
            checks=checks,
            cross_check=defaults.replay.cross_check,
            result_names=result_names,
        )

    def _cell_language(self, index: int) -> str:
        """Language recorded for the cell at *index*; Python for legacy rows."""
        from jumper_extension.adapters.ai_reviewer.language import resolve_language

        history = self.reporter.printer.cell_history.view()
        if history is None or history.empty or "language" not in history.columns:
            return "python"
        rows = history[history["cell_index"] == index]
        if rows.empty:
            return "python"
        return resolve_language(rows.iloc[-1]["language"])

    def _prefix_cells(self, target_index: int) -> Optional[list]:
        """Every cell executed before *target_index*, as the replay needs them."""
        history = self.reporter.printer.cell_history.view()
        if history is None or history.empty:
            logger.warning("[JUmPER]: no cell history to replay for the benchmark")
            return None

        cells = []
        for row in history.itertuples(index=False):
            index = int(row.cell_index)
            if index >= target_index:
                break
            cells.append(
                {
                    "index": index,
                    "raw_cell": row.raw_cell,
                    "cell_magics": list(getattr(row, "cell_magics", None) or []),
                    "language": getattr(row, "language", None) or "python",
                }
            )
        return cells

    def _get_resume_graph(self, shell: Any):
        """Lazily build (and cache) the resume AI review graph."""
        if self._resume_graph is None:
            from jumper_extension.adapters.ai_reviewer.agent.graph import build_resume_graph
            from jumper_extension.adapters.ai_reviewer.llm.client import LLMClientConfig, build_llm

            llm = build_llm(LLMClientConfig.from_config())
            self._resume_graph = build_resume_graph(llm, shell)
        return self._resume_graph

    def review(
        self,
        shell: Any,
        cell_range: Optional[Tuple[int, int]] = None,
        level: str = "process",
        strategy: str = "faster",
        note: str = "",
        benchmark: bool = False,
        benchmark_options: Optional[dict] = None,
    ) -> None:
        """Run the AI-powered performance review on a fresh cell selection.

        The chosen ``strategy`` resolves to a set of overrides that steer
        both which context sources are gathered and which prompt rules
        apply; ``note`` is a free-text instruction folded into the
        prompt. Collects the cell code and performance context, asks the
        LLM to identify the bottleneck and propose optimizations, then
        displays the numbered options together with ``--resume`` commands.
        """
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return

        from jumper_extension.adapters.ai_reviewer.agent.state import empty_state
        from jumper_extension.adapters.ai_reviewer.strategy import get_strategy

        resolved = get_strategy(strategy)
        if resolved.require_note and not note:
            logger.warning(
                f"[JUmPER]: strategy '{strategy}' requires a --note instruction"
            )
            return

        run_id = uuid.uuid4().hex[:8]
        initial_state = empty_state(
            run_id=run_id,
            cell_range=cell_range,
            level=level,
            overrides=resolved.overrides,
            note=note,
            benchmark=benchmark,
            benchmark_options=benchmark_options,
        )
        final_state = self._get_review_graph().invoke(initial_state)
        self._pending_reviews[run_id] = final_state

    def benchmark(self, run_id: str, benchmark_options: Optional[dict] = None) -> None:
        """Replay and time the suggestions of a review that already ran.

        Benchmarking costs far more than the review itself, so it is a separate
        decision: look at the options first, then spend the machine time.
        """
        state = self._pending_reviews.get(run_id)
        if state is None:
            logger.warning(f"[JUmPER]: No pending AI review found for run_id '{run_id}'")
            return
        if not self.monitor.running:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_ACTIVE_MONITOR]
            )
            return

        final_state = self._get_benchmark_graph().invoke({
            **state,
            "benchmark": True,
            "benchmark_options": benchmark_options or {},
        })
        self._pending_reviews[run_id] = final_state

    def resume(
        self,
        shell: Any,
        run_id: str,
        select: int,
        note: str = "",
    ) -> None:
        """Apply a previously suggested optimization, optionally refined.

        Loads the state stored under ``run_id`` by a prior ``review``
        run, marks suggestion ``select`` as chosen and runs the resume
        graph: if ``note`` is provided, the suggestion is rewritten per
        that instruction first; either way the resulting code is placed
        into the next cell via ``shell.set_next_input``.
        """
        state = self._pending_reviews.get(run_id)
        if state is None:
            logger.warning(f"[JUmPER]: No pending AI review found for run_id '{run_id}'")
            return

        if not 1 <= select <= len(state["suggestions"]):
            logger.warning(
                f"[JUmPER]: Invalid suggestion index {select} for run_id '{run_id}' "
                f"(expected 1-{len(state['suggestions'])})"
            )
            return

        resume_state = {
            **state,
            "chosen_index": select - 1,
            "note": note,
            "refined_code": None,
        }
        final_state = self._get_resume_graph(shell).invoke(resume_state)
        self._pending_reviews[run_id] = final_state


class UnavailableAIReviewer:
    """A stub that type-checks against AIReviewerProtocol but only logs
    that the optional ``[ai]`` dependencies are missing.
    """
    def __init__(self, reason: str = _AI_EXTRAS_MISSING_REASON):
        self._reason = reason

    def attach(self, monitor: MonitorProtocol) -> None: ...

    def review(
        self,
        shell: Any,
        cell_range: Optional[Tuple[int, int]] = None,
        level: str = "process",
        strategy: str = "faster",
        note: str = "",
        benchmark: bool = False,
        benchmark_options: Optional[dict] = None,
    ) -> None:
        self._warn()

    def benchmark(self, run_id: str, benchmark_options: Optional[dict] = None) -> None:
        self._warn()

    def resume(
        self,
        shell: Any,
        run_id: str,
        select: int,
        note: str = "",
    ) -> None:
        self._warn()

    def _warn(self) -> None:
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.AI_REVIEW_NOT_AVAILABLE].format(
                reason=self._reason,
                install_cmd=_ai_extras_install_cmd(),
            )
        )


def build_ai_reviewer(reporter: PerformanceReporter) -> AIReviewerProtocol:
    """Build an AI reviewer adapter attached to *reporter*.

    Falls back to :class:`UnavailableAIReviewer` when the optional
    ``[ai]`` extras (``langgraph``, ``langchain-openai``) are not
    installed, so ``%perfmonitor_ai_review`` can explain how to enable
    the feature instead of raising an ``ImportError``.
    """
    try:
        import langgraph  # noqa: F401
        import langchain_openai  # noqa: F401
    except ImportError:
        return UnavailableAIReviewer()
    return AIReviewer(reporter)
