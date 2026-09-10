"""The in-kernel session: watch one review, then write it down.

Three calls, in the order the runner injects them. ``bootstrap`` once per
kernel, ``begin`` before each magic invocation, ``capture`` after it. Between
``begin`` and ``capture`` nothing here touches the reviewer - the magic runs
exactly as a user's would, and this only reads the state it left behind.

``capture`` prints one marker line, which is how the runner outside the kernel
learns the reviewer's run id and can build the ``--resume`` invocation that
re-scores the same suggestions under another replay mode.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import time
from pathlib import Path

from jumper_ablations.records.schema import (
    PHASE_REVIEW,
    BenchmarkRecord,
    LLMCall,
    RunCost,
    RunEnvironment,
    RunIdentity,
    RunInputs,
    RunOutputs,
    RunRecord,
    SuggestionRecord,
)
from jumper_ablations.records.store import RecordStore
from jumper_ablations.runtime import log_capture, llm_recorder
from jumper_ablations.runtime.serialize import jsonable, messages_as_dicts

# The runner greps stdout for this. Kept boring on purpose: it has to survive
# whatever the notebook prints around it.
CAPTURE_MARKER = "JUMPER_ABLATION_CAPTURE "
TARGET_MARKER = "JUMPER_ABLATION_TARGET "

_SESSION: "AblationSession | None" = None


def _reviewer():
    """The live AIReviewer inside this kernel, or None before %load_ext."""
    from jumper_extension.ipython import extension

    magics = getattr(extension, "_perfmonitor_magics", None)
    if magics is None:
        return None
    return magics.magic_adapter.service.ai_reviewer


def _cell_history():
    """The live cell history, whichever attribute this version exposes."""
    reviewer = _reviewer()
    if reviewer is None:
        return None
    reporter = reviewer.reporter
    printer = getattr(reporter, "printer", None)
    history = getattr(printer, "cell_history", None)
    if history is None:
        history = getattr(reporter, "cell_history", None)
    return history


def resolve_target(payload_path: str) -> int:
    """The cell index the payload was executed under, matched by its source.

    The magic's own default - "the last cell that was not short" - cannot be
    used here: the harness injects its own cells after the payload, and which
    of them counts as short is a timing accident. Matching the source is exact,
    and it survives the payload being re-executed.
    """
    source = Path(payload_path).read_text(encoding="utf-8").strip()
    history = _cell_history()
    view = history.view() if history is not None else None
    if view is None or view.empty:
        raise RuntimeError("no cell history: is the monitor running?")

    matches = [
        int(row.cell_index)
        for row in view.itertuples(index=False)
        if str(row.raw_cell).strip() == source
    ]
    if not matches:
        raise RuntimeError(
            "the payload cell is not in the cell history; the notebook and "
            "the harness disagree about which cell is the target"
        )
    index = matches[-1]
    print(TARGET_MARKER + json.dumps({"cell_index": index}), flush=True)
    return index


def _source_fields() -> dict:
    """The collector's own source table: id -> (state field, empty, default).

    Imported rather than copied. It is the single place that says which
    sources exist, and a copy here would go stale exactly when a new source is
    added - which is the moment the experiment most needs to know.
    """
    from jumper_extension.adapters.ai_reviewer.context.collector import (
        _SOURCE_FIELDS,
    )

    return _SOURCE_FIELDS


def _llm_config() -> dict:
    """The resolved model settings, with the API key left out."""
    try:
        from jumper_extension.adapters.ai_reviewer.llm.client import (
            LLMClientConfig,
        )
    except ImportError:
        return {}
    config = dataclasses.asdict(LLMClientConfig.from_config())
    config.pop("api_key", None)
    return config


@dataclasses.dataclass
class _Window:
    """One magic invocation, from ``begin`` to ``capture``."""

    generation: int
    phase: str
    requested_replay_mode: str
    reviewer_run_id: str | None
    known_run_ids: frozenset
    started_at: float


class AblationSession:
    """Everything one kernel needs to know about the run it belongs to."""

    def __init__(
        self,
        run_directory: Path,
        usecase: str,
        ablation: str,
        ablation_family: str,
        repetition: int,
        suite: str,
        run_id: str,
    ):
        self.store = RecordStore(Path(run_directory))
        self.usecase = usecase
        self.ablation = ablation
        self.ablation_family = ablation_family
        self.repetition = repetition
        self.suite = suite
        self.run_id = run_id
        self.recorder = llm_recorder.CallRecorder()
        self.warnings = log_capture.WarningCollector()
        self.window: _Window | None = None
        self.store.ensure()

    def install(self) -> None:
        self.warnings.attach()
        llm_recorder.install(self.recorder)

    def begin(
        self,
        generation: int,
        phase: str = PHASE_REVIEW,
        requested_replay_mode: str = "",
        reviewer_run_id: str | None = None,
    ) -> None:
        reviewer = _reviewer()
        known = frozenset(getattr(reviewer, "_pending_reviews", {}) or {})
        self.recorder.reset()
        self.warnings.reset()
        self.window = _Window(
            generation=generation,
            phase=phase,
            requested_replay_mode=requested_replay_mode,
            reviewer_run_id=reviewer_run_id,
            known_run_ids=known,
            started_at=time.perf_counter(),
        )

    def capture(self) -> dict:
        """Write the record for the window that just closed."""
        if self.window is None:
            raise RuntimeError("capture() called without a preceding begin()")
        window, self.window = self.window, None

        reviewer = _reviewer()
        pending = dict(getattr(reviewer, "_pending_reviews", {}) or {})
        reviewer_run_id = _resolve_run_id(window, pending)
        state = pending.get(reviewer_run_id) if reviewer_run_id else None

        record = self._build_record(window, reviewer_run_id or "", state)
        self.store.write(record)

        summary = {
            "record_id": record.identity.record_id,
            "reviewer_run_id": record.identity.reviewer_run_id,
            "suggestions": len(record.outputs.suggestions),
            "benchmarked": bool(record.outputs.benchmarks),
            "degraded": record.environment.degraded,
            "populated_sources": populated_sources(record.inputs),
        }
        print(CAPTURE_MARKER + json.dumps(summary), flush=True)
        return summary

    def _build_record(
        self,
        window: _Window,
        reviewer_run_id: str,
        state,
    ) -> RunRecord:
        identity = RunIdentity(
            record_id=self._record_id(window),
            usecase=self.usecase,
            ablation=self.ablation,
            ablation_family=self.ablation_family,
            repetition=self.repetition,
            generation=window.generation,
            phase=window.phase,
            reviewer_run_id=reviewer_run_id,
            requested_replay_mode=window.requested_replay_mode,
            created_at=datetime.datetime.now().astimezone().isoformat(),
            suite=self.suite,
            run_id=self.run_id,
        )
        messages = self.warnings.drain()
        environment = RunEnvironment(
            llm_config=_llm_config(),
            hardware=jsonable((state or {}).get("hardware_info", {})),
            actual_replay_mode=log_capture.actual_replay_mode(
                window.requested_replay_mode,
                messages,
            ),
            degraded=log_capture.degraded_to_full(messages),
            warnings=messages,
        )
        cost = _cost_from(
            self.recorder.drain(),
            time.perf_counter() - window.started_at,
        )
        if state is None:
            environment.warnings.append(EMPTY_CONTEXT_WARNING)
            return RunRecord(
                identity=identity,
                cost=cost,
                environment=environment,
            )

        inputs = _inputs_from(state)
        if populated_sources(inputs) == 0:
            environment.warnings.append(EMPTY_CONTEXT_WARNING)
        return RunRecord(
            identity=identity,
            inputs=inputs,
            outputs=_outputs_from(state),
            cost=cost,
            environment=environment,
        )

    def _record_id(self, window: _Window) -> str:
        usecase = self.usecase.replace("/", "-")
        mode = window.requested_replay_mode or "none"
        return (
            f"{usecase}__{self.ablation}"
            f"__r{self.repetition:02d}__g{window.generation:02d}"
            f"__{window.phase}__{mode}"
        )


def _resolve_run_id(window: _Window, pending: dict) -> str | None:
    """Which pending review this window produced.

    A fresh review adds a key; a ``--resume`` benchmark updates one in place,
    so for that case the runner tells us which id it asked for.
    """
    if window.reviewer_run_id:
        return window.reviewer_run_id
    added = [key for key in pending if key not in window.known_run_ids]
    if len(added) == 1:
        return added[0]
    return added[-1] if added else None


EMPTY_CONTEXT_WARNING = (
    "[JUmPER ablations]: the reviewer collected no context at all - the "
    "analysis was produced from an empty message and is evidence of nothing. "
    "This happens when the target cell has no performance data (too short, or "
    "the monitor was not running): the reviewer logs a warning and calls the "
    "model anyway, and the model answers."
)


def populated_sources(inputs: RunInputs) -> int:
    """How many enabled sources actually carried a payload.

    Zero means the model was asked to analyse nothing. The reviewer does not
    stop there, so the experiment has to notice it here instead: a
    confabulated analysis scored as a measurement would corrupt every metric
    that reads this record.
    """
    return sum(
        1
        for source_id, enabled in inputs.enabled_sources.items()
        if enabled and inputs.context_payload.get(source_id)
    )


def _inputs_from(state) -> RunInputs:
    from jumper_extension.adapters.ai_reviewer.agent.nodes import (
        build_analyze_messages,
        build_suggest_messages,
    )

    overrides = dict(state.get("overrides") or {})
    enabled = {}
    payload = {}
    for source_id, (field, _empty, default) in _source_fields().items():
        enabled[source_id] = bool(overrides.get(source_id, default))
        payload[source_id] = jsonable(state.get(field))

    return RunInputs(
        overrides=overrides,
        enabled_sources=enabled,
        context_payload=payload,
        analyze_messages=messages_as_dicts(build_analyze_messages(state)),
        suggest_messages=messages_as_dicts(build_suggest_messages(state)),
        cell_code=state.get("cell_code", ""),
        level=state.get("level", "process"),
        note=state.get("note", ""),
    )


def _outputs_from(state) -> RunOutputs:
    suggestions = [
        SuggestionRecord(
            index=index,
            title=suggestion.title,
            description=suggestion.description,
            code=suggestion.code,
            target_cell_index=suggestion.target_cell_index,
        )
        for index, suggestion in enumerate(state.get("suggestions") or [], 1)
    ]
    benchmarks = {
        label: BenchmarkRecord(**jsonable(dataclasses.asdict(result)))
        for label, result in (state.get("benchmarks") or {}).items()
    }
    return RunOutputs(
        analysis=state.get("analysis", ""),
        analysis_reasoning=state.get("analysis_reasoning", ""),
        suggestions=suggestions,
        benchmarks=benchmarks,
    )


def _cost_from(calls, wall_seconds: float) -> RunCost:
    recorded = [
        LLMCall(
            step=call.step,
            latency_s=call.latency_s,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            total_tokens=call.total_tokens,
            model=call.model,
        )
        for call in calls
    ]
    totals = [call.total_tokens for call in recorded if call.total_tokens]
    return RunCost(
        llm_calls=recorded,
        llm_latency_s=round(sum(call.latency_s for call in recorded), 4),
        total_tokens=sum(totals) if totals else None,
        command_wall_s=round(wall_seconds, 4),
    )


def bootstrap(
    run_directory: str,
    usecase: str,
    ablation: str,
    ablation_family: str = "context",
    repetition: int = 0,
    suite: str = "",
    run_id: str = "",
) -> None:
    """Open the kernel's session and install the observers."""
    global _SESSION
    _SESSION = AblationSession(
        run_directory=Path(run_directory),
        usecase=usecase,
        ablation=ablation,
        ablation_family=ablation_family,
        repetition=repetition,
        suite=suite,
        run_id=run_id,
    )
    _SESSION.install()


def current_session() -> AblationSession:
    if _SESSION is None:
        raise RuntimeError("bootstrap() has not been called in this kernel")
    return _SESSION


def begin(
    generation: int,
    phase: str = PHASE_REVIEW,
    requested_replay_mode: str = "",
    reviewer_run_id: str | None = None,
) -> None:
    current_session().begin(
        generation=generation,
        phase=phase,
        requested_replay_mode=requested_replay_mode,
        reviewer_run_id=reviewer_run_id,
    )


def capture() -> dict:
    return current_session().capture()
