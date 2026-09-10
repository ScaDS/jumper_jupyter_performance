"""Reading cost off the reviewer's model calls.

The reviewer keeps no account of what it spends - no latency, no tokens - and
the experiment has to report both. Rather than change the reviewer, this wraps
the factory it builds its client with and attaches a callback to the model.

Model-level callbacks are used deliberately over a context-scoped hook: the
repair loop makes its calls from a thread pool, and a ContextVar does not
follow a `ThreadPoolExecutor.submit`, so half the calls of a benchmark would go
unrecorded.
"""
from __future__ import annotations

import dataclasses
import threading
import time

# The graph node that made the call, when LangGraph tells us. The mapping is
# one-way and small enough to keep here rather than import from the reviewer.
_NODE_STEPS = {
    "analyze_bottlenecks": "analyze",
    "generate_suggestions": "suggest",
    "refine_suggestion": "refine",
}

# When the node is unknown - the repair calls, which happen off the graph - the
# position in the current window says what it is. A review is always analyze,
# then suggest, then any number of repairs.
_POSITIONAL_STEPS = ("analyze", "suggest")
_FALLBACK_STEP = "fix"


@dataclasses.dataclass
class CallRecord:
    step: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    model: str = ""


class CallRecorder:
    """Collects one :class:`CallRecord` per model call, window by window.

    A window is one reviewer invocation. ``reset`` opens a new one; ``drain``
    closes it and hands back what was collected.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._started: dict[str, float] = {}
        self._nodes: dict[str, str] = {}
        self._records: list[CallRecord] = []

    def reset(self) -> None:
        with self._lock:
            self._started.clear()
            self._nodes.clear()
            self._records.clear()

    def drain(self) -> list[CallRecord]:
        with self._lock:
            records = list(self._records)
            self._records.clear()
            return records

    def note_start(self, run_key: str, node: str | None) -> None:
        with self._lock:
            self._started[run_key] = time.perf_counter()
            if node:
                self._nodes[run_key] = node

    def note_end(self, run_key: str, usage: dict, model: str) -> None:
        with self._lock:
            started = self._started.pop(run_key, None)
            node = self._nodes.pop(run_key, None)
            position = len(self._records)
            step = _NODE_STEPS.get(node or "") or _positional_step(position)
            self._records.append(
                CallRecord(
                    step=step,
                    latency_s=(
                        round(time.perf_counter() - started, 4)
                        if started is not None
                        else 0.0
                    ),
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    model=model,
                )
            )


def _positional_step(position: int) -> str:
    if position < len(_POSITIONAL_STEPS):
        return _POSITIONAL_STEPS[position]
    return _FALLBACK_STEP


def _usage_from(response) -> dict:
    """Token counts, wherever this model version decided to put them.

    LangChain moved usage from ``llm_output['token_usage']`` to the message's
    ``usage_metadata``; a vLLM endpoint may fill either, or neither.
    """
    output = getattr(response, "llm_output", None) or {}
    usage = dict(output.get("token_usage") or {})
    normalised = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    if any(value is not None for value in normalised.values()):
        return normalised

    for batch in getattr(response, "generations", None) or []:
        for generation in batch:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "usage_metadata", None) or {}
            if metadata:
                return {
                    "input_tokens": metadata.get("input_tokens"),
                    "output_tokens": metadata.get("output_tokens"),
                    "total_tokens": metadata.get("total_tokens"),
                }
    return normalised


def _build_handler(recorder: CallRecorder):
    """The LangChain callback, built lazily so importing this is cheap."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _Handler(BaseCallbackHandler):
        raise_error = False

        def on_chat_model_start(self, serialized, messages, **kwargs):
            recorder.note_start(
                str(kwargs.get("run_id")),
                (kwargs.get("metadata") or {}).get("langgraph_node"),
            )

        def on_llm_start(self, serialized, prompts, **kwargs):
            recorder.note_start(
                str(kwargs.get("run_id")),
                (kwargs.get("metadata") or {}).get("langgraph_node"),
            )

        def on_llm_end(self, response, **kwargs):
            output = getattr(response, "llm_output", None) or {}
            recorder.note_end(
                str(kwargs.get("run_id")),
                _usage_from(response),
                str(output.get("model_name") or ""),
            )

        def on_llm_error(self, error, **kwargs):
            recorder.note_end(str(kwargs.get("run_id")), {}, "")

    return _Handler()


def install(recorder: CallRecorder) -> bool:
    """Attach *recorder* to every model the reviewer builds from now on.

    Returns False when the optional AI dependencies are absent, in which case
    the experiment has bigger problems than missing token counts.
    """
    try:
        from jumper_extension.adapters.ai_reviewer.llm import client
    except ImportError:
        return False

    if getattr(client.build_llm, "_ablation_wrapped", False):
        return True

    handler = _build_handler(recorder)
    original = client.build_llm

    def build_llm_with_recorder(config):
        model = original(config)
        existing = list(getattr(model, "callbacks", None) or [])
        model.callbacks = existing + [handler]
        return model

    build_llm_with_recorder._ablation_wrapped = True
    client.build_llm = build_llm_with_recorder
    return True
