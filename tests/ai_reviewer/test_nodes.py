import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from jumper_extension.adapters.ai_reviewer.agent.nodes import (
    _other_options_as_diffs,
    _should_refine,
    _suggestion_list_schema,
    prompt_logger,
    analyze_bottlenecks_node,
    apply_suggestion_node,
    build_analyze_messages,
    build_suggest_messages,
    collect_context_node,
    display_results_node,
    generate_suggestions_node,
    refine_suggestion_node,
)
from jumper_extension.adapters.ai_reviewer.agent.state import Suggestion, empty_state
from jumper_extension.config.loader import load_config


def _state_with_suggestions(**overrides):
    state = empty_state(run_id="abc123")
    state["cell_code"] = "for i in range(n):\n    work(i)"
    state["analysis"] = "The loop is CPU-bound and runs serially."
    state["suggestions"] = [
        Suggestion(title="Vectorize loop", description="Use numpy", code="work_vectorized(n)"),
        Suggestion(title="Parallelize", description="Use joblib", code="parallel_work(n)"),
    ]
    state.update(overrides)
    return state


def test_collect_context_node_merges_collected_data_and_keeps_run_id():
    state = empty_state(run_id="abc123", cell_range=(1, 2), level="user")
    collected = empty_state(run_id="placeholder", cell_range=(1, 2), level="user")
    collected["cell_code"] = "x = 1"
    collector = Mock()
    collector.collect = Mock(return_value=collected)

    result = collect_context_node(state, collector)

    collector.collect.assert_called_once_with((1, 2), "user", {})
    assert result["run_id"] == "abc123"
    assert result["cell_code"] == "x = 1"


def test_collect_context_node_returns_state_unchanged_when_no_data():
    state = empty_state(run_id="abc123")
    collector = Mock()
    collector.collect = Mock(return_value=None)

    result = collect_context_node(state, collector)

    assert result == state


def test_analyze_bottlenecks_node_sets_analysis_from_llm_response():
    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = compute()"
    state["perf_tags"] = ["cpu_bound"]
    state["perf_summary"] = {"overall": {"cpu": {"mean": 80.0, "max": 99.0}}}
    state["hardware_info"] = {"num_cpus": 8}

    llm = Mock()
    llm.invoke = Mock(return_value=SimpleNamespace(content="CPU is the bottleneck."))

    result = analyze_bottlenecks_node(state, llm)

    assert result["analysis"] == "CPU is the bottleneck."
    llm.invoke.assert_called_once()
    messages = llm.invoke.call_args[0][0]
    assert "x = compute()" in messages[1].content
    assert "cpu_bound" in messages[1].content


def test_analyze_prompt_reports_each_cell_duration_over_a_range():
    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = load()\n---\ny = slow(x)"
    state["timing_info"] = {
        "total_duration_s": 5.0,
        "per_cell_duration_s": {2: 1.0, 3: 4.0},
    }

    messages = build_analyze_messages(state)

    human = messages[1].content
    assert "total: 5.0s" in human
    assert "cell 2: 1.0s" in human
    assert "cell 3: 4.0s" in human


def test_analyze_prompt_breaks_metrics_down_per_cell_over_a_range():
    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = load()\n---\ny = slow(x)"
    state["perf_summary"] = {
        "overall": {"cpu": {"mean": 50.0, "max": 90.0}},
        "per_cell": {
            2: {"cpu": {"mean": 10.0, "max": 10.0}},
            3: {"cpu": {"mean": 90.0, "max": 90.0}},
        },
    }

    human = build_analyze_messages(state)[1].content

    assert "cell 2: {'cpu': {'mean': 10.0, 'max': 10.0}}" in human
    assert "cell 3: {'cpu': {'mean': 90.0, 'max': 90.0}}" in human


def test_analyze_prompt_omits_per_cell_metrics_for_a_single_cell():
    state = empty_state(run_id="abc123")
    state["perf_summary"] = {"overall": {"cpu": {"mean": 80.0, "max": 99.0}}}

    human = build_analyze_messages(state)[1].content

    assert "overall: {'cpu': {'mean': 80.0, 'max': 99.0}}" in human
    assert "Per cell" not in human


def test_analyze_prompt_omits_timing_when_not_collected():
    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = compute()"

    messages = build_analyze_messages(state)

    assert "Execution time" not in messages[1].content


class _ListHandler(logging.Handler):
    """Collect formatted prompt-log messages without touching the real file."""

    def __init__(self, records: list[str]):
        super().__init__()
        self._records = records

    def emit(self, record):
        self._records.append(record.getMessage())


def _run_analyze_capturing_prompts(level: int) -> list[str]:
    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = compute()"
    llm = Mock()
    llm.invoke = Mock(return_value=SimpleNamespace(content="CPU is the bottleneck."))

    records: list[str] = []
    handler = _ListHandler(records)
    prompt_logger.addHandler(handler)
    extension_logger = logging.getLogger("extension")
    previous = extension_logger.level
    extension_logger.setLevel(level)
    try:
        analyze_bottlenecks_node(state, llm)
    finally:
        extension_logger.setLevel(previous)
        prompt_logger.removeHandler(handler)
    return records


def test_prompts_are_not_logged_below_debug():
    assert _run_analyze_capturing_prompts(level=logging.INFO) == []


def test_prompts_and_reply_are_logged_once_debug_is_enabled():
    records = _run_analyze_capturing_prompts(level=logging.DEBUG)

    joined = "\n".join(records)
    assert "x = compute()" in joined              # the request as sent
    assert "CPU is the bottleneck." in joined     # the reply as received
    assert "abc123" in joined                     # keyed by run_id


def test_generate_suggestions_node_builds_suggestions_from_structured_output():
    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = compute()"
    state["analysis"] = "CPU is the bottleneck."
    state["env_info"] = {"numpy": "1.26.0", "torch": "2.3.0"}

    structured_response = SimpleNamespace(suggestions=[
        SimpleNamespace(title="Vectorize", description="Use numpy", code="x = vectorized()", target_cell_index=None),
        SimpleNamespace(title="Cache", description="Memoize results", code="x = cached()", target_cell_index=None),
    ])
    structured_llm = Mock()
    structured_llm.invoke = Mock(return_value=structured_response)
    llm = Mock()
    llm.with_structured_output = Mock(return_value=structured_llm)

    result = generate_suggestions_node(state, llm)

    assert result["suggestions"] == [
        Suggestion(title="Vectorize", description="Use numpy", code="x = vectorized()"),
        Suggestion(title="Cache", description="Memoize results", code="x = cached()"),
    ]
    prompt = structured_llm.invoke.call_args[0][0][1].content
    assert "Available libraries" in prompt
    assert "numpy" in prompt
    assert "torch" in prompt



def test_suggest_step_asks_for_and_accepts_exactly_the_configured_count(monkeypatch):
    monkeypatch.setattr(load_config().ai.suggestions, "count", 5)
    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = compute()"
    state["analysis"] = "CPU is the bottleneck."
    structured_llm = Mock()
    structured_llm.invoke = Mock(return_value=SimpleNamespace(suggestions=[]))
    llm = Mock()
    llm.with_structured_output = Mock(return_value=structured_llm)

    generate_suggestions_node(state, llm)

    system_prompt = structured_llm.invoke.call_args[0][0][0].content
    assert "You must propose exactly 5 concrete optimization options." in system_prompt
    schema = llm.with_structured_output.call_args[0][0].model_json_schema()
    assert schema["properties"]["suggestions"]["minItems"] == 5
    assert schema["properties"]["suggestions"]["maxItems"] == 5


def test_a_single_option_is_asked_for_in_the_singular(monkeypatch):
    monkeypatch.setattr(load_config().ai.suggestions, "count", 1)
    state = empty_state(run_id="abc123")
    state["analysis"] = "CPU is the bottleneck."

    system_prompt = build_suggest_messages(state)[0].content

    assert "You must propose exactly 1 concrete optimization option." in system_prompt


@pytest.mark.parametrize("returned", [2, 4])
def test_a_reply_with_another_number_of_options_is_refused(returned):
    option = {"title": "Vectorize", "description": "Use numpy", "code": "x = vectorized()"}

    with pytest.raises(ValidationError):
        _suggestion_list_schema(3).model_validate({"suggestions": [option] * returned})


def test_display_results_node_delegates_to_review_display_and_keeps_state():
    state = _state_with_suggestions()
    review_display = Mock()

    result = display_results_node(state, review_display)

    review_display.display.assert_called_once_with(state)
    assert result == state


def test_refine_suggestion_node_sets_refined_code_from_llm_response():
    state = _state_with_suggestions(chosen_index=0, note="use multiprocessing")

    llm = Mock()
    llm.invoke = Mock(return_value=SimpleNamespace(content="work_multiprocessed(n)"))

    result = refine_suggestion_node(state, llm)

    assert result["refined_code"] == "work_multiprocessed(n)"
    prompt = llm.invoke.call_args[0][0][1].content
    assert "work_vectorized(n)" in prompt        # chosen full code
    assert "use multiprocessing" in prompt       # custom instruction
    assert "Option 2" in prompt                  # other option present as diff
    assert "Parallelize" in prompt


def test_other_options_as_diffs_skips_chosen_and_returns_unified_diff():
    suggestions = [
        Suggestion(title="Vectorize", description="", code="x = np.array(data)"),
        Suggestion(title="Cache", description="", code="x = cached(data)"),
    ]

    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = list(data)"

    result = _other_options_as_diffs(
        state,
        suggestions=suggestions,
        chosen_index=0,
    )

    assert "Option 1" not in result
    assert "Option 2 — Cache" in result
    assert "-x = list(data)" in result
    assert "+x = cached(data)" in result


def test_other_options_as_diffs_anchors_to_the_cell_the_option_targets():
    suggestions = [
        Suggestion(title="Chosen", description="", code="x = load_fast()", target_cell_index=2),
        Suggestion(title="Cache", description="", code="y = cached(x)", target_cell_index=3),
    ]
    state = empty_state(run_id="abc123")
    state["cell_code"] = "# --- cell 2 ---\nx = load()\n# --- cell 3 ---\ny = slow(x)"
    state["cell_sources"] = {2: "x = load()", 3: "y = slow(x)"}

    result = _other_options_as_diffs(state, suggestions, chosen_index=0)

    assert "-y = slow(x)" in result
    assert "+y = cached(x)" in result
    # Cells this option leaves alone must not read as deletions.
    assert "-x = load()" not in result
    assert "cell 2 ---" not in result


def test_other_options_as_diffs_returns_empty_string_for_single_suggestion():
    suggestions = [Suggestion(title="Only", description="", code="x = 1")]

    state = empty_state(run_id="abc123")
    state["cell_code"] = "x = 0"

    result = _other_options_as_diffs(state, suggestions, chosen_index=0)

    assert result == ""


def test_apply_suggestion_node_sets_next_input_to_chosen_suggestion_when_not_refined():
    state = _state_with_suggestions(chosen_index=1, refined_code=None)
    shell = Mock()

    result = apply_suggestion_node(state, shell)

    shell.set_next_input.assert_called_once_with("parallel_work(n)")
    assert result["applied"] is True


def test_apply_suggestion_node_prefers_refined_code_when_present():
    state = _state_with_suggestions(chosen_index=0, refined_code="work_refined(n)")
    shell = Mock()

    apply_suggestion_node(state, shell)

    shell.set_next_input.assert_called_once_with("work_refined(n)")


def test_should_refine_routes_based_on_note():
    assert _should_refine(_state_with_suggestions(note="use multiprocessing")) == "refine"
    assert _should_refine(_state_with_suggestions(note="")) == "apply"
