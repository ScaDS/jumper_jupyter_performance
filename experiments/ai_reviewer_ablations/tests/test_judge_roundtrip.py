"""Export a packet, judge it, read the verdict back."""

from __future__ import annotations

import json
import re

import pytest

from jumper_ablations.config.schema import JudgeConfig
from jumper_ablations.evaluation.judge import JudgeEvaluation, export_packets
from jumper_ablations.evaluation.judge.export import _schema_for
from jumper_ablations.evaluation.judge.layout import verdict_path
from jumper_ablations.metrics import (
    build_cell_contexts,
    build_run_contexts,
    get_metric,
)
from jumper_ablations.metrics.base import METHOD_JUDGE
from jumper_ablations.metrics.registry import all_metrics
from jumper_ablations.usecases.registry import (
    ReferenceFact,
    Usecase,
    UsecaseManifest,
)
from jumper_ablations.usecases.notebook import NotebookLayout
from tests.factories import make_record

FACTS = [
    ReferenceFact(
        id="sequential_loop",
        source="code",
        weight=2,
        fact="The work is a sequential loop.",
    ),
    ReferenceFact(
        id="duration",
        source="timing",
        weight=1,
        fact="It takes about ten seconds.",
    ),
]


def _usecase(tmp_path) -> Usecase:
    return Usecase(
        manifest=UsecaseManifest(
            id="synthetic/loop",
            title="Loop",
            payload_type="cpu_bound_python_loop",
            reference_facts=FACTS,
        ),
        notebook_path=tmp_path / "notebook.ipynb",
        layout=NotebookLayout(
            prefix_indices=(0,),
            payload_index=1,
            review_index=2,
            review_line="%perfmonitor_ai_review",
        ),
    )


@pytest.fixture
def exported(tmp_path):
    """One packet for the shared evidence-coverage rubric."""
    record = make_record(disabled=("timing",))
    usecases = {"synthetic/loop": _usecase(tmp_path)}
    run_contexts = build_run_contexts([record], usecases)
    cell_contexts = build_cell_contexts([record], usecases)
    metrics = [
        get_metric("conditional_evidence_coverage"),
        get_metric("global_evidence_coverage"),
    ]
    packets = export_packets(
        run_directory=tmp_path,
        metrics=metrics,
        run_contexts=run_contexts,
        cell_contexts=cell_contexts,
        config=JudgeConfig(
            sources=["analyze_messages", "enabled_sources", "reference_facts"]
        ),
    )
    return tmp_path, packets, metrics, run_contexts, cell_contexts


def test_metrics_sharing_a_rubric_share_one_packet(exported):
    _root, packets, _metrics, _runs, _cells = exported

    assert len(packets) == 1
    assert packets[0].rubric == "evidence_coverage"
    assert set(packets[0].metrics) == {
        "conditional_evidence_coverage",
        "global_evidence_coverage",
    }


def test_packet_is_self_contained(exported):
    _root, packets, _metrics, _runs, _cells = exported
    packet = packets[0].directory

    assert (packet / "TASK.md").is_file()
    assert (packet / "rubric.md").is_file()
    assert (packet / "verdict.schema.json").is_file()
    assert (packet / "sources" / "analyze.messages.json").is_file()
    assert (packet / "output" / "analysis.md").is_file()

    # The rubric lists the facts to look for.
    assert "sequential_loop" in (packet / "rubric.md").read_text()


def test_the_preset_name_appears_nowhere_in_the_packet(exported):
    # The weak version of this check - "not in the header of TASK.md" - passed
    # while the ablation name sat in the directory name and in the unit id.
    # It has to be every byte of the packet, paths included.
    root, packets, _metrics, _runs, _cells = exported
    packet = packets[0].directory

    # An ablation id is a token, so match it as one: rubric prose says
    # "1-based" and "baseline", and neither of those names a preset.
    name = re.compile(rf"\b{re.escape('base')}\b")

    leaked = []
    for path in packet.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if name.search(relative):
            leaked.append(f"path {relative}")
        if path.is_file() and name.search(path.read_text(errors="ignore")):
            leaked.append(f"content of {relative}")

    assert not leaked, leaked

    # The surrogate is what the session is told to answer as.
    assert packets[0].packet_id != packets[0].unit_id
    assert packets[0].packet_id.startswith("unit-")
    assert packets[0].packet_id in (packet / "TASK.md").read_text()

    # The unblinding key exists, separately, and carries both ids.
    index = (root / "judge" / "index.csv").read_text()
    assert "packet_id" in index
    assert packets[0].unit_id in index


def test_blinding_can_be_turned_off(tmp_path):
    record = make_record(disabled=("timing",))
    usecases = {"synthetic/loop": _usecase(tmp_path)}
    packets = export_packets(
        run_directory=tmp_path,
        metrics=[get_metric("bottleneck_identification")],
        run_contexts=build_run_contexts([record], usecases),
        cell_contexts=build_cell_contexts([record], usecases),
        config=JudgeConfig(blind=False, sources=["analyze_messages"]),
    )

    assert packets[0].packet_id == packets[0].unit_id
    assert "base" in packets[0].directory.name


def test_a_missing_verdict_is_a_gap_and_not_a_zero(exported):
    root, _packets, metrics, run_contexts, cell_contexts = exported

    result = JudgeEvaluation(root).evaluate(
        metrics,
        run_contexts,
        cell_contexts,
        {},
    )

    assert all(row.value is None for row in result.rows)
    assert {gap["reason"] for gap in result.gaps} == {"missing"}


def test_one_verdict_feeds_both_denominators(exported):
    root, packets, metrics, run_contexts, cell_contexts = exported
    # A session answers as the packet addressed it, not as the harness knows
    # it internally; ingest is what maps the surrogate back.
    unit_id = packets[0].packet_id

    path = verdict_path(root, "evidence_coverage", unit_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "unit_id": unit_id,
                "rubric": "evidence_coverage",
                "judged_by": "test",
                "verdict": {
                    "covered_fact_ids": ["sequential_loop"],
                    "partially_covered_fact_ids": [],
                },
            }
        )
    )

    result = JudgeEvaluation(root).evaluate(
        metrics,
        run_contexts,
        cell_contexts,
        {},
    )
    recall = {
        row.metric: row.value
        for row in result.rows
        if row.reported_value == "weighted_recall"
    }

    # `timing` was ablated away, so the timing fact was unreachable: the
    # conditional denominator is the code fact alone, the global one is both.
    assert recall["conditional_evidence_coverage"] == pytest.approx(1.0)
    assert recall["global_evidence_coverage"] == pytest.approx(2 / 3)
    assert not result.gaps


def test_an_abstention_is_recorded_rather_than_scored(exported):
    root, packets, metrics, run_contexts, cell_contexts = exported
    unit_id = packets[0].packet_id

    path = verdict_path(root, "evidence_coverage", unit_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "unit_id": unit_id,
                "rubric": "evidence_coverage",
                "abstained": True,
                "reason": "the analysis is empty",
                "verdict": {},
            }
        )
    )

    result = JudgeEvaluation(root).evaluate(
        metrics,
        run_contexts,
        cell_contexts,
        {},
    )

    assert all(row.value is None for row in result.rows)
    assert {gap["reason"] for gap in result.gaps} == {"abstained"}


def test_a_review_context_carries_the_benchmark_that_measured_it(tmp_path):
    # The review and the benchmark are separate invocations, so a metric that
    # judges the suggestions and then checks them against measurements has to
    # be handed both halves.
    from jumper_ablations.records.schema import PHASE_REBENCHMARK
    from tests.factories import make_cell

    records = make_cell(generations=2)
    usecases = {"synthetic/loop": _usecase(tmp_path)}
    contexts = build_run_contexts(records, usecases)

    assert len(contexts) == 2
    for context in contexts:
        assert context.record.identity.phase != PHASE_REBENCHMARK
        assert not context.record.outputs.benchmarks
        assert context.measurement is not None
        assert set(context.benchmarks()) == {"baseline", "1", "2", "3"}


def test_resource_effect_agreement_reads_the_measured_direction(tmp_path):
    from tests.factories import make_cell

    records = make_cell(generations=1)
    usecases = {"synthetic/loop": _usecase(tmp_path)}
    context = build_run_contexts(records, usecases)[0]

    metric = get_metric("resource_effect_agreement")
    verdict = metric.verdict_model.model_validate(
        {
            "predictions": [
                # Baseline cpu_util_mean 0.9 -> variant 0.4: it went down.
                {
                    "suggestion_index": 1,
                    "resource": "cpu",
                    "direction": "down",
                },
                {"suggestion_index": 2, "resource": "cpu", "direction": "up"},
            ]
        }
    )
    values = metric.from_verdict(verdict, context)

    assert values["checked_predictions"] == 2
    assert values["agreement_rate"] == pytest.approx(0.5)


def _pointers(node, found: list[str]) -> list[str]:
    """Every `$ref` string anywhere in a schema."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                found.append(value)
            else:
                _pointers(value, found)
    elif isinstance(node, list):
        for item in node:
            _pointers(item, found)
    return found


def _resolve(schema: dict, pointer: str):
    """Walk a `#/a/b` pointer from the document root, or raise."""
    assert pointer.startswith("#/"), pointer
    node = schema
    for step in pointer[2:].split("/"):
        step = step.replace("~1", "/").replace("~0", "~")
        assert step in node, f"{pointer} stops at {step!r}"
        node = node[step]
    return node


@pytest.mark.parametrize(
    "metric_id",
    [
        metric.id
        for metric in all_metrics().values()
        if metric.spec.evaluation_method == METHOD_JUDGE
    ],
)
def test_every_ref_in_an_exported_schema_resolves(metric_id):
    # A verdict model with a nested model - resource_effect_agreement has one
    # - carries its definitions in a `$defs` of its own and points at them
    # from the document root. Inlining the payload under `properties.verdict`
    # used to move the definitions without the pointers, so the packet
    # shipped a schema that rejected every answer to it, including its own
    # example. A judge found this by being unable to validate its verdict.
    metric = get_metric(metric_id)
    schema = _schema_for(metric.spec.id, metric)

    for pointer in _pointers(schema, []):
        _resolve(schema, pointer)


def test_a_record_whose_usecase_is_gone_refuses_to_export(tmp_path):
    # Reference facts come from the manifest on disk, so a usecase renamed
    # after a run leaves its records unresolvable. Exporting anyway produced
    # an empty fact sheet, and evidence coverage is a recall: an empty
    # denominator scores every analysis alike and looks like a measurement.
    record = make_record()
    run_contexts = build_run_contexts([record], {})
    cell_contexts = build_cell_contexts([record], {})

    with pytest.raises(ValueError, match="not in the registry"):
        export_packets(
            run_directory=tmp_path,
            metrics=[get_metric("conditional_evidence_coverage")],
            run_contexts=run_contexts,
            cell_contexts=cell_contexts,
            config=JudgeConfig(),
        )
