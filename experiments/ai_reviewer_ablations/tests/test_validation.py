"""Values the models must refuse.

Each of these parses fine as JSON or YAML and produces a number that looks
like a measurement. Refusing them at the boundary is cheaper than finding a
precision above one in a finished report.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jumper_ablations.config.schema import (
    BenchmarkProtocol,
    BootstrapConfig,
    ProtocolConfig,
)
from jumper_ablations.metrics.analysis.factuality_groundedness import (
    FactualityVerdict,
)
from jumper_ablations.metrics.analysis.relevance_conciseness import (
    RelevanceVerdict,
)
from jumper_ablations.metrics.analysis.analysis_to_suggestion_transfer import (
    TransferVerdict,
)
from jumper_ablations.metrics.suggestions.resource_effect_agreement import (
    ResourcePrediction,
)
from jumper_ablations.metrics.suggestions.suggestion_diversity import (
    DiversityVerdict,
)
from jumper_ablations.usecases.registry import ReferenceFact, UsecaseManifest


def test_classified_claims_cannot_outnumber_the_claims():
    # 8 of 5 claims supported reports a precision of 1.6.
    with pytest.raises(ValidationError):
        FactualityVerdict(
            total_claims=5,
            supported_claims=8,
            contradictions=0,
            hallucinations=0,
        )


def test_the_three_claim_kinds_may_partition_the_total():
    verdict = FactualityVerdict(
        total_claims=5,
        supported_claims=3,
        contradictions=1,
        hallucinations=1,
    )

    assert verdict.total_claims == 5


def test_relevant_claims_cannot_outnumber_the_claims():
    with pytest.raises(ValidationError):
        RelevanceVerdict(total_claims=2, relevant_claims=3)


def test_aligned_suggestions_cannot_outnumber_the_suggestions():
    with pytest.raises(ValidationError):
        TransferVerdict(
            suggestions_total=3,
            bottleneck_aligned=4,
            constraints_stated=0,
            suggestions_preserving_all=0,
            constraint_observations=0,
            constraints_preserved=0,
        )


def test_a_suggestion_cannot_preserve_more_constraints_than_were_checked():
    with pytest.raises(ValidationError):
        TransferVerdict(
            suggestions_total=3,
            bottleneck_aligned=3,
            constraints_stated=2,
            suggestions_preserving_all=3,
            constraint_observations=6,
            constraints_preserved=7,
        )


def test_there_cannot_be_more_labels_than_suggestions():
    with pytest.raises(ValidationError):
        DiversityVerdict(
            suggestions_total=2,
            technique_labels=["numba", "numpy", "joblib"],
        )


@pytest.mark.parametrize(
    "field, value",
    [("resource", "disk"), ("direction", "sideways")],
)
def test_a_resource_prediction_stays_in_the_stated_vocabulary(field, value):
    payload = {
        "suggestion_index": 1,
        "resource": "cpu",
        "direction": "down",
        field: value,
    }

    with pytest.raises(ValidationError):
        ResourcePrediction(**payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"samples": 0},
        {"confidence": 1.0},
        {"confidence": 0.0},
        {"confidence": 95.0},
    ],
)
def test_a_bootstrap_that_cannot_produce_an_interval_is_refused(payload):
    with pytest.raises(ValidationError):
        BootstrapConfig(**payload)


@pytest.mark.parametrize(
    "payload",
    [{"generations_per_target": 0}, {"repetitions": 0}],
)
def test_a_protocol_that_measures_nothing_is_refused(payload):
    with pytest.raises(ValidationError):
        ProtocolConfig(**payload)


def test_a_benchmark_needs_a_warm_up_and_a_measurement():
    # The reviewer's median drops the first run, so one run measures nothing.
    with pytest.raises(ValidationError):
        BenchmarkProtocol(runs=1)


def test_a_fact_must_come_from_a_source_the_collector_gates():
    # "telemetry" is not one of the seven ids, so a fact attributed to it
    # could never be in the enabled set of any preset.
    with pytest.raises(ValidationError):
        ReferenceFact(id="f", source="telemetry", fact="...")


def test_a_weightless_fact_is_refused():
    with pytest.raises(ValidationError):
        ReferenceFact(id="f", source="code", fact="...", weight=0.0)


def test_two_facts_cannot_share_an_id():
    with pytest.raises(ValidationError, match="duplicate"):
        UsecaseManifest(
            id="synthetic/loop",
            reference_facts=[
                {"id": "loop", "source": "code", "fact": "one"},
                {"id": "loop", "source": "timing", "fact": "another"},
            ],
        )
