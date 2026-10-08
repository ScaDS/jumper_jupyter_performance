"""Several alternatives, or one idea rephrased three times?

`requested_vs_returned` counts options; this asks whether the options were
different. A response with three paraphrases of "vectorise the loop" offers a
user one choice, not three, and every per-suggestion rate above it is then
built on one idea sampled three times.

Judged rather than computed: two rewrites can share almost no source text and
still be the same idea, and can differ by one line and be different ideas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_JUDGE,
    SCOPE_RUN,
    JudgeMetric,
    MetricSpec,
)
from jumper_ablations.statistics import rate


class DiversityVerdict(BaseModel):
    suggestions_total: int = Field(ge=0)
    # One short label per suggestion, naming the technique it applies. Two
    # suggestions share a label exactly when they are the same idea.
    technique_labels: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _one_label_per_suggestion(self) -> "DiversityVerdict":
        if len(self.technique_labels) > self.suggestions_total:
            raise ValueError(
                f"{len(self.technique_labels)} labels for "
                f"{self.suggestions_total} suggestions"
            )
        return self

    @property
    def distinct_techniques(self) -> int:
        return len(
            {
                label.strip().lower()
                for label in self.technique_labels
                if label.strip()
            }
        )


class SuggestionDiversity(JudgeMetric):
    spec = MetricSpec(
        id="suggestion_diversity",
        display_name="Suggestion Diversity",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_JUDGE,
        scope=SCOPE_RUN,
        reported_values=(
            "distinct_technique_count",
            "distinct_technique_rate",
            "duplicate_rate",
            "suggestions_total",
        ),
        description=(
            "Number of genuinely distinct optimization techniques in one "
            "response, and the share of suggestions that repeat one already "
            "offered."
        ),
        supporting_values=(
            "distinct_technique_count",
            "suggestions_total",
        ),
    )
    verdict_model = DiversityVerdict

    def from_verdict(self, verdict: DiversityVerdict, context) -> dict:
        total = verdict.suggestions_total or len(verdict.technique_labels)
        distinct = verdict.distinct_techniques
        return {
            "distinct_technique_count": distinct,
            "distinct_technique_rate": rate(distinct, total),
            "duplicate_rate": rate(total - distinct, total),
            "suggestions_total": total,
        }
