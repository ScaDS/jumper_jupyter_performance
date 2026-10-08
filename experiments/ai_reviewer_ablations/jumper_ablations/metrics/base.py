"""What a metric is, whichever category it belongs to.

Analysis metrics and suggestion metrics answer different questions and are
kept in different packages, but they are the same kind of object: a
declaration of what it produces, and a way of producing it. That shared shape
is what lets one config file switch either of them on, one registry find both,
and one evaluation method score a whole group without knowing which category
it came from.

Two axes, and they are independent. ``category`` says what is being measured -
the analysis, or the suggestions. ``evaluation_method`` says how - by
computation over the records, or by an agent reading the same sources the
model read. A metric picks one of each.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic import BaseModel

CATEGORY_ANALYSIS = "analysis"
CATEGORY_SUGGESTIONS = "suggestions"
CATEGORIES = (CATEGORY_ANALYSIS, CATEGORY_SUGGESTIONS)

METHOD_DETERMINISTIC = "deterministic"
METHOD_JUDGE = "judge"
METHODS = (METHOD_DETERMINISTIC, METHOD_JUDGE)

# A metric either reads one reviewer invocation, or the whole group of them
# that shares a (usecase, ablation) cell of the grid. pass@k, best-of and rank
# correlation are only defined over the group; correctness rates are defined
# per invocation and averaged afterwards.
SCOPE_RUN = "run"
SCOPE_CELL = "cell"
SCOPES = (SCOPE_RUN, SCOPE_CELL)

# A reported value is either the metric's answer or a quantity that answer was
# computed from. Both are reported, because a rate without its denominator is
# unreadable - verified_rate = 1.0 over one suggestion and over sixty are
# different claims - but only the first kind is a result, and a reader has to
# be able to ask for just those.
KIND_RESULT = "result"
KIND_INPUT = "input"


@dataclasses.dataclass(frozen=True)
class MetricSpec:
    """A metric's declaration: what it is and what it returns."""

    id: str
    category: str
    evaluation_method: str
    scope: str
    reported_values: tuple[str, ...]
    description: str
    # The metric's name in the source spreadsheet, which is what a reader
    # recognises; the id is what the files are keyed by. Only a result
    # carries it: a denominator is not a row of that table.
    display_name: str = ""
    # The subset of reported_values that are the quantities the results were
    # computed from rather than results themselves, named as the source
    # spreadsheet's Units column files them under "count".
    supporting_values: tuple[str, ...] = ()

    def name_of(self, reported_value: str) -> str:
        """The spreadsheet's name for this value, or "" when it has none.

        Empty for every supporting quantity, and for a metric the
        spreadsheet does not have a row for - which is a fact worth seeing
        rather than papering over with the id.
        """
        if self.kind_of(reported_value) != KIND_RESULT:
            return ""
        return self.display_name

    def kind_of(self, reported_value: str) -> str:
        return (
            KIND_INPUT
            if reported_value in self.supporting_values
            else KIND_RESULT
        )

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        if self.evaluation_method not in METHODS:
            raise ValueError(
                f"{self.id}: unknown evaluation method "
                f"{self.evaluation_method!r}"
            )
        if self.scope not in SCOPES:
            raise ValueError(f"{self.id}: unknown scope {self.scope!r}")
        if not self.reported_values:
            raise ValueError(f"{self.id}: a metric has to report something")
        unknown = set(self.supporting_values) - set(self.reported_values)
        if unknown:
            raise ValueError(
                f"{self.id}: supporting values not reported: "
                f"{', '.join(sorted(unknown))}"
            )
        if not set(self.reported_values) - set(self.supporting_values):
            raise ValueError(
                f"{self.id}: every reported value is supporting, so the "
                "metric has no result"
            )


@dataclasses.dataclass(frozen=True)
class MetricRow:
    """One number, and everything needed to know what it is a number about."""

    metric: str
    reported_value: str
    value_kind: str
    value: float | None
    usecase: str
    ablation: str
    unit_id: str
    scope: str
    category: str
    evaluation_method: str
    sample_size: int = 0
    note: str = ""


class Metric:
    """Base for every metric.

    A subclass sets ``spec`` and implements the half its evaluation method
    calls: ``compute`` for the computed path, ``from_verdict`` for the judged
    one.
    """

    spec: MetricSpec

    @property
    def id(self) -> str:
        return self.spec.id

    def rows(self, context, values: dict, note: str = "") -> list[MetricRow]:
        """Turn a ``{reported value: number}`` mapping into rows.

        Every declared value produces a row even when it is None, so a gap in
        the results reads as a gap rather than as a missing metric.
        """
        return [
            MetricRow(
                metric=self.spec.id,
                reported_value=name,
                value_kind=self.spec.kind_of(name),
                value=values.get(name),
                usecase=context.usecase_id,
                ablation=context.ablation_id,
                unit_id=context.unit_id,
                scope=self.spec.scope,
                category=self.spec.category,
                evaluation_method=self.spec.evaluation_method,
                sample_size=context.sample_size,
                note=note,
            )
            for name in self.spec.reported_values
        ]


class DeterministicMetric(Metric):
    """A metric computed from the records, with no model in the loop."""

    def compute(self, context, parameters: dict) -> dict:
        raise NotImplementedError


class JudgeMetric(Metric):
    """A metric an agent session produces, reading the model's own sources.

    The harness never calls a judge model. It exports a packet holding exactly
    what the reviewer was given and what it answered, and reads back a verdict
    file. ``verdict_model`` is the shape of that file, and it is written into
    each packet as JSON Schema so the session knows what to produce.
    """

    verdict_model: "type[BaseModel]"

    def rubric_id(self) -> str:
        """Which rubric folder describes this metric's scale."""
        return self.spec.id

    def packet_extras(self, context) -> dict:
        """Extra files for this metric's packet, as ``{name: content}``."""
        return {}

    def from_verdict(self, verdict, context) -> dict:
        raise NotImplementedError
