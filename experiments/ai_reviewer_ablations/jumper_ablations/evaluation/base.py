"""What every evaluation method has to look like from outside."""
from __future__ import annotations

import dataclasses

from jumper_ablations.metrics.base import (
    SCOPE_CELL,
    SCOPE_RUN,
    Metric,
    MetricRow,
)


@dataclasses.dataclass
class EvaluationResult:
    """Rows produced, and the units that produced none.

    Gaps are first-class. A judged metric with no verdict yet is missing, not
    zero, and a report that could not tell the difference would quietly move
    every average towards whichever presets happened to be judged first.
    """

    rows: list[MetricRow] = dataclasses.field(default_factory=list)
    gaps: list[dict] = dataclasses.field(default_factory=list)

    def extend(self, other: "EvaluationResult") -> None:
        self.rows.extend(other.rows)
        self.gaps.extend(other.gaps)


def contexts_for(metric: Metric, run_contexts, cell_contexts):
    """The contexts a metric of this scope should see."""
    if metric.spec.scope == SCOPE_RUN:
        return run_contexts
    if metric.spec.scope == SCOPE_CELL:
        return cell_contexts
    raise ValueError(f"{metric.id}: unhandled scope {metric.spec.scope!r}")


class EvaluationMethod:
    """One way of turning metrics plus records into rows."""

    id: str

    def evaluate(
        self,
        metrics: list[Metric],
        run_contexts: list,
        cell_contexts: list,
        parameters: dict,
    ) -> EvaluationResult:
        raise NotImplementedError


def empty_rows(metric: Metric, context, note: str) -> list[MetricRow]:
    """Rows with no value, so a gap is visible instead of absent."""
    return metric.rows(context, {}, note=note)
