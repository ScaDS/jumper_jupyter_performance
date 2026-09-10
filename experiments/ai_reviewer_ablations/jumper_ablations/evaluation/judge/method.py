"""The judged path as an evaluation method.

From `cli.evaluate`'s side this is interchangeable with the computed one: it
takes metrics and contexts and returns rows. What it does instead of computing
is look for a verdict file, and what it does when there is none is say so.

Three outcomes, and they are not the same:

- **judged** - a verdict exists and validates; the metric turns it into rows.
- **abstained** - the session declined, with a reason. Excluded from the
  metric, recorded as a gap.
- **missing** - nobody has judged this unit yet. Also a gap.

None of the three ever becomes a zero.
"""
from __future__ import annotations

import logging
from pathlib import Path

from jumper_ablations.evaluation.base import (
    EvaluationMethod,
    EvaluationResult,
    contexts_for,
    empty_rows,
)
from jumper_ablations.evaluation.judge.ingest import VerdictProblem, load_verdict
from jumper_ablations.metrics.base import METHOD_JUDGE, JudgeMetric

logger = logging.getLogger("jumper_ablations")


class JudgeEvaluation(EvaluationMethod):
    id = METHOD_JUDGE

    def __init__(self, run_directory: Path):
        self.run_directory = Path(run_directory)

    def evaluate(
        self,
        metrics: list[JudgeMetric],
        run_contexts: list,
        cell_contexts: list,
        parameters: dict,
    ) -> EvaluationResult:
        result = EvaluationResult()
        for metric in metrics:
            rubric = metric.rubric_id()
            for context in contexts_for(metric, run_contexts, cell_contexts):
                self._one(result, metric, rubric, context)
        return result

    def _one(self, result, metric, rubric, context) -> None:
        try:
            payload, envelope = load_verdict(
                self.run_directory,
                rubric,
                context.unit_id,
                metric.verdict_model,
            )
        except VerdictProblem as failure:
            self._gap(result, metric, rubric, context, "invalid", str(failure))
            return

        if envelope is None:
            self._gap(result, metric, rubric, context, "missing", "")
            return
        if payload is None:
            self._gap(result, metric, rubric, context, "abstained", envelope.reason)
            return

        try:
            values = metric.from_verdict(payload, context)
        except Exception as failure:
            logger.exception(f"{metric.id} on {context.unit_id}: {failure}")
            self._gap(result, metric, rubric, context, "error", str(failure))
            return

        result.rows.extend(
            metric.rows(context, values, note=f"judged_by={envelope.judged_by}")
        )

    def _gap(self, result, metric, rubric, context, reason, detail) -> None:
        result.rows.extend(empty_rows(metric, context, note=reason))
        result.gaps.append(
            {
                "metric": metric.id,
                "rubric": rubric,
                "unit_id": context.unit_id,
                "usecase": context.usecase_id,
                "ablation": context.ablation_id,
                "reason": reason,
                "detail": detail,
            }
        )
