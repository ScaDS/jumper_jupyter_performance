"""The computed path: arithmetic over the records, no model in the loop.

Everything here is reproducible from the records alone. Re-running it after
adding a metric costs a second and touches nothing the expensive phase
produced, which is the reason generating and scoring are separate commands.

A metric that raises is reported as a gap rather than taking the run down: one
broken metric should not cost the other nineteen.
"""
from __future__ import annotations

import logging

from jumper_ablations.evaluation.base import (
    EvaluationMethod,
    EvaluationResult,
    contexts_for,
    empty_rows,
)
from jumper_ablations.metrics.base import METHOD_DETERMINISTIC, Metric

logger = logging.getLogger("jumper_ablations")


class DeterministicEvaluation(EvaluationMethod):
    id = METHOD_DETERMINISTIC

    def evaluate(
        self,
        metrics: list[Metric],
        run_contexts: list,
        cell_contexts: list,
        parameters: dict,
    ) -> EvaluationResult:
        result = EvaluationResult()
        for metric in metrics:
            metric_parameters = parameters.get(metric.id, {})
            for context in contexts_for(metric, run_contexts, cell_contexts):
                try:
                    values = metric.compute(context, metric_parameters)
                except Exception as failure:
                    logger.exception(
                        f"{metric.id} failed on {context.unit_id}: {failure}"
                    )
                    result.rows.extend(
                        empty_rows(metric, context, f"error: {failure}")
                    )
                    result.gaps.append(
                        {
                            "metric": metric.id,
                            "unit_id": context.unit_id,
                            "usecase": context.usecase_id,
                            "ablation": context.ablation_id,
                            "reason": "error",
                            "detail": str(failure),
                        }
                    )
                    continue
                result.rows.extend(metric.rows(context, values))
        return result
