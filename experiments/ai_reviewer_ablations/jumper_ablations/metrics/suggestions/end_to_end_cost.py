"""What a review costs: model latency, tokens, and machine time.

An ablation that removes a source removes tokens with it, so a preset can look
cheaper for reasons that have nothing to do with being better. Reporting cost
next to quality is what keeps "shorter context" from reading as an
improvement on its own.

The token counts come from the callback the harness attaches to the reviewer's
model client; the reviewer keeps no account of its own. When the endpoint does
not return usage, these are None rather than zero.
"""
from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.statistics import mean


class EndToEndCost(DeterministicMetric):
    spec = MetricSpec(
        id="end_to_end_cost",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "mean_review_latency_s",
            "mean_input_tokens",
            "mean_output_tokens",
            "mean_total_tokens",
            "mean_benchmark_wall_s",
            "mean_repair_calls",
        ),
        description=(
            "Per generation: model latency and token usage of the review, the "
            "wall time of the benchmark, and how many repair calls it took."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        latencies = []
        input_tokens = []
        output_tokens = []
        total_tokens = []
        for record in context.reviews():
            latencies.append(record.cost.llm_latency_s)
            input_tokens.append(
                _sum_of(call.input_tokens for call in record.cost.llm_calls)
            )
            output_tokens.append(
                _sum_of(call.output_tokens for call in record.cost.llm_calls)
            )
            total_tokens.append(record.cost.total_tokens)

        benchmark_wall = []
        repair_calls = []
        for record in context.primary_benchmarks():
            benchmark_wall.append(record.cost.command_wall_s)
            repair_calls.append(
                sum(1 for call in record.cost.llm_calls if call.step == "fix")
            )

        return {
            "mean_review_latency_s": mean(latencies),
            "mean_input_tokens": mean(input_tokens),
            "mean_output_tokens": mean(output_tokens),
            "mean_total_tokens": mean(total_tokens),
            "mean_benchmark_wall_s": mean(benchmark_wall),
            "mean_repair_calls": mean(repair_calls),
        }


def _sum_of(values) -> int | None:
    """Sum, or None when the endpoint reported nothing at all."""
    present = [value for value in values if value is not None]
    return sum(present) if present else None
