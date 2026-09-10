"""Scoring the records - two paths, one interface.

A metric declares which path scores it and nothing else about it. One path
computes from the records; the other exports what the model saw and reads back
what an agent session concluded. `cli.evaluate` groups the enabled metrics by
path, hands each group to its method, and merges the rows: from there on a
judged number and a computed one are the same kind of row, and the report does
not distinguish them.

Adding a third path - a second judge, a static analyser, a human panel - is a
subpackage with an `EvaluationMethod` and a new value for
`MetricSpec.evaluation_method`.
"""
from jumper_ablations.evaluation.base import EvaluationMethod, EvaluationResult

__all__ = ["EvaluationMethod", "EvaluationResult"]
