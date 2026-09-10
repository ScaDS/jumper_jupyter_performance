"""The statistics every layer shares.

Metrics reduce a cell of the grid to a number, aggregation puts an interval
around it, and both need the same handful of estimators. Keeping them here
rather than inside either layer is what stops a metric and the report from
computing "the median" two subtly different ways.

Two choices are deliberate. Speedups are averaged geometrically, because they
are ratios and an arithmetic mean of ratios is not a ratio anyone can act on.
Intervals are bootstrapped rather than assumed normal, because the samples are
small, bounded and visibly skewed - and because the experiment plan sizes its
own N from the width of these intervals.
"""
from __future__ import annotations

import math
import statistics as _statistics

import numpy as np

Number = float | int


def clean(values) -> list[float]:
    """Finite numbers only - None and NaN are absence, not zero."""
    kept = []
    for value in values or []:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            kept.append(number)
    return kept


def mean(values) -> float | None:
    numbers = clean(values)
    return _statistics.fmean(numbers) if numbers else None


def median(values) -> float | None:
    numbers = clean(values)
    return _statistics.median(numbers) if numbers else None


def geometric_mean(values) -> float | None:
    """The right average for a set of speedups.

    Values at or below zero are dropped rather than made to work: a
    non-positive speedup is a measurement that failed, and folding it in as a
    small number would quietly punish the preset that produced it.
    """
    positive = [value for value in clean(values) if value > 0]
    if not positive:
        return None
    return float(math.exp(_statistics.fmean(math.log(value) for value in positive)))


def rate(count: Number, total: Number) -> float | None:
    """A proportion, or None when there was nothing to take it over."""
    if not total:
        return None
    return float(count) / float(total)


def bootstrap_interval(
    values,
    statistic=mean,
    samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """A percentile bootstrap interval for *statistic* over *values*.

    Returns ``(None, None)`` for fewer than two values: an interval around a
    single observation would be a statement the data cannot support.
    """
    numbers = clean(values)
    if len(numbers) < 2:
        return (None, None)

    generator = np.random.default_rng(seed)
    array = np.asarray(numbers, dtype=float)
    draws = generator.integers(0, len(array), size=(samples, len(array)))
    estimates = [statistic(array[row].tolist()) for row in draws]
    estimates = clean(estimates)
    if not estimates:
        return (None, None)

    tail = (1.0 - confidence) / 2.0
    lower = float(np.percentile(estimates, 100.0 * tail))
    upper = float(np.percentile(estimates, 100.0 * (1.0 - tail)))
    return (lower, upper)


def spearman(first, second) -> float | None:
    """Rank correlation, or None when it would be undefined.

    Fewer than two pairs, or a constant series, has no rank correlation; the
    right answer there is "cannot say", not zero.
    """
    left = clean(first)
    right = clean(second)
    if len(left) != len(right) or len(left) < 2:
        return None
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    result = spearmanr(left, right)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def dispersion(values) -> float | None:
    """Max over min - how far apart the extremes of a set of ratios are."""
    positive = [value for value in clean(values) if value > 0]
    if len(positive) < 2:
        return None
    return max(positive) / min(positive)
