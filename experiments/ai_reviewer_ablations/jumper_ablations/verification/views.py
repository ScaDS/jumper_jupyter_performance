"""Reading cli.verify's measurements from a cell, the same way in every metric.

A cell can be a bootstrap resample, so the suggestions are collected per
review the cell actually holds rather than per file on disk: a review drawn
twice counts twice, one not drawn does not count.
"""

from __future__ import annotations

from jumper_ablations.verification.schema import Measurement, References


def references_of(context, kind: str) -> References | None:
    """The cell's references, or None when this cell was not verified as *kind*."""
    verification = context.verification
    if verification is None or verification.references.kind != kind:
        return None
    return verification.references


def verified_suggestions(context) -> list[Measurement]:
    """Every verified suggestion of the cell's reviews, in review order."""
    verification = context.verification
    if verification is None:
        return []
    measurements = []
    for record in context.reviews():
        review = verification.reviews.get(record.identity.reviewer_run_id)
        if review is None:
            continue
        measurements.extend(
            review.suggestions[label]
            for label in sorted(review.suggestions, key=int)
        )
    return measurements


def correct_suggestions(context) -> list[Measurement]:
    return [
        measurement
        for measurement in verified_suggestions(context)
        if measurement.ok and measurement.correct
    ]


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    """``numerator / denominator``, or None when either side is missing."""
    if numerator is None or not denominator:
        return None
    return numerator / denominator
