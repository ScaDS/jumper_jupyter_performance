"""Whose verdict about a suggestion counts, decided in one place.

Every suggestion of a usecase with a ``verification`` block has been judged
twice. The reviewer compared statistical signatures of whatever the cell
happened to bind; ``cli.verify`` re-ran the same code in a fresh process and
compared the whole result the way the source paper does - ``allclose`` on
fresh inputs for KernelBench, the human fast solution's answers token by
token for ECCO. Where both exist the second is authoritative, and not
marginally so: on the ECCO pilot the reviewer's own check called 55
suggestions "could not be compared" that had in fact produced nothing at all,
and 19 others that were correct.

The timings stay the reviewer's throughout. It is the only instrument that
measures a variant against the cell under review, which is what a speedup in
this experiment means; ``cli.verify`` measures against the paper's reference
instead, and those are the ``reference_*`` metrics.

Where no external measurement exists - minian, robotics, the self-test, every
usecase without a ``verification`` block - the reviewer's verdict is all there
is, and these helpers fall back to it rather than reporting nothing.
"""

from __future__ import annotations

from jumper_ablations.metrics.verdicts import (
    is_correct,
    ordered_verdicts,
    ran,
)
from jumper_ablations.verification.schema import STATUS_UNMEASURABLE


def external_verdicts(context, record) -> dict | None:
    """``cli.verify``'s measurements for this review, by suggestion label.

    None when the cell was never verified, which is not the same as an empty
    dict: a verified cell whose review is missing from the verification
    directory is a gap worth falling back for, not an answer.
    """
    verification = getattr(context, "verification", None)
    if verification is None:
        return None
    review = verification.reviews.get(record.identity.reviewer_run_id)
    if review is None:
        return None
    return review.suggestions


def _external_for(context, record, verdict):
    external = external_verdicts(context, record)
    if not external:
        return None
    return external.get(verdict.label)


def is_correct_in(context, record, verdict) -> bool:
    """It ran under the reviewer, and the better of the two checks passed it."""
    if not ran(verdict):
        return False
    measurement = _external_for(context, record, verdict)
    if measurement is None:
        return is_correct(verdict)
    return bool(measurement.ok and measurement.correct)


def did_the_work(context, record, verdict) -> bool:
    """The variant ran *and* produced what the protocol asks a program for.

    The gate for every speedup reported without conditioning on correctness.
    A rewrite that defines a faster solver and never calls it runs in
    microseconds and divides into the baseline as millions of times faster;
    before this was applied, that is what the un-conditioned columns of the
    ECCO pilot averaged. ``cli.verify`` already knows: no ``answers`` for
    ECCO, no ``Model`` class for KernelBench, and it reports the protocol as
    inapplicable rather than as a crash. Such a run is dropped here - not for
    being wrong, which it was never shown to be, but for having measured
    nothing.
    """
    if not ran(verdict):
        return False
    measurement = _external_for(context, record, verdict)
    if measurement is None:
        return True
    if measurement.status == STATUS_UNMEASURABLE:
        return False
    # A failed run whose correctness is nonetheless False is the protocol
    # saying the program was not one: ECCO's "incomplete cell: it defines no
    # answers". A crash or a timeout leaves correctness None instead, and that
    # is a real program that failed in verify's environment, not here.
    return measurement.ok or measurement.correct is None


def correct_speedups_in(context, record) -> list[float]:
    return [
        verdict.speedup
        for verdict in ordered_verdicts(record)
        if is_correct_in(context, record, verdict) and verdict.speedup is not None
    ]


def measured_speedups_in(context, record) -> list[float]:
    return [
        verdict.speedup
        for verdict in ordered_verdicts(record)
        if did_the_work(context, record, verdict) and verdict.speedup is not None
    ]


def correct_but_untimed(context, record) -> int:
    """Correct by the paper's protocol, and never timed by the reviewer.

    Almost always a variant the reviewer killed on its own timeout budget,
    which is a multiple of the baseline rather than the protocol's limit.
    There is no speedup to pool for these - the measurement does not exist -
    so they cannot raise coverage, and leaving them unsaid would make coverage
    read as a finding instead of as a floor. Reported, so the gap is visible
    and can be closed by re-measuring with a wider budget.
    """
    external = external_verdicts(context, record) or {}
    count = 0
    for verdict in ordered_verdicts(record):
        if ran(verdict):
            continue
        measurement = external.get(verdict.label)
        if measurement is not None and measurement.ok and measurement.correct:
            count += 1
    return count
