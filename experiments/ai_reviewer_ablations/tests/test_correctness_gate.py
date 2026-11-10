"""Which of the two verdicts a metric listens to, and what that changes.

Every suggestion of a verified usecase is judged twice: by the reviewer's
fingerprint, inside the kernel, and by cli.verify afterwards under the source
paper's protocol. These tests pin the rule that the second one wins where it
exists - including the three cases the ECCO pilot actually produced, where the
two disagreed in both directions.
"""

from __future__ import annotations

import pytest

from jumper_ablations.metrics import build_cell_contexts, get_metric
from jumper_ablations.verification.schema import (
    HUMAN_FAST,
    ORIGINAL,
    Measurement,
    References,
    UsecaseVerification,
    Verification,
)
from tests.factories import make_cell

USECASE = "synthetic/loop"


def _references() -> References:
    return References(
        usecase=USECASE,
        kind="ecco",
        entries={
            ORIGINAL: Measurement(label=ORIGINAL, status="ok", time_s=20.0, correct=True),
            HUMAN_FAST: Measurement(label=HUMAN_FAST, status="ok", time_s=2.0, correct=True),
        },
        expected_answers=[["7"]],
    )


def _verified(verdicts: list[dict], suggestions: list[dict]):
    """One generation: the reviewer's verdicts, and verify's measurements.

    *verdicts* is what the reviewer concluded, *suggestions* what cli.verify
    measured for the same code, index by index.
    """
    reviews = {
        "run01": Verification(
            usecase=USECASE,
            ablation="base",
            reviewer_run_id="run01",
            suggestions={
                str(index): Measurement(label=str(index), **measurement)
                for index, measurement in enumerate(suggestions, 1)
            },
        )
    }
    return build_cell_contexts(
        make_cell(generations=1, verdicts=verdicts),
        verifications={
            USECASE: UsecaseVerification(references=_references(), reviews=reviews)
        },
    )[0]


def _unverified_cell():
    """The same records with nothing for them in the verification directory."""
    return build_cell_contexts(
        make_cell(generations=1, verdicts=[{"speedup": 4.0, "correctness": "unverified"}])
    )[0]


def _values(metric_id: str, context) -> dict:
    return get_metric(metric_id).compute(context, {})


# --- the external verdict decides, in both directions ---

def test_an_unverified_suggestion_that_verify_passed_counts_as_correct():
    # The ECCO pilot's nineteen: the rewrite moved a constant inside a
    # function, so the reviewer had a binding missing and called the whole
    # thing uncomparable. The answers themselves were right.
    cell = _verified(
        verdicts=[{"speedup": 4.0, "correctness": "unverified"}],
        suggestions=[{"status": "ok", "time_s": 5.0, "correct": True}],
    )

    values = _values("correctness_conditioned_speedup", cell)

    assert values["correct_suggestions"] == 1
    assert values["coverage"] == pytest.approx(1.0)
    assert values["geometric_mean_speedup"] == pytest.approx(4.0)


def test_a_matching_suggestion_that_verify_rejected_is_not_correct():
    # The other direction, and the reason the external verdict has to win
    # rather than merely break ties: five statistics can agree where the whole
    # output does not.
    cell = _verified(
        verdicts=[{"speedup": 4.0, "correctness": "match"}],
        suggestions=[{"status": "ok", "time_s": 5.0, "correct": False}],
    )

    values = _values("correctness_conditioned_speedup", cell)

    assert values["correct_suggestions"] == 0
    assert values["coverage"] == pytest.approx(0.0)


def test_without_a_verification_the_reviewer_decides():
    # minian, robotics, the self-test: no external measurement exists, so the
    # fingerprint is all there is and `unverified` is not correct.
    values = _values("correctness_conditioned_speedup", _unverified_cell())

    assert values["correct_suggestions"] == 0
    assert values["suggestions"] == 1


# --- a run that measured nothing is not a measurement ---

def test_a_cell_that_produced_no_answers_is_out_of_the_measured_speedups():
    # The 30,000,000x "speedup": the rewrite defined a faster solver, never
    # called it, and the timer measured an empty cell. verify reports the
    # protocol as inapplicable - no answers - and that is enough to keep the
    # number out of the un-conditioned columns.
    cell = _verified(
        verdicts=[{"speedup": 30_000_000.0, "correctness": "unverified"}],
        suggestions=[
            {
                "status": "failed",
                "correct": False,
                "error": "incomplete cell: it defines no answers",
            }
        ],
    )

    values = _values("mean_median_speedup", cell)

    assert values["measured_suggestions"] == 0
    assert values["mean_speedup"] is None


def test_a_crash_inside_verify_does_not_discard_the_reviewer_s_measurement():
    # A program that verify could not run - a missing import in its
    # environment, a timeout at the protocol's limit - did run here, and its
    # timing is real. Only the correctness claim is withheld.
    cell = _verified(
        verdicts=[{"speedup": 3.0, "correctness": "match"}],
        suggestions=[
            {"status": "failed", "correct": None, "error": "NameError: joblib"}
        ],
    )

    measured = _values("mean_median_speedup", cell)
    conditioned = _values("correctness_conditioned_speedup", cell)

    assert measured["measured_suggestions"] == 1
    assert measured["mean_speedup"] == pytest.approx(3.0)
    assert conditioned["correct_suggestions"] == 0


# --- the deficit the join cannot close ---

def test_a_suggestion_verify_passed_but_the_reviewer_never_timed_is_reported():
    # The pilot's seventeen: killed by the reviewer's own timeout budget, which
    # is a multiple of the baseline rather than the protocol's limit. There is
    # no speedup to pool, so coverage cannot count them - but saying nothing
    # would make coverage read as a finding instead of as a floor.
    cell = _verified(
        verdicts=[
            {
                "status": "failed",
                "duration_s": None,
                "speedup": None,
                "correctness": "",
                "error": "Exceeded the 95s budget and was killed.",
            }
        ],
        suggestions=[{"status": "ok", "time_s": 5.0, "correct": True}],
    )

    values = _values("correctness_conditioned_speedup", cell)

    assert values["correct_suggestions"] == 0
    assert values["correct_but_untimed"] == 1


def test_pass_at_k_follows_the_same_gate():
    cell = _verified(
        verdicts=[
            {"speedup": 4.0, "correctness": "unverified"},
            {"speedup": 2.0, "correctness": "match"},
        ],
        suggestions=[
            {"status": "ok", "time_s": 5.0, "correct": True},
            {"status": "ok", "time_s": 9.0, "correct": False},
        ],
    )

    values = _values("pass_at_k", cell)

    # The option the model ranked first is the one verify passed.
    assert values["pass_at_1"] == pytest.approx(1.0)
    assert values["pass_at_k"] == pytest.approx(1.0)
