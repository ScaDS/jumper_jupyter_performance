"""The envelope every verdict file shares.

A verdict is written by an agent session, by hand or by a script, and read
back here. The envelope carries the things the harness needs regardless of
which metric it is - which unit it answers, which rubric it followed, and who
produced it - and leaves the metric-specific payload to the metric.

``judged_by`` is not bookkeeping. The judging is a manual session, so judge
drift and an ablation effect look identical in the numbers unless the run that
produced each verdict is on record and a sample is re-judged.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class VerdictEnvelope(BaseModel):
    """One judged unit, as it is written to disk."""

    unit_id: str
    rubric: str
    # The model or person that produced this verdict, and when. Free text on
    # purpose: it has to survive being filled in by whatever did the judging.
    judged_by: str = ""
    judged_at: str = ""
    # An honest refusal is a result. A unit whose sources do not support a
    # judgement is excluded from the metric rather than scored as a zero.
    abstained: bool = False
    reason: str = ""
    verdict: dict = Field(default_factory=dict)


class MissingVerdict(BaseModel):
    """A unit that was exported but has not come back."""

    unit_id: str
    rubric: str
    metric: str
    usecase: str
    ablation: str
    packet: str
