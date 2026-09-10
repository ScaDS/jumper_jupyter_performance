"""The judged path: export what the model saw, read back what a session ruled.

No API client lives here, deliberately. Judging is done by an agent session
following JUDGE_PROTOCOL.md, so this package has two halves that never meet:
`export` writes a self-contained task packet per unit, and `ingest` validates
the verdict files that come back. Between them sits a person and whichever
model they chose.

The packets carry the reviewer's own messages, verbatim, rather than a
reconstruction of them. That is the only basis on which a judgement about "the
sources the model had" can be claimed to be about the sources the model had.
"""
from jumper_ablations.evaluation.judge.export import export_packets
from jumper_ablations.evaluation.judge.ingest import load_verdicts
from jumper_ablations.evaluation.judge.method import JudgeEvaluation
from jumper_ablations.evaluation.judge.schemas import VerdictEnvelope

__all__ = [
    "JudgeEvaluation",
    "VerdictEnvelope",
    "export_packets",
    "load_verdicts",
]
