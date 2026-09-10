"""Reading verdict files back, and refusing to guess what a missing one said.

Every verdict is validated twice: the envelope, and then the rubric's own
payload model. A file that does not validate is reported as a gap with the
reason, never coerced - a judged metric silently reading a malformed verdict
as zeros would be worse than having no verdict at all.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from jumper_ablations.evaluation.judge.layout import verdict_path, verdicts_root
from jumper_ablations.evaluation.judge.schemas import VerdictEnvelope

logger = logging.getLogger("jumper_ablations")


class VerdictProblem(Exception):
    """A verdict exists but cannot be used."""


def load_envelope(path: Path) -> VerdictEnvelope:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as failure:
        raise VerdictProblem(f"not valid JSON: {failure}") from failure
    try:
        return VerdictEnvelope.model_validate(payload)
    except ValidationError as failure:
        raise VerdictProblem(f"does not match the envelope: {failure}") from failure


def load_verdict(
    run_directory: Path,
    rubric: str,
    unit_id: str,
    verdict_model,
):
    """The parsed payload for one unit, or None when it has not been judged.

    Returns ``(payload, envelope)``. An abstention comes back as
    ``(None, envelope)`` so the caller can tell "declined" from "not yet".
    """
    path = verdict_path(run_directory, rubric, unit_id)
    if not path.is_file():
        return (None, None)

    envelope = load_envelope(path)
    if envelope.unit_id != unit_id:
        raise VerdictProblem(
            f"claims unit '{envelope.unit_id}' but is filed under '{unit_id}'"
        )
    if envelope.rubric != rubric:
        raise VerdictProblem(
            f"claims rubric '{envelope.rubric}' but is filed under '{rubric}'"
        )
    if envelope.abstained:
        return (None, envelope)

    try:
        return (verdict_model.model_validate(envelope.verdict), envelope)
    except ValidationError as failure:
        raise VerdictProblem(f"payload is not a valid verdict: {failure}") from failure


def load_verdicts(run_directory: Path, rubric: str) -> dict:
    """Every envelope filed under *rubric*, keyed by unit id."""
    directory = verdicts_root(run_directory) / rubric
    if not directory.is_dir():
        return {}
    found = {}
    for path in sorted(directory.glob("*.json")):
        try:
            envelope = load_envelope(path)
        except VerdictProblem as failure:
            logger.warning(f"{path}: {failure}")
            continue
        found[envelope.unit_id] = envelope
    return found
