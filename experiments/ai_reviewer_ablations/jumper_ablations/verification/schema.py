"""What cli.verify writes, and what the paper metrics read back."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
# The suggestion no longer defines what the paper's protocol calls: no
# `Model` class for KernelBench, no `answers` for ECCO.
STATUS_UNMEASURABLE = "unmeasurable"

# Reference labels.
EAGER = "eager"
TORCH_COMPILE = "torch_compile"
ORIGINAL = "original"
HUMAN_FAST = "human_fast"

VERIFICATION_DIRECTORY = "verification"
REFERENCES_NAME = "references.json"


class Measurement(BaseModel):
    """One program or model, measured once under the paper's protocol."""

    label: str
    status: str
    time_s: float | None = None
    peak_memory_bytes: int | None = None
    # None when the protocol had nothing to compare against.
    correct: bool | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.time_s is not None


class References(BaseModel):
    """Everything a suggestion of one usecase is compared against.

    ``population`` is the CodeNet submissions an ECCO percentile ranks
    against; it is empty for KernelBench, as is ``expected_answers``.
    """

    usecase: str
    kind: str
    entries: dict[str, Measurement] = Field(default_factory=dict)
    population: list[Measurement] = Field(default_factory=list)
    # ECCO only: what the human fast solution printed, token by token, per
    # scaled input. Correctness of everything else is equality with this.
    expected_answers: list[list[str]] = Field(default_factory=list)


class Verification(BaseModel):
    """The suggestions of one review, keyed by their 1-based index."""

    usecase: str
    ablation: str
    reviewer_run_id: str
    suggestions: dict[str, Measurement] = Field(default_factory=dict)


class UsecaseVerification(BaseModel):
    """One usecase's references and every verified review, as metrics see it."""

    references: References
    reviews: dict[str, Verification] = Field(default_factory=dict)


def usecase_directory(run_path: Path, usecase_id: str) -> Path:
    return Path(run_path) / VERIFICATION_DIRECTORY / usecase_id.replace("/", "-")


def write_model(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def load_verification(run_path: Path) -> dict[str, UsecaseVerification]:
    """Every usecase cli.verify has written for this run, keyed by usecase id."""
    root = Path(run_path) / VERIFICATION_DIRECTORY
    if not root.is_dir():
        return {}
    loaded = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        references_path = directory / REFERENCES_NAME
        if not references_path.is_file():
            continue
        references = References.model_validate_json(
            references_path.read_text(encoding="utf-8")
        )
        reviews = {}
        for path in sorted(directory.glob("*.json")):
            if path.name == REFERENCES_NAME:
                continue
            review = Verification.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            reviews[review.reviewer_run_id] = review
        loaded[references.usecase] = UsecaseVerification(
            references=references,
            reviews=reviews,
        )
    return loaded
