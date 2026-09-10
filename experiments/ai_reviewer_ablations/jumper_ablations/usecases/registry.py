"""Discovering usecases, so adding one is dropping a directory in.

A usecase is a directory under ``usecases/`` holding ``notebook.ipynb`` and
``usecase.yaml``. Nothing else in the harness has to be edited to add one: the
suite config names it by id and the registry finds it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from jumper_ablations.paths import USECASES_DIR
from jumper_ablations.usecases.notebook import NotebookLayout, read_layout

MANIFEST_NAME = "usecase.yaml"
NOTEBOOK_NAME = "notebook.ipynb"


class ReferenceFact(BaseModel):
    """One fact a correct analysis is expected to contain.

    ``source`` names the single context source the fact can be read from,
    which is what makes conditional coverage - recall over the facts this
    ablation could still reach - a different number from global coverage.
    """

    id: str
    source: str
    fact: str
    weight: float = 1.0
    tolerance: float | None = None


class UsecaseBenchmark(BaseModel):
    replay_mode: str = "full"
    extra_replay_modes: list[str] = Field(default_factory=list)


class UsecaseManifest(BaseModel):
    """Everything about a usecase that the notebook cannot state itself."""

    id: str
    title: str
    payload_type: str
    description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    benchmark: UsecaseBenchmark = Field(default_factory=UsecaseBenchmark)
    reference_facts: list[ReferenceFact] = Field(default_factory=list)

    def facts_by_source(self) -> dict[str, list[ReferenceFact]]:
        grouped: dict[str, list[ReferenceFact]] = {}
        for fact in self.reference_facts:
            grouped.setdefault(fact.source, []).append(fact)
        return grouped

    def total_weight(self) -> float:
        return sum(fact.weight for fact in self.reference_facts)


@dataclasses.dataclass(frozen=True)
class Usecase:
    """A manifest, its notebook, and where the target sits inside it."""

    manifest: UsecaseManifest
    directory: Path
    notebook_path: Path
    layout: NotebookLayout

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def slug(self) -> str:
        """The id as one filesystem- and column-safe token."""
        return self.manifest.id.replace("/", "-")


def _load(directory: Path, root: Path) -> Usecase:
    manifest_path = directory / MANIFEST_NAME
    notebook_path = directory / NOTEBOOK_NAME
    if not notebook_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} has no {NOTEBOOK_NAME} beside it"
        )

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest = UsecaseManifest.model_validate(data)

    expected_id = directory.relative_to(root).as_posix()
    if manifest.id != expected_id:
        raise ValueError(
            f"{manifest_path} declares id '{manifest.id}' but sits at "
            f"'{expected_id}'; the path is the id"
        )

    return Usecase(
        manifest=manifest,
        directory=directory,
        notebook_path=notebook_path,
        layout=read_layout(notebook_path),
    )


def discover_usecases(root: Path = USECASES_DIR) -> dict[str, Usecase]:
    """Every usecase under *root*, keyed by id, in path order."""
    root = Path(root)
    usecases = {}
    for manifest_path in sorted(root.rglob(MANIFEST_NAME)):
        usecase = _load(manifest_path.parent, root)
        usecases[usecase.id] = usecase
    return usecases


def get_usecase(usecase_id: str, root: Path = USECASES_DIR) -> Usecase:
    """One usecase by id, with the available ids in the error when it is not."""
    directory = Path(root) / usecase_id
    if (directory / MANIFEST_NAME).is_file():
        return _load(directory, Path(root))
    available = ", ".join(discover_usecases(root)) or "none"
    raise KeyError(f"unknown usecase '{usecase_id}'; available: {available}")
