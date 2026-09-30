"""Discovering usecases, so adding one is dropping a notebook in.

A usecase is a notebook under ``usecases/``. Its id is its path without the
extension - ``minian/cell_40``, ``robotics/vae_model_training`` - so the
directory tree is the naming scheme and nothing has to be registered.

The manifest beside it is **optional**. Without one the harness still runs the
notebook: the title and payload type come from the notebook's own header, and
the benchmark falls back to the protocol's defaults. What a manifest adds is
the thing a notebook genuinely cannot state - the reference facts a correct
analysis is measured against - plus per-usecase benchmark overrides. A usecase
with no facts is measurable on its suggestions and abstains on evidence
coverage, which is the honest outcome rather than a zero.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from jumper_ablations.paths import USECASES_DIR
from jumper_ablations.usecases.notebook import (
    NotebookLayout,
    payload_type_of,
    read_layout,
    read_notebook,
    title_of,
)

NOTEBOOK_SUFFIX = ".ipynb"
MANIFEST_SUFFIX = ".yaml"

# Checkpoints, exports and the like: anything under a dot directory or an
# ipynb checkpoint folder is not a usecase.
_IGNORED_PARTS = (".ipynb_checkpoints",)


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
    title: str = ""
    payload_type: str = ""
    description: str = ""
    preconditions: list[str] = Field(default_factory=list)
    # Environment the notebook needs, exported into the kernel and inherited
    # by the benchmark's replay children. This is how a usecase pins an
    # absolute path to data it reads: a replay runs in a fresh temporary
    # directory, so anything the notebook resolves relative to the working
    # directory is simply not there when it is measured.
    #
    # ${workspace} expands to the directory the notebook is executed in.
    environment: dict = Field(default_factory=dict)
    benchmark: UsecaseBenchmark = Field(default_factory=UsecaseBenchmark)
    reference_facts: list[ReferenceFact] = Field(default_factory=list)

    def facts_by_source(self) -> dict[str, list[ReferenceFact]]:
        grouped: dict[str, list[ReferenceFact]] = {}
        for fact in self.reference_facts:
            grouped.setdefault(fact.source, []).append(fact)
        return grouped

    def total_weight(self) -> float:
        return sum(fact.weight for fact in self.reference_facts)

    def resolved_environment(self, workspace: Path) -> dict:
        """The declared environment with ${workspace} filled in."""
        return {
            name: str(value).replace("${workspace}", str(workspace))
            for name, value in self.environment.items()
        }


@dataclasses.dataclass(frozen=True)
class Usecase:
    """A notebook, where the target sits inside it, and what it is about."""

    manifest: UsecaseManifest
    notebook_path: Path
    layout: NotebookLayout
    manifest_path: Path | None = None

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def directory(self) -> Path:
        return self.notebook_path.parent

    @property
    def slug(self) -> str:
        """The id as one filesystem- and column-safe token."""
        return self.manifest.id.replace("/", "-")

    @property
    def has_reference_facts(self) -> bool:
        return bool(self.manifest.reference_facts)


def _identifier(notebook_path: Path, root: Path) -> str:
    return notebook_path.relative_to(root).with_suffix("").as_posix()


def _load(notebook_path: Path, root: Path) -> Usecase:
    identifier = _identifier(notebook_path, root)
    manifest_path = notebook_path.with_suffix(MANIFEST_SUFFIX)
    notebook = read_notebook(notebook_path)

    data = {}
    if manifest_path.is_file():
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        declared = data.get("id")
        if declared and declared != identifier:
            raise ValueError(
                f"{manifest_path} declares id '{declared}' but sits at "
                f"'{identifier}'; the path is the id"
            )

    data.setdefault("id", identifier)
    data.setdefault("title", title_of(notebook) or notebook_path.stem)
    data.setdefault("payload_type", payload_type_of(notebook))

    return Usecase(
        manifest=UsecaseManifest.model_validate(data),
        notebook_path=notebook_path,
        layout=read_layout(notebook_path),
        manifest_path=manifest_path if manifest_path.is_file() else None,
    )


def _is_candidate(path: Path) -> bool:
    return not any(
        part.startswith(".") or part in _IGNORED_PARTS for part in path.parts
    )


def discover_usecases(root: Path = USECASES_DIR) -> dict[str, Usecase]:
    """Every usecase notebook under *root*, keyed by id, in path order."""
    root = Path(root)
    usecases = {}
    for notebook_path in sorted(root.rglob(f"*{NOTEBOOK_SUFFIX}")):
        if not _is_candidate(notebook_path.relative_to(root)):
            continue
        usecase = _load(notebook_path, root)
        usecases[usecase.id] = usecase
    return usecases


def get_usecase(usecase_id: str, root: Path = USECASES_DIR) -> Usecase:
    """One usecase by id; a wrong id lists the ones that exist."""
    root = Path(root)
    notebook_path = root / f"{usecase_id}{NOTEBOOK_SUFFIX}"
    if notebook_path.is_file():
        return _load(notebook_path, root)
    available = ", ".join(discover_usecases(root)) or "none"
    raise KeyError(f"unknown usecase '{usecase_id}'; available: {available}")
