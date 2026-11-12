from __future__ import annotations

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

_PROMPTS_DIR = Path(__file__).parent
_FRAGMENTS_DIR = _PROMPTS_DIR / "fragments"


class PromptLibrary:
    """Composes the AI-reviewer system prompts from per-prompt templates + specs.

    Each prompt has its own folder (``analyze``/``suggest``/``refine``) with a
    ``template.md`` (Jinja prose + loops over list items) and a ``spec.yaml``
    (the toggleable items, each ``{id, enabled, text|fragment}``).

    ``render`` takes an ``overrides`` map (``id -> enabled``) so any caller - a
    deterministic dialog or one an agent builds at runtime - can flip items
    on/off without this layer knowing where the choice came from, and a
    ``variables`` map that fills plain placeholders such as the number of
    options the suggest prompt asks for.
    """

    def __init__(self, specs: dict[str, dict], env: Environment):
        self._specs = specs
        self._env = env

    @classmethod
    def load(cls, root: Path = _PROMPTS_DIR) -> "PromptLibrary":
        env = Environment(
            loader=FileSystemLoader(str(root)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        specs = {
            prompt_dir.name: _resolve_fragments(
                yaml.safe_load((prompt_dir / "spec.yaml").read_text()) or {}
            )
            for prompt_dir in sorted(root.iterdir())
            if (prompt_dir / "template.md").exists()
        }
        return cls(specs, env)

    def prompt_ids(self) -> list[str]:
        return list(self._specs)

    def render(
        self,
        prompt_id: str,
        overrides: dict | None = None,
        note: str | None = None,
        variables: dict | None = None,
    ) -> str:
        overrides = overrides or {}
        context = {}
        for key, value in self._specs[prompt_id].items():
            if isinstance(value, list):
                context[key] = [
                    {**item, "enabled": overrides.get(item["id"], item["enabled"])}
                    for item in value
                ]
            else:
                context[key] = {**value, "enabled": bool(note), "text": note or ""}
        template = self._env.get_template(f"{prompt_id}/template.md")
        return template.render(**(variables or {}), **context).strip() + "\n"


def _resolve_fragments(spec: dict) -> dict:
    resolved = {}
    for key, value in spec.items():
        if isinstance(value, list):
            resolved[key] = [_resolve_item(item) for item in value]
        else:
            resolved[key] = _resolve_item(value)
    return resolved


def _resolve_item(item: dict) -> dict:
    if "fragment" in item:
        return {**item, "text": (_FRAGMENTS_DIR / item["fragment"]).read_text().strip()}
    return item
