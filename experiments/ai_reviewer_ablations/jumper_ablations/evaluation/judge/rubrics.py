"""Rendering the rubric a packet carries.

Each rubric is a Jinja template beside this module, so writing a new judged
metric is writing prose, not code. The template is handed the unit it is about
- the usecase's reference facts, which sources were enabled - because a rubric
that had to be read alongside a separate file would be one more thing a
session could skip.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

RUBRICS_DIR = Path(__file__).parent / "rubrics"
TEMPLATE_NAME = "template.md"


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(RUBRICS_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )


def available_rubrics() -> list[str]:
    return sorted(
        directory.name
        for directory in RUBRICS_DIR.iterdir()
        if (directory / TEMPLATE_NAME).is_file()
    )


def render_rubric(rubric: str, context) -> str:
    """The rubric text for one unit.

    A missing rubric is an error rather than a blank page: a packet without a
    scale would be judged on whatever the session felt like, which is exactly
    the variance the experiment is trying to keep out.
    """
    if rubric not in available_rubrics():
        raise FileNotFoundError(
            f"no rubric '{rubric}' under {RUBRICS_DIR}; "
            f"available: {', '.join(available_rubrics()) or 'none'}"
        )
    template = _environment().get_template(f"{rubric}/{TEMPLATE_NAME}")
    return template.render(**_template_context(context)).strip() + "\n"


def _template_context(context) -> dict:
    """What every rubric may refer to."""
    record = getattr(context, "record", None)
    usecase = getattr(context, "usecase", None)
    facts = list(usecase.manifest.reference_facts) if usecase else []
    enabled = dict(record.inputs.enabled_sources) if record else {}
    return {
        "unit_id": context.unit_id,
        "payload_type": usecase.manifest.payload_type if usecase else "",
        "usecase_title": usecase.manifest.title if usecase else "",
        "reference_facts": facts,
        "reachable_facts": [
            fact for fact in facts if enabled.get(fact.source, False)
        ],
        "enabled_sources": sorted(
            source for source, is_on in enabled.items() if is_on
        ),
        "disabled_sources": sorted(
            source for source, is_on in enabled.items() if not is_on
        ),
        "suggestion_count": len(record.outputs.suggestions) if record else 0,
    }
