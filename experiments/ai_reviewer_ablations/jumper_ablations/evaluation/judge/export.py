"""Writing a task packet: everything a judgement needs, and nothing else.

A packet is self-contained by design. The session that judges it reads only
what is inside the directory - no repository, no database, no earlier packet -
so a judgement cannot quietly depend on context the reviewer did not have.

Two properties of the packet do the work:

**The sources are verbatim.** `sources/analyze.messages.json` is the message
list the reviewer sent, recorded by calling the same functions the graph
called. Not a summary of it, not a re-render.

**The packet is blind by default.** It does not name the ablation. A judge
who can see "no_timing" on the folder is scoring a label, and the whole
comparison between presets rests on that not happening. The mapping from unit
to preset lives in judge/index.csv and is rejoined at ingest.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import yaml

from jumper_ablations.config.schema import JudgeConfig
from jumper_ablations.evaluation.judge.layout import (
    index_path,
    packet_directory,
    verdict_path,
)
from jumper_ablations.evaluation.judge.rubrics import render_rubric
from jumper_ablations.metrics.base import JudgeMetric, SCOPE_RUN

logger = logging.getLogger("jumper_ablations")

_SOURCE_FILES = {
    "analyze_messages": "analyze.messages.json",
    "suggest_messages": "suggest.messages.json",
    "context_payload": "context_payload.json",
    "enabled_sources": "enabled_sources.json",
    "reference_facts": "reference_facts.yaml",
}


@dataclasses.dataclass
class ExportedPacket:
    rubric: str
    unit_id: str
    metrics: tuple[str, ...]
    directory: Path
    usecase: str
    ablation: str


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sources_for(context, config: JudgeConfig) -> dict:
    record = context.record
    available = {
        "analyze_messages": record.inputs.analyze_messages,
        "suggest_messages": record.inputs.suggest_messages,
        "context_payload": record.inputs.context_payload,
        "enabled_sources": record.inputs.enabled_sources,
        "reference_facts": [
            fact.model_dump()
            for fact in (
                context.usecase.manifest.reference_facts if context.usecase else []
            )
        ],
    }
    return {
        name: available[name]
        for name in config.sources
        if name in available
    }


def _outputs_for(context) -> dict:
    record = context.record
    return {
        "analysis.md": record.outputs.analysis,
        "analysis_reasoning.md": record.outputs.analysis_reasoning,
        "suggestions.json": [
            suggestion.model_dump() for suggestion in record.outputs.suggestions
        ],
    }


def _task_text(
    rubric: str,
    unit_id: str,
    metrics: tuple[str, ...],
    verdict_file: Path,
    blind: bool,
) -> str:
    """The instruction sheet an agent session reads first."""
    lines = [
        f"# Judging task: {rubric}",
        "",
        f"Unit: `{unit_id}`",
        f"Feeds metric(s): {', '.join(metrics)}",
        "",
        "## What this is",
        "",
        "`sources/` holds exactly what the AI reviewer's model was given for",
        "this cell - the message lists are verbatim, not a summary. `output/`",
        "holds what it answered. Judge the output against those sources and",
        "against nothing else: not your own knowledge of the library, not",
        "what you think the code probably does.",
        "",
    ]
    if blind:
        lines += [
            "The preset that produced this is deliberately not named. Do not",
            "try to infer it; scoring a label instead of an answer is the one",
            "failure this whole experiment cannot survive.",
            "",
        ]
    lines += [
        "## What to produce",
        "",
        "Read `rubric.md`, then write one JSON file validating against",
        "`verdict.schema.json` to:",
        "",
        f"    {verdict_file}",
        "",
        "Set `judged_by` to the model or person that produced the verdict and",
        "`judged_at` to an ISO timestamp - a manual judgement whose author is",
        "unknown cannot be checked for drift later.",
        "",
        "If the sources do not support a judgement, set `abstained` to true",
        "and say why in `reason`. An abstention is excluded from the metric.",
        "A guess is not.",
        "",
        "## Files",
        "",
        "- `rubric.md` - the scale, and what each value means",
        "- `verdict.schema.json` - the exact shape of the answer",
        "- `sources/` - what the reviewer's model was given",
        "- `output/` - what it answered",
    ]
    return "\n".join(lines) + "\n"


def export_packets(
    run_directory: Path,
    metrics: list[JudgeMetric],
    run_contexts: list,
    cell_contexts: list,
    config: JudgeConfig,
) -> list[ExportedPacket]:
    """Write one packet per (rubric, unit) and return what was written.

    Metrics that share a rubric share a packet: the two evidence-coverage
    metrics ask the judge one question and divide the answer two ways, so
    exporting it twice would double the manual work and invite the two copies
    to disagree.
    """
    run_directory = Path(run_directory)
    by_rubric: dict[str, list[JudgeMetric]] = {}
    for metric in metrics:
        by_rubric.setdefault(metric.rubric_id(), []).append(metric)

    exported: list[ExportedPacket] = []
    for rubric, rubric_metrics in sorted(by_rubric.items()):
        scopes = {metric.spec.scope for metric in rubric_metrics}
        if len(scopes) > 1:
            raise ValueError(
                f"rubric '{rubric}' is shared by metrics of different scopes: "
                f"{sorted(scopes)}"
            )
        contexts = run_contexts if scopes == {SCOPE_RUN} else cell_contexts
        for context in _sampled(contexts, config):
            exported.append(
                _write_packet(
                    run_directory=run_directory,
                    rubric=rubric,
                    metrics=rubric_metrics,
                    context=context,
                    config=config,
                )
            )

    _write_index(run_directory, exported)
    return exported


def _sampled(contexts: list, config: JudgeConfig) -> list:
    """At most `sample_per_cell` units per grid cell, earliest first.

    Manual judging does not scale to every generation of a full sweep. Taking
    the first N of each cell rather than a random N keeps the sample paired
    across presets: generation 1 of every preset, then generation 2.
    """
    if config.sample_per_cell is None:
        return list(contexts)

    taken: dict[tuple[str, str], int] = {}
    selected = []
    for context in contexts:
        key = (context.usecase_id, context.ablation_id)
        if taken.get(key, 0) >= config.sample_per_cell:
            continue
        taken[key] = taken.get(key, 0) + 1
        selected.append(context)
    return selected


def _write_packet(
    run_directory: Path,
    rubric: str,
    metrics: list[JudgeMetric],
    context,
    config: JudgeConfig,
) -> ExportedPacket:
    directory = packet_directory(run_directory, rubric, context.unit_id)
    (directory / "sources").mkdir(parents=True, exist_ok=True)
    (directory / "output").mkdir(parents=True, exist_ok=True)

    for name, payload in _sources_for(context, config).items():
        filename = _SOURCE_FILES[name]
        path = directory / "sources" / filename
        if filename.endswith(".yaml"):
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        else:
            _write_json(path, payload)

    for filename, payload in _outputs_for(context).items():
        path = directory / "output" / filename
        if filename.endswith(".json"):
            _write_json(path, payload)
        else:
            path.write_text(str(payload or ""), encoding="utf-8")

    (directory / "rubric.md").write_text(
        render_rubric(rubric, context),
        encoding="utf-8",
    )
    _write_json(
        directory / "verdict.schema.json",
        _schema_for(rubric, metrics[0]),
    )
    (directory / "TASK.md").write_text(
        _task_text(
            rubric=rubric,
            unit_id=context.unit_id,
            metrics=tuple(metric.id for metric in metrics),
            verdict_file=verdict_path(run_directory, rubric, context.unit_id),
            blind=config.blind,
        ),
        encoding="utf-8",
    )

    return ExportedPacket(
        rubric=rubric,
        unit_id=context.unit_id,
        metrics=tuple(metric.id for metric in metrics),
        directory=directory,
        usecase=context.usecase_id,
        ablation=context.ablation_id,
    )


def _schema_for(rubric: str, metric: JudgeMetric) -> dict:
    """The envelope schema with this rubric's payload inlined."""
    from jumper_ablations.evaluation.judge.schemas import VerdictEnvelope

    envelope = VerdictEnvelope.model_json_schema()
    envelope["properties"]["verdict"] = metric.verdict_model.model_json_schema()
    envelope["properties"]["rubric"]["const"] = rubric
    envelope["required"] = ["unit_id", "rubric", "verdict"]
    return envelope


def _write_index(run_directory: Path, exported: list[ExportedPacket]) -> Path:
    """The unblinding key: which preset each unit actually came from."""
    import pandas as pd

    path = index_path(run_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "rubric": packet.rubric,
                "unit_id": packet.unit_id,
                "metrics": " ".join(packet.metrics),
                "usecase": packet.usecase,
                "ablation": packet.ablation,
                "packet": str(packet.directory),
            }
            for packet in exported
        ],
        columns=[
            "rubric",
            "unit_id",
            "metrics",
            "usecase",
            "ablation",
            "packet",
        ],
    )
    frame.to_csv(path, index=False)
    return path
