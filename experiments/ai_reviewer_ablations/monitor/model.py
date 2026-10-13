"""A run directory, read into one plain dictionary.

Everything here is a pure function of what is on disk. No HTTP, no widgets,
no framework: the monitor's whole job is to describe a run it must not be
able to disturb, and keeping the reading separate from the serving is what
makes that checkable - this module can be pointed at a finished run in a
test and its answer asserted against a dictionary.

The snapshot is assembled fresh on every request. A run is hours long and
written by several jobs at once, so there is no state worth caching and a
stale cache would be worse than a slow read.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

# The seven ids the reviewer's collector actually gates. Ordered for display,
# not alphabetically: the cell source first, then what was measured about it,
# then what the machine and the environment contributed.
SOURCE_IDS = (
    "code",
    "timing",
    "tags",
    "perf",
    "raw_perf",
    "hardware",
    "packages",
)

STATE_OK = "ok"
STATE_RUNNING = "running"
STATE_PENDING = "pending"
STATE_LOST = "lost"

_RECORD_NAME = re.compile(
    r"^(?P<usecase>.+?)__(?P<ablation>.+?)__r(?P<repetition>\d+)"
    r"__g(?P<generation>\d+)__(?P<phase>[^_]+)__(?P<mode>.+)$"
)


# The layout this reader understands. A run stamped higher was written by a
# newer harness: it is still shown, with a warning, because refusing to draw
# a running sweep is worse than drawing part of it.
SUPPORTED_SCHEMA = 1


def as_int(value, default: int = 0) -> int:
    """An integer, whatever was actually in the file.

    Every number the monitor reads came out of somebody else's JSON, and a
    run must be viewable even when a field it did not expect holds a string,
    a null, or a list. Guessing zero and carrying on beats a stack trace in
    place of the page.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def as_sequence(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value)
    return []


def _load_json(path: Path, default=None):
    """Parse a JSON file, or return *default* if it is absent or half-written.

    A run is being written while it is being watched, so a truncated read is
    a normal event rather than an error: the next refresh will see the whole
    file. Reporting nothing beats crashing the page.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _load_csv(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def slug(usecase_id: str) -> str:
    """The usecase id as it appears in a record's file name."""
    return usecase_id.replace("/", "-")


def parse_record_name(name: str) -> dict | None:
    """Pull the grid position out of a record's file name.

    The name carries usecase, ablation, repetition, generation, phase and
    replay mode, which is enough to place a record in the grid without
    opening it - and opening several hundred files is what a refresh should
    avoid doing before it knows it has to.
    """
    match = _RECORD_NAME.match(name)
    if match is None:
        return None
    return {
        "usecase_slug": match["usecase"],
        "ablation": match["ablation"],
        "repetition": int(match["repetition"]),
        "generation": int(match["generation"]),
        "phase": match["phase"],
        "replay_mode": match["mode"],
    }


def read_meta(run: Path) -> dict:
    """The run's own definition, written once by its first invocation."""
    return _load_json(run / "meta.json", {}) or {}


def read_pass_entries(run: Path) -> list[dict]:
    """Finished passes, from every shard's index.

    Read together because the shards of one attempt need not be the shards
    of the next; the legacy single index is included for runs written before
    the split.
    """
    entries: list[dict] = []
    indexes = sorted(run.glob("passes-*.jsonl"))
    legacy = run / "passes.jsonl"
    if legacy.is_file():
        indexes.insert(0, legacy)
    for index in indexes:
        try:
            lines = index.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            entry = _load_json_line(line)
            if entry is not None:
                entries.append(entry)
    return entries


def _load_json_line(line: str) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # The last line of an index being appended to right now.
        return None


def read_invocations(run: Path) -> list[dict]:
    """Every job that has touched this run, oldest first."""
    directory = run / "invocations"
    if not directory.is_dir():
        return []
    entries = []
    for path in sorted(directory.glob("*.json")):
        entry = _load_json(path, None)
        if entry is not None:
            entries.append(entry)
    return entries


def read_records(run: Path) -> list[dict]:
    """Every record, opened.

    The file name alone would place a record in the grid, but the cost, the
    warnings and whether the reviewer was handed an empty context are only
    inside. Those are the numbers a person watches a sweep for, so the whole
    file is read - once per refresh, on the server, never in the browser.
    """
    directory = run / "records"
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        body = _load_json(path, None)
        if body is None:
            continue
        records.append(body)
    return records


def summarise_record(record: dict) -> dict:
    """What the monitor shows about one record.

    Deliberately not the record itself: it carries the verbatim messages
    sent to the model, which run to hundreds of kilobytes each. The browser
    gets the shape of the thing and fetches the body only when asked.
    """
    record = as_mapping(record)
    identity = as_mapping(record.get("identity"))
    cost = as_mapping(record.get("cost"))
    outputs = as_mapping(record.get("outputs"))
    environment = as_mapping(record.get("environment"))
    warnings = [
        one
        for one in as_sequence(environment.get("warnings"))
        if isinstance(one, str)
    ]
    return {
        "record_id": identity.get("record_id", ""),
        "usecase": identity.get("usecase", ""),
        "ablation": identity.get("ablation", ""),
        "repetition": as_int(identity.get("repetition")),
        "generation": as_int(identity.get("generation")),
        "phase": identity.get("phase", ""),
        "reviewer_run_id": identity.get("reviewer_run_id", ""),
        "created_at": identity.get("created_at", ""),
        "requested_replay_mode": identity.get("requested_replay_mode", ""),
        "actual_replay_mode": environment.get("actual_replay_mode", ""),
        "degraded": bool(environment.get("degraded")),
        "suggestions": len(as_sequence(outputs.get("suggestions"))),
        "benchmarks": len(as_mapping(outputs.get("benchmarks"))),
        "analysis_chars": len(str(outputs.get("analysis") or "")),
        "llm_calls": len(as_sequence(cost.get("llm_calls"))),
        "llm_latency_s": cost.get("llm_latency_s"),
        "total_tokens": cost.get("total_tokens"),
        "command_wall_s": cost.get("command_wall_s"),
        "warnings": warnings,
        "empty_context": any(
            warning.startswith("[JUmPER ablations]: the reviewer collected no")
            for warning in warnings
        ),
    }


def expected_records_per_generation(meta: dict, usecase_id: str) -> int:
    """Records one generation owes: the review, and each measurement.

    Recomputed from the run's own definition rather than imported from the
    harness, because the monitor must run without the experiment's
    dependencies - and because a monitor that shares code with the thing it
    watches cannot notice when that thing disagrees with its own config.
    """
    protocol = as_mapping(meta.get("protocol"))
    benchmark = as_mapping(protocol.get("benchmark"))
    if not benchmark.get("enabled", True):
        return 1

    manifest = as_mapping(as_mapping(meta.get("usecases")).get(usecase_id))
    usecase_benchmark = as_mapping(manifest.get("benchmark"))
    modes = {
        *as_sequence(usecase_benchmark.get("extra_replay_modes")),
        *as_sequence(benchmark.get("extra_replay_modes")),
    }
    if benchmark.get("mode", "separate") != "inline":
        requested = usecase_benchmark.get("replay_mode")
        if requested:
            modes.add(requested)
    return 1 + len(modes)


def sources_matrix(meta: dict) -> dict:
    """Which of the seven context sources each preset leaves on.

    The single most informative thing about an ablation sweep: it is the
    entire difference between the conditions being compared.
    """
    rows = []
    for ablation_id, definition in as_mapping(meta.get("ablations")).items():
        definition = as_mapping(definition)
        effect = as_mapping(definition.get("effect"))
        context = as_mapping(effect.get("context"))
        rows.append(
            {
                "ablation": ablation_id,
                "family": definition.get("family", "context"),
                "name": definition.get("name", ""),
                "sources": {
                    source: bool(context.get(source, False))
                    for source in SOURCE_IDS
                },
                "overrides": as_mapping(effect.get("overrides")),
            }
        )
    rows.sort(key=lambda row: row["ablation"])
    return {"sources": list(SOURCE_IDS), "rows": rows}


def build_grid(
    meta: dict,
    pass_entries: list[dict],
    record_summaries: list[dict],
    shard_count: int,
    shard_states: dict,
) -> list[dict]:
    """One entry per pass the run is supposed to produce.

    The order is the one the harness enumerates, because that is what decides
    which shard owns which pass: the share is taken by striding, so the pass
    at position *i* belongs to shard ``i % count``.
    """
    suite = as_mapping(meta.get("suite"))
    protocol = as_mapping(meta.get("protocol"))
    usecases = [str(one) for one in as_sequence(suite.get("usecases"))]
    ablations = [str(one) for one in as_sequence(suite.get("ablations"))]
    repetitions = max(1, as_int(protocol.get("repetitions"), 1))
    generations = max(0, as_int(protocol.get("generations_per_target")))

    outcomes = {
        (
            entry.get("usecase"),
            entry.get("ablation"),
            as_int(entry.get("repetition")),
        ): as_mapping(entry)
        for entry in pass_entries
    }
    observed: dict[tuple, list] = {}
    for summary in record_summaries:
        key = (
            summary["usecase"],
            summary["ablation"],
            summary["repetition"],
        )
        observed.setdefault(key, []).append(summary)

    grid = []
    position = 0
    for usecase_id in usecases:
        per_generation = expected_records_per_generation(meta, usecase_id)
        for ablation_id in ablations:
            for repetition in range(repetitions):
                key = (usecase_id, ablation_id, repetition)
                shard = position % shard_count if shard_count else 0
                grid.append(
                    _grid_entry(
                        key=key,
                        shard=shard,
                        outcome=outcomes.get(key),
                        records=observed.get(key, []),
                        expected=generations * per_generation,
                        shard_states=shard_states,
                    )
                )
                position += 1
    return grid


def _grid_entry(
    key: tuple,
    shard: int,
    outcome: dict | None,
    records: list[dict],
    expected: int,
    shard_states: dict,
) -> dict:
    usecase_id, ablation_id, repetition = key
    generations = sorted({record["generation"] for record in records})
    entry = {
        "usecase": usecase_id,
        "ablation": ablation_id,
        "repetition": repetition,
        "shard": shard,
        "records": len(records),
        "records_expected": expected,
        "generations_seen": generations,
        "empty_context": sum(
            1 for record in records if record["empty_context"]
        ),
        "degraded": sum(1 for record in records if record["degraded"]),
        "status": "",
        "error": "",
    }
    entry["state"] = _state_of(outcome, records, shard, shard_states, entry)
    if outcome is not None:
        entry["status"] = outcome.get("status", "")
        entry["error"] = outcome.get("error", "")
        entry["captures"] = outcome.get("captures")
        entry["expected_captures"] = outcome.get("expected_captures")
    return entry


def _state_of(
    outcome: dict | None,
    records: list[dict],
    shard: int,
    shard_states: dict,
    entry: dict,
) -> str:
    """What to colour this cell.

    A pass writes its index line only when it finishes, which for a full
    replay is hours away, so "no index line" cannot mean "nothing is
    happening". The records are the live signal: they land one per
    generation. What the index adds is the verdict.
    """
    if outcome is not None:
        return outcome.get("status") or STATE_OK
    if records:
        return STATE_RUNNING
    # Nothing at all. Whether that is "not started yet" or "the job died
    # before it wrote anything" is not visible in the run directory, which
    # is the whole reason the shard's job id is recorded.
    if shard_states.get(shard) in {"finished", "failed"}:
        return STATE_LOST
    return STATE_PENDING


def build_shards(
    invocations: list[dict],
    grid: list[dict],
    shard_count: int,
    jobs: dict,
) -> list[dict]:
    """One row per shard: who it is, what it is doing, how far it has got."""
    by_shard: dict[int, dict] = {}
    for invocation in invocations:
        label = invocation.get("shard") or ""
        index = _shard_index(label)
        if index is None:
            continue
        row = by_shard.setdefault(
            index,
            {"shard": index, "label": label, "invocations": 0},
        )
        row["invocations"] += 1
        row["node"] = invocation.get("node", "")
        row["recorded_at"] = invocation.get("recorded_at", "")
        job_id = invocation.get("slurm_job_id")
        if job_id:
            row["job_id"] = job_id
            row.update(jobs.get(str(job_id)) or {})

    rows = []
    for index in range(max(shard_count, 1)):
        row = by_shard.get(
            index,
            {"shard": index, "label": f"{index:02d}-of-{shard_count:02d}"},
        )
        mine = [entry for entry in grid if entry["shard"] == index]
        row["passes"] = len(mine)
        row["passes_done"] = sum(
            1 for entry in mine if entry["state"] == STATE_OK
        )
        row["records"] = sum(entry["records"] for entry in mine)
        row["records_expected"] = sum(
            entry["records_expected"] for entry in mine
        )
        running = [entry for entry in mine if entry["state"] == STATE_RUNNING]
        row["current"] = (
            f"{running[0]['usecase']} | {running[0]['ablation']}"
            if running
            else ""
        )
        rows.append(row)
    return rows


def _shard_index(label: str) -> int | None:
    head = label.split("-", 1)[0]
    return int(head) if head.isdigit() else None


def read_results(run: Path) -> dict:
    """What the offline commands have produced so far, if anything."""
    judge = run / "judge"
    files = {
        name: (run / name).is_file()
        for name in (
            "runs.csv",
            "metrics.csv",
            "summary.csv",
            "analysis_metrics.md",
            "suggestions_metrics.md",
        )
    }
    return {
        "files": files,
        "summary": _load_csv(run / "summary.csv"),
        "judge_gaps": _load_csv(judge / "missing.csv"),
        "judge_packets": _count_packets(judge / "tasks"),
        "judge_verdicts": _count_files(judge / "verdicts", "*.json"),
    }


def _count_packets(tasks: Path) -> int:
    if not tasks.is_dir():
        return 0
    return sum(1 for path in tasks.glob("*/*") if path.is_dir())


def _count_files(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def directory_size(path: Path) -> int:
    """Bytes under *path*, symlinks not followed."""
    if not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def list_runs(results_root: Path) -> list[dict]:
    """Every run under *results_root*, newest first."""
    if not results_root.is_dir():
        return []
    runs = []
    for child in sorted(results_root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        meta = read_meta(child)
        runs.append(
            {
                "run_id": meta.get("run_id", child.name),
                "name": child.name,
                "path": str(child),
                "suite": as_mapping(meta.get("suite")).get("id", ""),
                "records": _count_files(child / "records", "*.json"),
            }
        )
    return runs


def snapshot(run: Path, jobs: dict | None = None) -> dict:
    """Everything the page shows, read fresh."""
    run = Path(run)
    meta = read_meta(run)
    jobs = jobs or {}

    problems = describe_problems(run, meta)
    invocations = read_invocations(run)
    shard_count = max(
        1,
        as_int(
            as_mapping(as_mapping(meta.get("config")).get("shard")).get(
                "count"
            ),
            1,
        ),
    )
    stale = _stale_shards(invocations)
    shard_states = {
        index: (
            ""
            if index in stale
            else as_mapping(jobs.get(str(job_id))).get("state", "")
        )
        for index, job_id in _shard_jobs(invocations).items()
    }

    records = [summarise_record(record) for record in read_records(run)]
    pass_entries = read_pass_entries(run)
    grid = build_grid(
        meta,
        pass_entries,
        records,
        shard_count,
        shard_states,
    )

    return {
        "read_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema": {
            "written": as_int(meta.get("schema_version"), 1),
            "understood": SUPPORTED_SCHEMA,
        },
        "problems": problems,
        "run": {
            "run_id": meta.get("run_id", run.name),
            "path": str(run),
            "suite": as_mapping(meta.get("suite")),
            "machine": as_mapping(meta.get("machine")),
            "protocol": as_mapping(meta.get("protocol")),
            "size_bytes": directory_size(run),
        },
        "inputs": {
            "matrix": sources_matrix(meta),
            "usecases": as_mapping(meta.get("usecases")),
            "files": _input_files(run, meta),
        },
        "grid": grid,
        "shards": build_shards(invocations, grid, shard_count, jobs),
        "shard_count": shard_count,
        "records": {
            "total": len(records),
            "by_phase": _count_by(records, "phase"),
            "empty_context": sum(1 for one in records if one["empty_context"]),
            "degraded": sum(1 for one in records if one["degraded"]),
            "llm_latency_s": _total(records, "llm_latency_s"),
            "total_tokens": _total(records, "total_tokens"),
            "rows": records,
        },
        "results": read_results(run),
        "totals": _totals(grid),
    }


def describe_problems(run: Path, meta: dict) -> list[str]:
    """Everything about this run the monitor could not make sense of.

    Said out loud rather than swallowed. A blank panel is indistinguishable
    from a sweep that has not started, and the difference matters most
    exactly when something has gone wrong.
    """
    problems = []
    if not run.is_dir():
        return [f"no directory at {run}"]
    if not (run / "meta.json").is_file():
        problems.append(
            "no meta.json: the run's definition is unknown, so the expected "
            "grid cannot be drawn. Records and results are still read."
        )
    elif not meta:
        problems.append("meta.json is present but could not be parsed")

    written = as_int(meta.get("schema_version"), 1)
    if written > SUPPORTED_SCHEMA:
        problems.append(
            f"written by a newer harness (layout {written}, this monitor "
            f"understands {SUPPORTED_SCHEMA}); unknown fields are ignored"
        )

    records = run / "records"
    if records.is_dir():
        unreadable = sum(
            1
            for path in records.glob("*.json")
            if _load_json(path, None) is None
        )
        if unreadable:
            problems.append(
                f"{unreadable} record(s) could not be parsed - normal for one "
                "being written right now, worth a look for more"
            )
    return problems


def comparable(runs: list[dict]) -> dict:
    """Whether several runs may be put in one table, and why not.

    Aggregation across runs is only meaningful when the runs asked the same
    question: the same presets, defined the same way, under the same
    protocol. Runs that differ are still shown - side by side, labelled -
    but never silently added together, because a pooled number over two
    different definitions is not a number about anything.
    """
    if len(runs) < 2:
        return {"comparable": True, "differs": []}
    first = runs[0]
    differs = []
    for field in ("ablations", "protocol", "usecases"):
        values = {json.dumps(run.get(field), sort_keys=True) for run in runs}
        if len(values) > 1:
            differs.append(field)
    return {
        "comparable": not differs,
        "differs": differs,
        "baseline": first.get("run_id", ""),
    }


def _definition(run: Path, name: str) -> dict:
    """What a run says it asked, in the shape comparable() compares."""
    meta = read_meta(run)
    return {
        "run_id": meta.get("run_id", name),
        "name": name,
        "ablations": meta.get("ablations") or {},
        "protocol": as_mapping(meta.get("protocol")),
        "usecases": sorted(as_mapping(meta.get("usecases"))),
    }


def aggregate(results_root: Path, names: list[str]) -> dict:
    """Several runs' metrics in one table, keyed by what identifies a value.

    The key is (usecase, ablation, metric, reported value) - everything that
    says what a number is about - and the run id stays on every row. Nothing
    is averaged across runs here: what a person needs first is to see the
    same cell from two runs next to each other.
    """
    root = Path(results_root)
    definitions, rows = [], []
    for name in names:
        run = root / name
        definitions.append(_definition(run, name))
        for row in _load_csv(run / "summary.csv"):
            rows.append({**row, "run": name})

    keys = ("usecase", "ablation", "metric", "reported_value")
    grouped: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in keys)
        cell = grouped.setdefault(
            key, {field: row.get(field, "") for field in keys}
        )
        cell[row["run"]] = {
            "estimate": row.get("estimate", ""),
            "paired_delta": row.get("paired_delta", ""),
            "paired_delta_ci_low": row.get("paired_delta_ci_low", ""),
            "paired_delta_ci_high": row.get("paired_delta_ci_high", ""),
            "n": row.get("n", ""),
        }

    return {
        "runs": [one["name"] for one in definitions],
        "compatibility": comparable(definitions),
        "rows": [grouped[key] for key in sorted(grouped)],
    }


def _shard_jobs(invocations: list[dict]) -> dict:
    """The newest job each shard has claimed, by shard index."""
    jobs = {}
    for invocation in invocations:
        index = _shard_index(invocation.get("shard") or "")
        job_id = invocation.get("slurm_job_id")
        if index is not None and job_id:
            jobs[index] = job_id
    return jobs


def _stale_shards(invocations: list[dict]) -> set:
    """Shards whose last word about themselves predates someone else's.

    A resubmission is invisible until its jobs start: a job records itself
    on startup, so a shard queued right now still shows the job of the
    attempt before - which has ended. Reading that as "this shard died"
    turns a queued pass red and says the run is broken when it is waiting.

    What is knowable from the directory is the ordering. If another shard
    has spoken more recently than this one, there is a newer attempt under
    way that this shard has not joined yet, so its silence is not a verdict.
    """
    latest = {}
    for invocation in invocations:
        index = _shard_index(invocation.get("shard") or "")
        stamp = invocation.get("recorded_at") or ""
        if index is not None and stamp > latest.get(index, ""):
            latest[index] = stamp
    if not latest:
        return set()
    newest = max(latest.values())
    return {index for index, stamp in latest.items() if stamp < newest}


def _input_files(run: Path, meta: dict) -> dict:
    return {
        "strategies": str(run / "strategies.yaml"),
        "composed_config": str(run / ".hydra" / "config.yaml"),
        "notebooks": {
            usecase_id: as_mapping(manifest).get("id", usecase_id)
            for usecase_id, manifest in as_mapping(
                meta.get("usecases")
            ).items()
        },
    }


def _count_by(rows: list[dict], field: str) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[field]] = counts.get(row[field], 0) + 1
    return counts


def _total(rows: list[dict], field: str) -> float:
    return sum(row[field] or 0 for row in rows)


def _totals(grid: list[dict]) -> dict:
    states: dict[str, int] = {}
    for entry in grid:
        states[entry["state"]] = states.get(entry["state"], 0) + 1
    return {
        "passes": len(grid),
        "states": states,
        "records": sum(entry["records"] for entry in grid),
        "records_expected": sum(entry["records_expected"] for entry in grid),
    }


# -- averaging over usecases --------------------------------------------

# One number per usecase, then the mean of those numbers: the usecase is the
# unit of analysis, so a payload measured five times does not outvote one
# measured twice. The alternative - throwing every unit into one pool - would
# answer a different question and needs the harness to recompute each metric
# on the pooled units, so it is deliberately not offered here.
COMBINE_GEOMETRIC = "geometric"
COMBINE_ARITHMETIC = "arithmetic"

# Below this many usecases a bootstrap over them says nothing: resampling two
# values with replacement yields three distinct samples, and the percentiles
# of those are a restatement of the two values rather than an interval.
MINIMUM_FOR_INTERVAL = 3

# The same draw count the harness uses for its cell-scope bootstrap, and
# for the same reason: past a few hundred the interval stops moving and the
# cost does not. At 2000 a table of 350 rows took 1.4 s, which is too slow
# for something that recomputes on every click.
_RESAMPLES = 400
_COMBINED_KEY = ("ablation", "metric", "reported_value")
_COMBINED_CARRIED = (
    "metric_name",
    "value_kind",
    "combine",
    "category",
    "evaluation_method",
)


def combine_values(values: list, how: str) -> float | None:
    """The mean of one metric's per-usecase values.

    Geometric for a ratio of runtimes, because the arithmetic mean of ratios
    is biased upwards - twice as fast and twice as slow average to 1.0, not
    to 1.25. A non-positive value has no logarithm, so a geometric mean over
    one is refused rather than silently dropped.
    """
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    if how == COMBINE_GEOMETRIC:
        if any(number <= 0 for number in numbers):
            return None
        return math.exp(sum(math.log(number) for number in numbers) / len(numbers))
    return sum(numbers) / len(numbers)


def _interval(values: list, how: str, seed: int = 0) -> tuple:
    """A percentile interval for the combined value, over the usecases.

    What it measures is how much the usecases disagree, not the total
    uncertainty: each usecase's own interval is not propagated. With fewer
    than MINIMUM_FOR_INTERVAL usecases there is nothing to resample.
    """
    numbers = [value for value in values if value is not None]
    if len(numbers) < MINIMUM_FOR_INTERVAL:
        return (None, None)

    # A geometric mean is an arithmetic mean of logarithms, so the logarithm
    # is taken once per usecase rather than once per resample. That is the
    # whole difference between a table that redraws on a click and one that
    # does not.
    geometric = how == COMBINE_GEOMETRIC
    if geometric:
        if any(number <= 0 for number in numbers):
            return (None, None)
        numbers = [math.log(number) for number in numbers]

    generator = random.Random(seed)
    size = len(numbers)
    drawn = []
    for _ in range(_RESAMPLES):
        total = 0.0
        for _ in range(size):
            total += numbers[generator.randrange(size)]
        drawn.append(total / size)
    if geometric:
        drawn = [math.exp(one) for one in drawn]
    if len(drawn) < 2:
        return (None, None)
    drawn.sort()
    low = drawn[int(0.025 * (len(drawn) - 1))]
    high = drawn[int(0.975 * (len(drawn) - 1))]
    return (low, high)


def read_summary(run: Path) -> list[dict]:
    """A run's summary rows, or none when it has not been scored."""
    return _load_csv(Path(run) / "summary.csv")


_SCORED_CACHE: dict = {}


def _summary_signature(results_root: Path) -> tuple:
    """What would make a cached read of the summaries wrong."""
    stamps = []
    for run in sorted(Path(results_root).glob("*")):
        summary = run / "summary.csv"
        try:
            stamps.append((run.name, summary.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(stamps)


def scored_runs(results_root: Path) -> dict:
    """Every scored run under *results_root*, with its summary rows.

    One pass over the directory: the catalogue and the rows come out of the
    same read, because doing it twice doubles the only part of this that
    touches a shared filesystem.
    """
    root = Path(results_root)
    signature = _summary_signature(root)
    cached = _SCORED_CACHE.get(str(root))
    if cached and cached[0] == signature:
        return cached[1]

    found = {}
    for run in sorted(root.glob("*")):
        if not run.is_dir() or run.name.startswith("."):
            continue
        rows = read_summary(run)
        if rows:
            found[run.name] = rows
    _SCORED_CACHE[str(root)] = (signature, found)
    return found


def usecase_catalogue(
    results_root: Path, scored: dict | None = None
) -> dict:
    """Every scored usecase under *results_root*, and which run holds it.

    A usecase usually has a run to itself - one payload, one sweep - so the
    thing a reader selects is the usecase, and which run supplied it is an
    implementation detail of where it was written.
    """
    scored = scored_runs(results_root) if scored is None else scored
    found: dict[str, list[str]] = {}
    for name, rows in scored.items():
        for row in rows:
            usecase = row.get("usecase")
            if usecase and name not in found.setdefault(usecase, []):
                found[usecase].append(name)
    return found


def rows_for_usecases(
    results_root: Path, usecases: list[str] | None = None
) -> tuple[list[dict], list[str], dict]:
    """Summary rows for the chosen usecases, wherever they were written.

    Also reports whether the presets mean the same thing in every run that
    contributed. Usecases and protocol are *expected* to differ here - that
    is what averaging over usecases is for - but a preset that was defined
    differently in two runs is not one preset, and averaging across that
    would compare two things under one name.
    """
    root = Path(results_root)
    scored = scored_runs(root)
    catalogue = usecase_catalogue(root, scored)
    wanted = set(usecases) if usecases else set(catalogue)
    runs, rows = [], []
    for name, rows_of_run in scored.items():
        mine = [row for row in rows_of_run if row.get("usecase") in wanted]
        if not mine:
            continue
        runs.append(name)
        rows.extend(
            {**row, "run_id": row.get("run_id") or name} for row in mine
        )

    definitions = {
        name: json.dumps(
            read_meta(root / name).get("ablations") or {}, sort_keys=True
        )
        for name in runs
    }
    agree = len(set(definitions.values())) <= 1
    return rows, runs, {
        "presets_agree": agree,
        "runs": runs,
        "detail": (
            ""
            if agree
            else "the preset definitions differ between these runs"
        ),
    }


def per_usecase(rows: list[dict], usecases: list[str] | None = None) -> dict:
    """Summary rows averaged over the usecases, one row per preset and value.

    Everything needed is already in the rows: a usecase's value is a result
    the harness computed and wrote. Combining them is arithmetic over stored
    numbers, which is why this needs no recomputation and no harness.
    """
    wanted = set(usecases) if usecases else None
    buckets: dict[tuple, dict] = {}
    labels: dict[tuple, dict] = {}
    for row in rows:
        if wanted is not None and row.get("usecase") not in wanted:
            continue
        key = tuple(row.get(field, "") for field in _COMBINED_KEY)
        labels.setdefault(
            key, {field: row.get(field, "") for field in _COMBINED_CARRIED}
        )
        estimate = _as_number(row.get("estimate"))
        delta = _as_number(row.get("paired_delta"))
        if estimate is None and delta is None:
            continue
        entry = buckets.setdefault(
            key, {"estimates": {}, "deltas": {}, "clashes": set()}
        )
        usecase = row.get("usecase", "")
        if estimate is not None:
            if usecase in entry["estimates"]:
                entry["clashes"].add(usecase)
            entry["estimates"][usecase] = estimate
        if delta is not None:
            entry["deltas"][usecase] = delta

    combined = []
    for key in sorted(set(labels) | set(buckets)):
        entry = buckets.get(
            key, {"estimates": {}, "deltas": {}, "clashes": set()}
        )
        how = labels[key].get("combine") or COMBINE_ARITHMETIC
        estimates = list(entry["estimates"].values())
        # A difference is a difference whatever the value is: averaging the
        # per-usecase deltas geometrically would be a ratio of ratios, which
        # is not what the rest of the report calls paired_delta.
        deltas = list(entry["deltas"].values())
        low, high = _interval(estimates, how)
        delta_low, delta_high = _interval(deltas, COMBINE_ARITHMETIC, seed=1)
        row = dict(zip(_COMBINED_KEY, key))
        row.update(labels[key])
        row.update(
            {
                "usecase": "",
                "estimate": combine_values(estimates, how),
                "ci_low": low,
                "ci_high": high,
                "paired_delta": combine_values(deltas, COMBINE_ARITHMETIC),
                "paired_delta_ci_low": delta_low,
                "paired_delta_ci_high": delta_high,
                "usecases": sorted(entry["estimates"]),
                "k": len(estimates),
                "per_usecase": entry["estimates"],
                # A usecase measured by two runs: the reader is told rather
                # than handed whichever the directory listed last.
                "clashes": sorted(entry["clashes"]),
            }
        )
        combined.append(row)
    return {
        "mode": "per_usecase",
        "usecases": sorted(
            {row.get("usecase", "") for row in rows if row.get("usecase")}
            if wanted is None
            else wanted
        ),
        "minimum_for_interval": MINIMUM_FOR_INTERVAL,
        "rows": combined,
    }


def _as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
