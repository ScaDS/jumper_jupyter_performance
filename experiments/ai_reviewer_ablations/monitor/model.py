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


# -- joining two runs ---------------------------------------------------

# Taking a value from one run or the other is not pooling: nothing is
# averaged, and every row says which run it came from. That is what makes it
# allowed between runs whose fingerprints differ, where an average would not
# be - see monitor/PROTOCOL.md section 6.
JOIN_MINE_WINS = "mine_wins"
JOIN_THEIRS_WIN = "theirs_win"
JOIN_SHARED = "shared"
JOIN_MISSING_HERE = "missing_here"
JOIN_VARIANTS = (
    JOIN_MINE_WINS,
    JOIN_THEIRS_WIN,
    JOIN_SHARED,
    JOIN_MISSING_HERE,
)

_JOIN_KEY = ("usecase", "ablation", "metric", "reported_value")
_JOIN_CARRIED = (
    "metric_name",
    "value_kind",
    "category",
    "evaluation_method",
    "estimate",
    "ci_low",
    "ci_high",
    "n",
    "paired_delta",
    "paired_delta_ci_low",
    "paired_delta_ci_high",
    "paired_n",
    "gaps",
)


def has_estimate(row: dict | None) -> bool:
    """Whether a summary row carries a number rather than a blank.

    The distinction the join turns on. A run writes a row for every metric it
    was asked for, so a key being present says nothing: an unjudged metric
    and a measured one differ only in whether the estimate parses.
    """
    if not row:
        return False
    try:
        float(row.get("estimate", ""))
    except (TypeError, ValueError):
        return False
    return True


def join(
    results_root: Path,
    mine: str,
    theirs: str,
    variant: str = JOIN_MINE_WINS,
) -> dict:
    """Two runs' summaries as one table, each row naming where it came from.

    The four fields that say what a number is about are the key, so the same
    cell measured in two runs meets. What happens where both have a value is
    the variant's business, and every variant keeps the provenance: a merged
    table whose rows cannot be traced back is worse than two tables.
    """
    if variant not in JOIN_VARIANTS:
        variant = JOIN_MINE_WINS
    root = Path(results_root)
    left = _summary_by_key(root / mine)
    right = _summary_by_key(root / theirs)

    rows = []
    for key in sorted(set(left) | set(right)):
        row = _joined_row(
            key, left.get(key), right.get(key), mine, theirs, variant
        )
        if row is not None:
            rows.append(row)

    return {
        "mine": mine,
        "theirs": theirs,
        "variant": variant,
        "variants": list(JOIN_VARIANTS),
        "compatibility": comparable(
            [_definition(root / name, name) for name in (mine, theirs)]
        ),
        "counts": {
            "total": len(rows),
            "from_mine": sum(1 for row in rows if row["source"] == mine),
            "from_theirs": sum(1 for row in rows if row["source"] == theirs),
            "empty": sum(1 for row in rows if not row["source"]),
            # Measured by both, not merely present in both files. Counting
            # keys here would report "352 in both" beside a `shared` table
            # of none, which is the confusion has_estimate exists to avoid.
            "in_both": sum(
                1
                for key in set(left) & set(right)
                if has_estimate(left[key]) and has_estimate(right[key])
            ),
        },
        "rows": rows,
    }


def _joined_row(
    key: tuple,
    ours: dict | None,
    theirs_row: dict | None,
    mine: str,
    theirs: str,
    variant: str,
) -> dict | None:
    """One row of the join, or None when this variant drops the key."""
    we_have, they_have = has_estimate(ours), has_estimate(theirs_row)

    if variant == JOIN_SHARED and not (we_have and they_have):
        return None
    if variant == JOIN_MISSING_HERE and (we_have or not they_have):
        return None

    if variant == JOIN_THEIRS_WIN:
        if they_have:
            chosen, source = theirs_row, theirs
        else:
            chosen, source = ours, (mine if we_have else "")
    elif variant == JOIN_MISSING_HERE:
        chosen, source = theirs_row, theirs
    else:
        # mine_wins and shared both prefer this run, which is the point of
        # looking at it: the other run fills what this one has not measured.
        if we_have:
            chosen, source = ours, mine
        else:
            chosen, source = theirs_row, (theirs if they_have else "")

    # Something has to describe the key even when neither run measured it,
    # or the row would be blank where the filter expects a metric name.
    labels = ours or theirs_row or {}
    row = dict(zip(_JOIN_KEY, key))
    row.update({field: labels.get(field, "") for field in _JOIN_CARRIED})
    if chosen:
        row.update({field: chosen.get(field, "") for field in _JOIN_CARRIED})
    row["source"] = source
    row["in_both"] = we_have and they_have
    return row


def _summary_by_key(run: Path) -> dict:
    rows = {}
    for row in _load_csv(run / "summary.csv"):
        rows[tuple(row.get(field, "") for field in _JOIN_KEY)] = row
    return rows


def _definition(run: Path, name: str) -> dict:
    meta = read_meta(run)
    return {
        "run_id": meta.get("run_id", name),
        "name": name,
        "ablations": meta.get("ablations") or {},
        "protocol": as_mapping(meta.get("protocol")),
        "usecases": sorted(as_mapping(meta.get("usecases"))),
    }
