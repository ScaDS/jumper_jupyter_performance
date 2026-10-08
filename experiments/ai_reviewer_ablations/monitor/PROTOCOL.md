# The contract between a run and anything that reads one

The harness writes run directories. The monitor reads them. Nothing else
passes between the two: the harness does not import the monitor, the monitor
does not import the harness, and neither is on the other's dependency list.

This document is that interface. It exists because the alternative - a reader
that shares code with the thing it watches - cannot notice when a run
disagrees with its own configuration, and cannot be run at all without
installing the experiment first.

`tests/test_monitor.py` is the executable half of this document. Every rule
below that can be broken by a program is asserted there.

---

## 1. The only shared thing is a directory

```
<results_root>/
├── runs_index.csv                every run, and which may be pooled
└── <run_id>/                     one run
    ├── meta.json                 what this run was asked to do
    ├── strategies.yaml           the presets it measured
    ├── passes-<shard>.jsonl      finished passes, one file per writer
    ├── passes.jsonl              the same, from before shards existed
    ├── invocations/*.json        one per job that touched the run
    ├── records/*.json            one per reviewer invocation
    ├── runs.csv, runs.jsonl      flat views of the records
    ├── metrics.csv               one row per metric value
    ├── metric_units.csv          the same, at unit grain
    ├── summary.csv               one row per (usecase, preset, metric)
    ├── analysis_metrics.md       rendered tables
    ├── suggestions_metrics.md
    ├── judge/                    packets, verdicts, gaps
    └── .hydra/config.yaml        the fully composed configuration
```

**Everything in that list is optional except the directory.** A reader is
given a run that holds nothing but `records/` and must describe it. A run
whose first job died before writing `meta.json` is a run, and is shown.

## 2. Five rules, and what each one buys

### R1. A reader ignores what it does not know

Unknown files, unknown keys, unknown values in known keys. The harness may
add a field at any time without touching any reader, and does not announce
it.

*Cost of breaking it:* every field added to the harness becomes a change to
every reader, and the two are no longer independently releasable.

### R2. A reader assumes nothing about a value's type

Every number is read through a coercion that has a default; every mapping and
every sequence through one that returns an empty one. A field the reader
expected to be an integer and that holds `"many"` yields its default, not a
traceback.

*Cost of breaking it:* the monitor is looked at precisely when a run has gone
strange. That is the worst possible moment for it to refuse to draw.

### R3. Absence is a value, never an error

A missing file is an empty section. A missing definition means the expected
grid cannot be computed - so it is not computed, and the records are still
counted and shown.

### R4. A half-written file is normal

Runs are watched while several jobs write them. A truncated JSON file, a
partial last line in an index: skipped, counted, and named in `problems`.
Never fatal, never silent.

### R5. The reader does not write

No file in a run directory is created, modified or removed by a reader. A
reader may be started, stopped and pointed at a live sweep with no effect on
it. The only external command it runs is a queue query.

## 3. The fields a reader may rely on

Everything else in these files exists and may be read, but is not promised.
Removing or repurposing anything below is a layout change - see §5.

### `meta.json`

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | Layout version. Absent means 1 |
| `run_id` | str | The run's name |
| `suite.id` | str | Which suite was run |
| `suite.usecases` | list[str] | Usecase ids, in execution order |
| `suite.ablations` | list[str] | Preset ids, in execution order |
| `protocol.generations_per_target` | int | Reviews per pass |
| `protocol.repetitions` | int | Passes per (usecase, preset) |
| `protocol.benchmark.enabled` | bool | Whether suggestions are measured |
| `protocol.benchmark.mode` | str | `separate` or `inline` |
| `protocol.benchmark.extra_replay_modes` | list[str] | Modes re-scored |
| `usecases.<id>` | object | The usecase manifest, as it was used |
| `usecases.<id>.benchmark.replay_mode` | str | The mode it asked for |
| `usecases.<id>.reference_facts` | list | Facts, each with `source`/`weight` |
| `ablations.<id>.effect.context` | object | `source id -> bool` |
| `ablations.<id>.family` | str | `context` or `prompt` |
| `config.shard.count` | int | How many jobs shared the grid |
| `machine` | object | Where the first invocation ran |

The manifests and preset definitions are **embedded**, not referenced. That
is what lets a run be scored and displayed after its notebooks have been
renamed, and what lets a reader work without a YAML parser.

### `passes-<shard>.jsonl`, `passes.jsonl`

One JSON object per line, appended by exactly one writer per file.

| Field | Type | Meaning |
|---|---|---|
| `usecase`, `ablation`, `repetition` | str, str, int | Which pass |
| `status` | str | `ok`, or why not |
| `captures`, `expected_captures` | int | Records written versus owed |
| `empty_context` | int | Generations the reviewer saw nothing in |
| `error` | str | Free text |

A line appears when a pass **ends**. Under a full replay that is hours after
it started, so this file cannot be used to tell whether anything is
happening. See §4.

### `records/<record id>.json`

The record id is also the file name and encodes the grid position:

```
<usecase with / as ->__<preset>__r<NN>__g<NN>__<phase>__<replay mode>
```

| Field | Meaning |
|---|---|
| `identity.*` | usecase, ablation, repetition, generation, phase, reviewer run id |
| `outputs.suggestions`, `outputs.benchmarks` | what the reviewer produced |
| `cost.llm_latency_s`, `cost.total_tokens` | what it cost |
| `environment.actual_replay_mode`, `.degraded` | what actually ran |
| `environment.warnings` | list of strings; empty context is one of them |

### `invocations/<shard>-<stamp>.json`

| Field | Meaning |
|---|---|
| `shard` | `NN-of-NN`, naming the slice this job ran |
| `slurm_job_id` | The queue's id, when there was a queue |
| `node`, `recorded_at` | Where and when |

### `summary.csv`

One row per `(run_id, usecase, ablation, metric, reported_value)`, with
`estimate`, `ci_low`, `ci_high`, `paired_delta`, `paired_delta_ci_low`,
`paired_delta_ci_high`, `paired_n`, `gaps`. Two runs' rows join on those
five columns - but see §6 before adding any of them together.

### `metric_units.csv`

One row per `(run_id, usecase, ablation, metric, reported_value, repetition,
generation)`, with `value`, `scope`, `category`, `evaluation_method`, `note`.

This file exists because `metrics.csv` holds two grains at once, and a reader
that treats them alike will be wrong without looking wrong. A row whose
`scope` is `run` is one reviewer invocation. A row whose `scope` is `cell`
has already reduced every repetition to one number, and says how many in
`sample_size`. In the pilot, 63 of 164 cell values differ from the mean of
their own units - so concatenating cell rows from two runs weights them by
nothing in particular, and the paired interval cannot be rebuilt from them at
all, because the paired bootstrap resamples units.

Pooling therefore reads this file, concatenates the units, and recomputes.

A per-unit value of a cell-scope metric is an **input to that recomputation,
never a number to report**: `pass_at_k` over a single generation is
degenerate by construction, and so is any rate whose denominator is one
suggestion.

### `runs_index.csv`

At the root rather than in a run. One row per run, with `run_id`,
`fingerprint`, `created_at`, `schema_version`, `suite`, `usecases`,
`ablations`, `generations_per_target`, `repetitions`, `benchmark_mode`,
`shard_count`, `records`, `has_metrics`, `has_units`, `has_summary`, `path`.

`fingerprint` is a digest of exactly the three `meta.json` fields of §6, so
equal fingerprint means the runs asked the same question. It turns "which
runs may be pooled" into a group-by instead of a walk that opens every run.

Derived and disposable: rebuilt by `python -m jumper_ablations.cli.index`,
never written by a job, so no shard contends for it. A reader treats it as a
shortcut, not as truth - absent or stale, the per-run `meta.json` still
decides, and R3 applies.

## 4. Progress is counted from records, not from passes

The only rule here that is about meaning rather than mechanics, and the one
most likely to be got wrong by a new reader.

A pass writes its index line when it finishes. A full-replay pass over a
real payload takes hours. A reader that derives progress from the index will
show a working sweep as idle for most of its life, and cannot distinguish
that from a stalled one.

Records land one per generation, so they are the live signal. The index adds
the **verdict**: whether the pass ended well.

| Cell state | Derived from |
|---|---|
| `ok`, `incomplete`, `<failure>` | the pass index line |
| `running` | records exist, no index line |
| `pending` | neither, and the shard's job is queued, running or unknown |
| `lost` | neither, and the shard's job has ended |

`lost` is the whole reason `slurm_job_id` is recorded. On disk, "not started
yet" and "died before writing anything" are the same thing.

One qualification, and it is the difference between a useful red cell and a
misleading one. A job records itself when it **starts**, so a shard that has
been resubmitted and is sitting in a queue still shows the job of the attempt
before - which has ended. Calling that `lost` reports a run as broken while
it is waiting. A reader therefore compares invocation timestamps: when
another shard has recorded itself more recently than this one, a newer
attempt is under way that this shard has not joined yet, and its silence is
not a verdict. Only a shard whose last word is as recent as anyone's can be
`lost`.

## 5. Versioning

`meta.json.schema_version` names the layout. Readers declare the highest they
understand.

| Change | Version |
|---|---|
| Adding a field, a file, a directory | unchanged - R1 covers it |
| Adding a value to an existing enum | unchanged - R2 covers it |
| Moving or renaming a file a reader relies on | bumped |
| Changing what an existing field means | bumped |
| Removing a field from §3 | bumped |

A reader given a higher version **still renders**, and says so. Refusing to
draw a running sweep because its layout is one ahead trades a small
inaccuracy for a total one.

## 6. Reading several runs

Runs are independent directories. A reader may list and display any number.

Putting their numbers in **one table** is a different matter, and is only
meaningful when the runs asked the same question. Comparability is decided by
three fields of `meta.json`:

- `ablations` - the preset definitions, not just their names
- `protocol` - generations, repetitions, benchmark shape, sampling
- `usecases` - which payloads

Runs that differ in any of them are shown side by side and labelled, never
pooled. A number averaged over two different definitions is not a number
about anything, and the label is what stops someone reading it as one.

`runs_index.csv` hashes those same three fields into `fingerprint`, so the
test is a comparison of two strings rather than of two nested dictionaries.
Equal fingerprint is the *permission* to pool, not the method: pooling joins
`metric_units.csv` and recomputes the estimate and the interval on the
combined units. Averaging two runs' `summary.csv` rows is not pooling, and
there is no weighting that makes it one.

## 7. What a reader must never do

| | Why |
|---|---|
| Write into a run directory | A sweep is days of machine time; a reader is not worth risking it for |
| Import the harness | Then it needs the harness's dependencies, and cannot run where it is wanted |
| Cache a snapshot between requests | Several jobs are writing; anything remembered is a guess |
| Require any file to exist | §1 |
| Raise on unexpected content | §2, R2 |
| Bind to anything but loopback unless told | A run holds every prompt and answer, and a login node is shared |
