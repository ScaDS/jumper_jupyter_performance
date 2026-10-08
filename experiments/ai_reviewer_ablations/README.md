# Context-source ablations for the AI reviewer

One question: **how much does each source of context the reviewer feeds its
model contribute to the analysis it produces and to the rewrites it
proposes?**

Not a test suite. Nothing here asserts; it runs the real
`%perfmonitor_ai_review`, records everything it produced, and scores it.

The design turns on one fact about the reviewer: a *strategy* is already a flat
`id -> enabled` map over context sources and prompt items, and it already
steers both the collector and the prompt library. So an ablation is not a
special mode - it **is** a strategy. The harness writes its presets in the
reviewer's own schema, points the reviewer at the file through
`JUMPER_AI_STRATEGIES_PATH`, and from the pipeline's side `--strategy
no_timing` is indistinguishable from `--strategy faster`. What gets measured
is the production code path, not a copy of it.

## Installing

Separately from the extension, because it drags in Hydra, a notebook client
and a statistics stack the extension has no business depending on:

```bash
pip install -e experiments/ai_reviewer_ablations
export JUMPER_AI_API_KEY=...    # the reviewer's own key; the harness only passes it through
```

The report's figures are interactive and need nothing extra. Exporting them as
PNG does, and it cannot be done in the same environment as the extension: the
extension pins `kaleido==0.2.1` for its own plotly 5 charts, and plotly 6+
needs `kaleido>=1`. `pip install -e 'experiments/ai_reviewer_ablations[figures]'`
if you want the PNGs and can live with that; otherwise the export step prints
why it skipped and everything else is unaffected.

## Running one

The setup is declarative. A **suite** names the usecases to cross with the
ablations; a **protocol** says how hard each cell of that grid is hit:

```yaml
# configs/suite/context_sources.yaml
id: context_sources
usecases: [minian/cell_40, minian/cell_77]
ablations: [base, no_timing, no_tags, no_perf, no_raw_perf,
            no_hardware, no_packages, code_only, no_code]
```

```bash
# the whole grid
python -m jumper_ablations.cli.run suite=context_sources

# a pilot first - size the real N from these interval widths
python -m jumper_ablations.cli.run suite=context_sources protocol=pilot

# the wiring check: one synthetic payload, no dataset, a few minutes
python -m jumper_ablations.cli.run suite=harness_selftest \
    protocol.generations_per_target=1 protocol.benchmark.runs=2
```

Suites that ship: `context_sources` (the minian pair against every context
ablation), `robotics` (the six robotics payloads), `prompt_rules` (prompt
items, full context), `fact_sheet` (see below), `harness_selftest`, and
`smoke`.

Before the first real sweep, collect the material for the reference facts:

```bash
python -m jumper_ablations.cli.run suite=fact_sheet protocol=fact_sheet
```

That runs each payload once with the benchmark off, which is the cheap part of
a run. Every usecase currently ships code-derived facts only, and until each
has facts sourced from `timing`, `tags` or `perf`, conditional and global
evidence coverage return the same number for every preset that keeps `code`
on - so the pair cannot show what removing telemetry costs. The suite's own
comments say where to read the telemetry from and how to phrase it.

Everything lands in `results/<run id>/`, and Hydra's own record of the fully
composed config is written to `results/<run id>/.hydra/config.yaml` - a result
and the setup that produced it are never separated.

Then:

```bash
python -m jumper_ablations.cli.export_judge   # packets for the judged metrics
# ... an agent session judges them; see JUDGE_PROTOCOL.md
python -m jumper_ablations.cli.evaluate       # metrics.csv + metric_units.csv
python -m jumper_ablations.cli.report         # summary.csv + the two tables
python -m jumper_ablations.cli.index          # runs_index.csv, across all runs
```

The first three read an existing run (`target_run=<run id>`, or the newest by
default); `cli.index` reads all of them. None touches the reviewer. That
split is deliberate: adding a metric costs a second, not a sweep.

`metric_units.csv` is the grain that can be pooled across runs, and
`runs_index.csv` says which runs may be - both are explained in
`monitor/PROTOCOL.md`.

## What lives where

| Path | Responsibility |
|---|---|
| `configs/` | the entire setup, composed by Hydra. Nothing is configured in code. |
| `configs/ablation/` | one file per ablation. `base.yaml` has every source on; the rest are deltas from it. |
| `configs/suite/` | which usecases cross which ablations. |
| `usecases/` | one notebook per experiment, with an optional manifest beside it. |
| `strategies/strategies.yaml` | generated; what the reviewer reads. Never edited. |
| `jumper_ablations/runner/` | drives a kernel cell by cell. No scoring. |
| `jumper_ablations/runtime/` | runs *inside* the kernel: watches the magic, writes the record. |
| `jumper_ablations/metrics/` | one module per metric, split by category. No storage, no statistics. |
| `jumper_ablations/evaluation/` | one subpackage per evaluation method: computed, or judged. |
| `jumper_ablations/aggregate/`, `report/` | estimates, intervals, the two tables. |
| `JUDGE_PROTOCOL.md` | the contract with the agent session that judges. |

## Adding things

The whole point of the layout is that each of these is one small edit:

| To add | Do |
|---|---|
| an ablation | one YAML file in `configs/ablation/` |
| a usecase | a notebook in `usecases/<family>/<name>.ipynb` (see below) |
| an experiment | one file in `configs/suite/` naming usecases and ablations |
| a metric | one module in `metrics/analysis/` or `metrics/suggestions/`, plus one item in the matching `configs/metrics/*/default.yaml` |
| a judged metric | the same, plus a rubric folder and a verdict model (see `JUDGE_PROTOCOL.md` §7) |
| an evaluation method | a subpackage under `evaluation/` and a new `evaluation_method` value |

Metrics are discovered by scanning their package, so no `__init__` needs
editing. Usecases are discovered by scanning `usecases/`. Ablations are
composed by Hydra from the group directory.

## Usecases

A usecase is one notebook under `usecases/`, and its path is its id:
`usecases/minian/cell_40.ipynb` is `minian/cell_40`. One notebook is one
experiment - one payload cell, one review.

**A usecase notebook has to run on its own.** Open it, run it top to bottom,
and it loads the extension, starts the monitor and asks for the review. The
harness runs the same sequence and only appends `--strategy` and `--cells` to
the review line the notebook already carries. That is the point: what the
experiment measured and what the notebook does are the same thing, and either
can be checked against the other by hand.

So a notebook needs two cells of its own - after the title:

```python
%load_ext jumper_extension
%perfmonitor_fast_setup
```

and at the end, after the payload cell:

```python
%perfmonitor_ai_review --benchmark --replay-mode full
```

`python -m jumper_ablations.cli.usecases` lists what is there and what is
missing; `--prepare` writes those two cells into any notebook that lacks them.

Beside the notebook, `usecases/<family>/<name>.yaml` is **optional**. Without
it the title and payload type are read from the notebook's own header and the
benchmark falls back to the protocol. What it adds is the one thing a notebook
cannot state - the reference facts a correct analysis is scored against:

```yaml
payload_type: cpu_bound_python_loop
benchmark: {replay_mode: full, extra_replay_modes: []}
reference_facts:
  - id: sequential_loop
    source: code      # the only context source this fact can be read from
    weight: 2
    fact: >-
      The dominant work is a sequential Python for loop; the iterations are
      independent and nothing runs them concurrently.
```

`source` is what separates conditional coverage from global coverage, and the
weights are what stop "missed the bottleneck" and "missed a detail" counting
the same. A usecase with no facts still measures everything on the suggestions
side; its evidence-coverage metrics abstain rather than score zero.

Telemetry-derived facts - a duration, a tag, a utilisation - must be
transcribed from a measured `base` record (`records/<id>.json` carries
`inputs.context_payload`), never guessed. A fabricated fact silently corrupts
every coverage number computed against it.

## How a run works

Per `(usecase x ablation x repetition)`, a **fresh kernel**:

1. the notebook's own cells run verbatim, up to and including the payload;
2. the harness injects a bootstrap cell, then resolves the payload's cell
   index by matching its source against the cell history;
3. the notebook's own review line runs, with `--strategy <ablation>` and
   `--cells <index>` appended and nothing else changed;
4. a capture cell serialises what the reviewer produced into a record;
5. steps 3-4 repeat `generations_per_target` times;
6. for each replay mode, `--resume <id> --benchmark` re-scores the **stored**
   suggestions. The model is never asked again for another mode.

The prefix runs once per pass, which is what makes ten generations affordable,
and every generation of a pass saw the same machine state.

Two details that are not obvious:

**`--cells` is pinned.** Left off, the magic picks "the last cell that was not
short" - and the harness's own injected cells sit right after the payload.
Whether one of them counts as short is a timing accident, not a design.
`protocol.pin_target_cell=false` restores the plain invocation.

**The benchmark is a separate command by default.**
`protocol.benchmark.mode=separate` asks for the review on its own and then
`--resume <id> --benchmark`, which is how the two commands are meant to be
used together. It is also the only way to see the suggestions as the model
wrote them: an inline `--benchmark` lets the repair loop overwrite them before
anything can record them, and `repairs.idea_change_rate` would have nothing to
compare. `mode=inline` runs the notebook's line exactly as written.

## Reading the results

```
results/<run id>/
├── .hydra/config.yaml     # the setup, as composed
├── meta.json              # suite, protocol, usecases, ablations, machine
├── strategies.yaml        # what `--strategy base` meant on that day
├── records/<id>.json      # one reviewer invocation, complete
├── runs.jsonl, runs.csv   # index and flat view
├── passes.jsonl           # one line per kernel pass, with its status
├── passes/<pass>/logs/    # the extension's log, incl. ai_prompts.log
├── judge/                 # packets, verdicts, the unblinding key, the gaps
├── metrics.csv            # one row per metric value per unit
├── summary.csv            # estimate, interval, delta from baseline
└── analysis_metrics.md, suggestions_metrics.md
```

Nothing is written outside that directory except the workspace the notebooks
execute in. `run_id` defaults to a timestamp; pass `run_id=<name>` to choose
it, and re-running with the same one **resumes** - passes already recorded as
`ok` in `passes.jsonl` are skipped rather than paid for twice.

Four ways to look at a finished run, cheapest first:

| Want | Do |
|---|---|
| Did it run at all | `cat results/<run>/passes.jsonl` |
| One row per invocation | `results/<run>/runs.csv` - speedup, tokens, replay mode, degraded |
| The tables | `results/<run>/analysis_metrics.md`, `suggestions_metrics.md` |
| Figures, and the numbers behind them | `jupyter lab report.ipynb` - opens the newest run, or the one `JUMPER_ABLATION_RESULTS` names |

Everything below `records/` is the raw material: one JSON per reviewer
invocation holding the verbatim messages, the collected context, the
suggestions, the benchmark verdicts and the sampling that was applied. The
tables are regenerable from it at any time with `cli.evaluate` and
`cli.report`, so a new metric never costs another sweep.

`passes.jsonl` is the first file to open. A pass with `prefix_failed` never
got to the reviewer; `capture_failed` means the magic ran and nothing came
back.

Then check the run log for **empty context**. The reviewer collects nothing
when the target cell has no performance data - too short to sample, or the
monitor was not running - and it then logs a warning and calls the model
anyway. The model answers, fluently and specifically, about nothing. The
harness counts populated sources per record and shouts about it, but the fix
is in the usecase: the payload has to run long enough to be sampled.

In the tables, three numbers are read together and never apart:

- a **correctness-conditioned speedup** always carries its **coverage**. 3.0x
  over 10% of suggestions and 1.8x over 90% are not close.
- **conditional** evidence coverage next to **global**. High conditional and
  low global is the signature finding: the model used its context well, and
  the context was not enough.
- any judged metric next to `judge/missing.csv`. A metric averaged over the
  third of units that happened to be judged is not a measurement.

Judging is manual and its isolation is instructional, not enforced: the packet
is self-contained and blind, and `JUDGE_PROTOCOL.md` tells the session to judge
from it alone, but nothing stops a session from reading more. Blinding hides
the preset's name behind a surrogate id; it cannot hide which context sources
the reviewer had, because the messages in the packet are verbatim. The
practical control is the overlap re-judge in `JUDGE_PROTOCOL.md` §6, which
measures how stable the judging actually was.

## Cost

One benchmark is `(1 + suggestions) x runs` replays of the whole notebook
prefix under `full`. Across ten generations and nine ablations and two
usecases that is the dominant cost of the experiment, and it is why
`protocol=pilot` exists. Size the real run from the bootstrap interval widths
the pilot produces, not from a guess.
