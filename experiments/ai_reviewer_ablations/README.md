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

## Running one

The setup is declarative. A **suite** names the usecases to cross with the
ablations; a **protocol** says how hard each cell of that grid is hit:

```yaml
# configs/suite/context_sources.yaml
id: context_sources
usecases: [minian/cell_40, minian/cell_77]
ablations: [base, no_timing, no_tags, no_perf, no_raw_perf,
            no_hardware, no_packages, code_only, telemetry_only]
```

```bash
# the whole grid
python -m jumper_ablations.cli.run suite=context_sources

# a pilot first - the plan sizes the real N from these interval widths
python -m jumper_ablations.cli.run suite=context_sources protocol=pilot

# the smallest thing that proves the wiring works
python -m jumper_ablations.cli.run suite=smoke protocol.generations_per_target=1
```

Everything lands in `results/<run id>/`, and Hydra's own record of the fully
composed config is written to `results/<run id>/.hydra/config.yaml` - a result
and the setup that produced it are never separated.

Then:

```bash
python -m jumper_ablations.cli.export_judge   # packets for the judged metrics
# ... an agent session judges them; see JUDGE_PROTOCOL.md
python -m jumper_ablations.cli.evaluate       # metrics.csv
python -m jumper_ablations.cli.report         # summary.csv + the two tables
```

The last three read an existing run (`target_run=<run id>`, or the newest by
default) and never touch the reviewer. That split is deliberate: adding a
metric costs a second, not a sweep.

## What lives where

| Path | Responsibility |
|---|---|
| `configs/` | the entire setup, composed by Hydra. Nothing is configured in code. |
| `configs/ablation/` | one file per ablation. `base.yaml` has every source on; the rest are deltas from it. |
| `configs/suite/` | which usecases cross which ablations. |
| `usecases/` | one directory per experiment: a notebook and its manifest. |
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
| a usecase | a directory in `usecases/` with `notebook.ipynb` and `usecase.yaml` |
| an experiment | one file in `configs/suite/` naming usecases and ablations |
| a metric | one module in `metrics/analysis/` or `metrics/suggestions/`, plus one item in the matching `configs/metrics/*/default.yaml` |
| a judged metric | the same, plus a rubric folder and a verdict model (see `JUDGE_PROTOCOL.md` §7) |
| an evaluation method | a subpackage under `evaluation/` and a new `evaluation_method` value |

Metrics are discovered by scanning their package, so no `__init__` needs
editing. Usecases are discovered by scanning `usecases/`. Ablations are
composed by Hydra from the group directory.

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

## Cost

One benchmark is `(1 + suggestions) x runs` replays of the whole notebook
prefix under `full`. Across ten generations and nine ablations and two
usecases that is the dominant cost of the experiment, and it is why
`protocol=pilot` exists. Size the real run from the bootstrap interval widths
the pilot produces, not from a guess.
