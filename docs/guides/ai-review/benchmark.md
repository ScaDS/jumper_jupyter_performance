---
title: Measuring Suggestions
---

# Measuring Suggestions

Without `--benchmark` an option is a proposal. With it, JUmPER runs the
original cell and every suggestion, times them, compares their results, and the
card reports a verdict instead of a promise.

```python
%perfmonitor_ai_review --benchmark                 # review and measure in one go
%perfmonitor_ai_review --resume 7e5a8568 --benchmark   # measure a review you already have
```

!!! tip
    The second form is usually the better order: read the options first, then
    decide whether they are worth the machine time.

!!! warning "Benchmark executes generated code"
    Replays run the notebook prefix and LLM suggestions in separate processes,
    not a sandbox. They can still access the network, absolute paths and
    external services. Review suggestions first if the notebook has side effects.

## What a benchmark does

1. **Replays the prefix** — every cell before the target — so the cell has the
   state it depends on. This is what makes a benchmark slow; see
   [Replay modes](replay-modes.md).
2. **Times the original**, then each option, one at a time. `runs` replays each,
   the first dropped as warm-up, and the median reported.
3. **Compares results.** The values the cell binds at top level are reduced to
   statistical signatures (shape, dtype, mean, std, min, max) and checked
   against the original's. Exact equality is not required — reordering a sum is
   what vectorizing *does*. A value with no signature — a list, a dict, an
   object — carries no signal, and a cell whose result is one of those is
   verified by whatever else it binds.

   By default every binding counts, because nothing here can tell the cell's
   result from its bookkeeping. Name the result with `--result-names` and only
   those bindings decide: a rewrite is then free to move a constant inside the
   function that uses it — which is what deleting the code that read it looks
   like — without its results reading as uncomparable, and one that defines a
   faster function but never calls it is repaired and reported as failed
   instead of being timed as an empty cell.
4. **Repairs what breaks.** An option that raises, or is killed for running far
   past the original's time, goes back to the model with the error, up to
   `fix_attempts` times. Repaired code replaces the option, so applying it
   without another rewrite uses the measured version.

!!! note "Refining invalidates the verdict"
    Adding `--note` to `--resume ... --select ...` rewrites the suggestion after
    benchmarking. The code inserted into the next cell has not been measured.

!!! note
    `--benchmark` measures one cell. A review covering a range is skipped with a
    warning, because each option there rewrites a different cell and each would
    need its own prefix and its own baseline.

## Reading a verdict

| Verdict | Meaning |
|---|---|
| `3.4x faster` / `1.2x slower` | Median duration against the original's, both shown next to it. |
| `results differ from the original (…) - the speedup is unearned` | It got fast by computing something else. Named bindings say which. |
| `results could not be compared` | It ran, but nothing comparable could be captured — treat the speedup as unverified. |
| `it no longer binds X, so its results could not be compared` | Same verdict, with the reason: the rewrite stopped producing something the original produced. If `X` is the cell's result, the speedup is meaningless; if it is a constant that moved into a function, say so with `--result-names`. |
| `repaired after N failed attempt(s)` | The measured code is not the code first proposed. Read the diff. |
| `Failed after N attempt(s)` | Still broken after every repair round; the last error is shown. |
| `Syntax valid` / `not timed` | The timed run was off — the option parses, nothing more. |

## What it costs

Before repairs, a benchmark runs `(1 + options) × runs` measurements. A repaired
option may add up to `runs` more measurements per repair round. Under the default
mode, every measurement also replays the prefix.

Progress is logged as it goes: the original's time first, then each option with
an estimate of what is left.

## Knobs

| Argument | Config key | Default |
|---|---|---|
| `--benchmark-runs N` | `ai.benchmark.runs` | 3 |
| `--fix-attempts N` | `ai.benchmark.fix_attempts` | 3 |
| `--replay-mode MODE` | `ai.benchmark.replay.mode` | `full` — see [Replay modes](replay-modes.md) |
| `--check` / `--skip-check` | `ai.benchmark.checks` | all steps on |
| `--result-names a,b` | `ai.benchmark.result_names` | empty — every binding is compared |
| — | `ai.benchmark.interval` | 0.05 s sampling, finer than the live monitor's |
| — | `ai.benchmark.timeout_factor` | 10 — an option is killed past this multiple of the original's time |

## Running fewer steps

Two steps make up a benchmark, and either can be turned off:

| Step | Does |
|---|---|
| `validate_syntax` | Parses a suggestion before anything is replayed. |
| `run` | The timed replay and the result comparison. |

```python
%perfmonitor_ai_review --benchmark --skip-check run       # parse only, no replays
%perfmonitor_ai_review --benchmark --check validate_syntax  # same, as a whitelist
```

`--skip-check` wins over both `--check` and the config. With `run` off the
benchmark still repairs suggestions that do not parse — it just reports
`Syntax valid` instead of a time, and measures no baseline.

!!! note
    Steps are also gated by what the cell's language can do. A step you turned
    off is skipped silently; one that is on but unsupported is skipped with a
    warning naming the language, so an unsupported language still gets whatever
    it *can* have.

## What a verdict does not prove

- **A replay is not your notebook.** It runs in a temporary working directory
  with no IPython, so a prefix cell that depended on either behaves differently
  there — under every mode.
- **Memory numbers depend on the replay mode.** Under `fork` they are not
  comparable with a full replay's; see the
  [caveat](replay-modes.md#caveat-memory-metrics-under-fork).
- **The comparison is statistical.** Two results with the same shape and the
  same summary statistics are reported as matching.
