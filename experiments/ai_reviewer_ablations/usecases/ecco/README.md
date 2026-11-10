# ECCO usecases: how a submission becomes a payload cell

An ECCO row is a pair of **standalone programs**: read stdin, print to stdout,
judged on hidden tests. The reviewer under test reviews a **notebook cell**
instead. These three usecases are that conversion, and nothing else: the slow
submission's own code is kept verbatim, with a wrapper around it and two lines
after it.

## The shape

```python
# prefix cell: inputs, fixed by seed
def generate_input(seed, ...):
    ...                                   # within the problem's constraints
scaled_inputs = [generate_input(seed) for seed in range(N)]

# payload cell - the one under review
def solve(text):                          # the submission, verbatim, inside
    lines = iter(text.splitlines())
    input = lambda: next(lines)           # its own input() now reads `text`
    ...
    return str(answer)                    # what it used to print

answers  = [solve(text) for text in scaled_inputs]
checksum = sum(zlib.crc32(answer.encode()) for answer in answers)
```

## Why each part is there

| Part | Why |
|---|---|
| wrapped in a function | A submission's `input()` reads the kernel's stdin, which a notebook cell and a replay have nothing to put on. Rebinding `input` locally lets the original lines run unchanged against a string. |
| `scaled_inputs` in the **prefix**, not the payload | The payload has to be the thing measured and nothing else. Building the inputs there would time the generator too, and the reviewer would be shown data construction as if it were the bottleneck. |
| inputs generated, not taken from the dataset | ECCO's tests are private; a published row carries neither inputs nor outputs. Generation within the problem's stated constraints, fixed by seed, is the reproducible substitute. |
| inputs scaled up, several of them | One original test runs in milliseconds. The monitor samples at `level`'s interval, so a payload that short yields an empty context and the reviewer is asked to analyse nothing. Scaling to tens of seconds is what makes the cell measurable at all. |
| `return` instead of `print` | The cell's output has to be a value: `cli.verify` reads `answers` out of the namespace for the cell, and captures stdout for the dataset's own programs. Both sides are then normalised token-wise (`str(answer).split()`), so "printed" and "returned" compare equal. |
| `answers = [...]` as the last-but-one line | The only observable the cell leaves. A rewrite that defines a faster solver and never calls it binds nothing, and `cli.verify` reports `incomplete cell: it defines no answers` rather than timing an empty cell. Named in each manifest's `benchmark.result_names`. |
| `checksum = ...` | Scaffolding for the **reviewer's own** check, not part of ECCO: `answers` is a list of strings, which the reviewer's fingerprint cannot summarise, so the checksum is the one comparable value the cell produces. `cli.verify` never reads it. |

## The three

| Usecase | ECCO `edit/test` row · problem | Wrapper | Scaled input |
|---|---|---|---|
| `job_deadlines_day_loop` | 418 · p02996 | `solve(text)`, `input = lambda: next(lines)` | 4 inputs, 2·10⁵ jobs, deadlines ≤ 4·10⁷ (constraint 10⁹); even seeds schedulable, odd seeds not, so both answers occur |
| `sequence_coloring_resort` | 705 · p02973 | `solve(text)`, `input = lambda: next(lines)` | 4 inputs, N = 30 000 values in [0, 10⁹] (constraint 10⁵) |
| `subset_sum_mint_objects` | 738 · p02734 | `main(text)`, `input = iter(text.splitlines(keepends=True)).__next__` | 3 inputs, N = S = 1100 (constraint 3000) |

Each directory beside them holds `human_fast.py` — the row's paired fast
submission, which `cli.verify` runs as a script for the expected answers and
the reference ratios — and `submissions/`, the accepted CodeNet programs that
form the Runtime % / Memory % spectrum (`scripts/fetch_codenet_submissions.py`).

## In flight

`jumper_ablations/verification/program_io.py` runs a stdin/stdout program
in-process, in every way a submission may read or print, and is meant to be
pasted into a generated notebook so that a submission needs no wrapper at all.
`cli.verify` already uses it; these three notebooks still carry the hand-written
wrapper above. Until a builder replaces them, this file describes what is in
them.
