# Rubric: relevance and conciseness

The analysis step was asked for the bottleneck in two to four sentences, and
told explicitly **not** to propose code changes.

Count, in `output/analysis.md`:

- `total_claims` - assertions about the code, the measurements or the
  hardware, as in the factuality rubric.
- `relevant_claims` - those that bear on deciding what to optimize. A true
  observation that would not change the decision is not relevant.
- `irrelevant_observations` - true things that do not bear on the decision:
  narrating the telemetry, restating the code, general advice.
- `premature_suggestions` - code changes proposed here, where none were asked
  for. Naming a technique as the cure counts; naming the cause does not.

Two failure shapes to watch for, because the ablations are expected to produce
both: too little context tends to produce padding, and too much tends to
produce narration of the numbers.
