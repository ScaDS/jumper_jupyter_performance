# Judging protocol

Six of the analysis metrics and two of the suggestion metrics cannot be
computed. Whether an analysis named the right bottleneck, whether a claim is
grounded in the evidence the model was shown, whether three suggestions are
three ideas or one - these are judgements, and this experiment makes them
deliberately, by hand, rather than pretending a second model call is an
oracle.

The harness therefore does not call a judge model. It writes **task packets**
and reads back **verdict files**. What happens in between is an agent session:
a person with a capable model, working through packets. This document is the
contract between the two halves.

It is written for any agent, not a particular tool. Where it says *workspace*
it means whatever directory the session can read and write; where it says
*scratchpad* it means somewhere to keep intermediate notes that is not the
verdict file.

---

## 1. Produce the packets

```bash
python -m jumper_ablations.cli.export_judge target_run=<run id>
```

This writes, under the run directory:

```
judge/
├── index.csv                     # the unblinding key - do not open while judging
├── tasks/<rubric>/<unit id>/     # one packet per unit
└── verdicts/<rubric>/            # where the answers go
```

Adjust the volume before starting, not after:

- `judge.sample_per_cell=2` exports at most two units per (usecase, ablation)
  cell instead of every generation. Units are taken earliest-first, so the
  sample stays paired across presets: generation 1 of every preset, then
  generation 2.
- `judge.blind=false` turns off blinding. Do not, except when debugging the
  harness itself; see §3.

## 2. What a packet contains

```
tasks/evidence_coverage/minian-cell_40__base__r00__g01__review__none/
├── TASK.md                      # what to do, and where the answer goes
├── rubric.md                    # the scale, filled in for this unit
├── verdict.schema.json          # the exact shape of the answer
├── sources/
│   ├── analyze.messages.json    # verbatim messages the reviewer sent
│   ├── suggest.messages.json
│   ├── context_payload.json     # collected state, per context source
│   ├── enabled_sources.json     # which sources this preset had switched on
│   └── reference_facts.yaml     # the usecase's fact sheet
└── output/
    ├── analysis.md              # what the reviewer answered
    ├── analysis_reasoning.md
    └── suggestions.json
```

`sources/analyze.messages.json` and `sources/suggest.messages.json` are the
message lists the reviewer's model actually received, not a summary and not a
re-render: the harness obtains them by calling the same functions the reviewer
graph called, on the same state. That is the entire basis on which a
judgement about "the sources the model had" is a judgement about the sources
the model had. If you find yourself wanting to reconstruct what the model
"probably" saw, stop - it is in the packet.

## 3. Rules for the session

**Judge from the packet alone.** Not from the repository, not from your
knowledge of the library, not from another packet. A claim that is true of
dask and unsupported by `sources/` is unsupported.

**Do not unblind.** A packet does not name the ablation that produced it. The
whole comparison between presets rests on the judge not scoring a label, and a
folder name is enough to bias a score. `judge/index.csv` holds the mapping and
is rejoined automatically at ingest - there is never a reason to open it while
judging.

**Withheld sources are the experiment, not a defect.** `enabled_sources.json`
will often show sources switched off. That is what is being measured. Which
rubric cares, and how, is stated in each `rubric.md`; in general the
bottleneck and factuality rubrics score what the model did with what it had,
and the coverage rubric deliberately does *not* forgive a missing fact,
because the harness computes forgiveness itself from two denominators.

**Abstain rather than guess.** If the packet does not support a judgement, set
`abstained: true` and say why. An abstention is excluded from the metric; a
guess is averaged into it.

**Keep one judgement per unit independent.** Work a packet, write its verdict,
move on. Reading five analyses and then scoring them together produces
comparative scores, and every metric here is absolute.

**Use a scratchpad for working notes.** Counting claims, listing facts,
weighing a borderline score - keep that outside the verdict file. The verdict
file holds conclusions.

## 4. Write the verdict

One JSON file per unit, at the path `TASK.md` names:

```
judge/verdicts/<rubric>/<unit id>.json
```

It must validate against the packet's `verdict.schema.json`:

```json
{
  "unit_id": "minian-cell_40__base__r00__g01__review__none",
  "rubric": "evidence_coverage",
  "judged_by": "claude-opus-5",
  "judged_at": "2026-09-10T15:04:00+02:00",
  "abstained": false,
  "reason": "",
  "verdict": {
    "covered_fact_ids": ["sequential_loop", "repeated_compute"],
    "partially_covered_fact_ids": ["shared_input"],
    "notes": "The analysis names the loop and the repeated compute calls..."
  }
}
```

`judged_by` is not bookkeeping. The judging is manual, so judge drift and an
ablation effect are indistinguishable in the numbers unless every verdict says
what produced it. Fill it in even when it is obvious.

A file that does not validate is reported as a gap with the reason, and never
coerced into numbers.

## 5. Fold the verdicts back in

```bash
python -m jumper_ablations.cli.evaluate target_run=<run id>
python -m jumper_ablations.cli.report   target_run=<run id>
```

`evaluate` writes `metrics.csv` and lists everything still unjudged in
`judge/missing.csv`, with a reason per row: `missing` (nobody judged it),
`abstained` (a session declined), `invalid` (a verdict that would not parse).
Judged and computed metrics become the same kind of row from here on.

Re-run both freely. Neither touches the records, so scoring is cheap and the
expensive phase never repeats.

## 6. Check the judge

Two things are worth doing once the first batch is in, and neither is
optional if the results are going to be quoted:

**Re-judge an overlap set.** Take five to ten units that already have
verdicts, judge them again in a fresh session, and compare. The agreement rate
is the noise floor of every judged metric in the report. A preset difference
smaller than that floor is not a finding.

**Check one packet against the log.** Pick any unit and compare
`sources/analyze.messages.json` with the corresponding
`run <id> | analyze | SystemMessage` / `HumanMessage` block in that pass's
`logs/jumper_logs_*/ai_prompts.log`. They should be identical. That equality is
the fidelity claim of this whole experiment, and it costs a minute to verify.

## 7. Adding a judged metric

Four things, none of them large:

1. a rubric: `jumper_ablations/evaluation/judge/rubrics/<rubric>/template.md`
2. a verdict model: a pydantic class in the metric's module
3. the metric: a `JudgeMetric` subclass with a `spec` and `from_verdict`
4. one item in `configs/metrics/analysis/default.yaml` or
   `configs/metrics/suggestions/default.yaml`

Two metrics may share a rubric by returning the same `rubric_id()` - the two
evidence-coverage metrics do, because they ask the judge one question and
divide the answer by two different denominators. A shared rubric is judged
once.
