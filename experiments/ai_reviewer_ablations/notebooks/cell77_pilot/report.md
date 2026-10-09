# How to read the report: context-source ablations on minian cell 77

**Notebook:** [`pilot_report.ipynb`](pilot_report.ipynb)

**Run:** `cell77_pilot` — 4 presets × 5 repetitions × 1 generation, 40 records,
352 summary rows, 56 judge verdicts

**Plan:** [`agents/reviewer/ablation_experiments_plan.md`](../../../../../agents/reviewer/ablation_experiments_plan.md)

Every number here is read from the notebook's executed cells, which read
`summary.csv` and `metric_units.csv` of that run. The notebook carries headings
and figures only; this document is where each figure is explained.

---

## First — what is actually being compared

One payload, one cell: `minian/cell_77`, a CNMF temporal-update step over a
zarr-backed store. Four presets, differing **only in which context sources the
reviewer was handed**:

| Preset | Withheld |
|---|---|
| `base` | nothing — all seven sources |
| `no_timing` | per-line and per-cell timing |
| `no_perf` | the performance summary and the raw sampled arrays |
| `code_only` | everything but the source code |

Everything else is frozen: the same notebook, the same prefix, the same model,
`temperature` and `seed` pinned, the same replay mode. There is one question:
**what does each source contribute to the review?**

**Terms you will need:**

- **unit** — one `(repetition, generation)` pair. Five repetitions, one
  generation each, so five units per preset. The comparison against `base` is
  **paired** on the unit: the same preset order, the same seed, so whatever the
  machine and the payload contributed is in both numbers and cancels.
- **paired difference** — the ablation's value minus `base`'s, on the same
  units, with a bootstrap interval for that difference. The point estimate alone
  says nothing; the interval is what decides whether a preset differs.
- **result vs input** — a reported value is either the metric's answer
  (`hallucination_rate`) or a quantity it was computed from (`total_claims`).
  Both are reported, because a rate without its denominator is unreadable.
- **conditional vs global coverage** — the same numerator over two
  denominators: the facts this preset could reach, and all the facts there are.
- **n** — how many paired units a comparison rests on. Deterministic metrics
  have 5. Judged metrics have **2**, because only two units per cell were
  sampled for judging. Figure 7 is about exactly this.

---

## 1. What each preset was shown

The whole experiment in one picture: a filled cell is a source the reviewer was
given, an empty one is a source it was not. `code_only` keeps one of seven.

This is worth looking at before any result, because it fixes what the other six
figures are differences *of*. No preset changes the prompt, the model, the
payload or the replay mode — only this grid moves.

---

## 2. Which comparisons moved

**32 of 352 comparisons have a paired interval that stays clear of zero.** Each
dot is one of them, positioned by its difference from `base` as a fraction of
`base`, with the interval of that difference and the absolute change labelled
beside it.

Relative change is what puts token counts and hallucination rates on one axis.
The absolute number is the label, so nothing is hidden by the normalisation.

What to read off it:

- **`code_only` owns most of the list.** It is the only preset that moves
  anything about the *quality* of the review rather than its cost or its
  wordiness.
- **`no_timing` moves only counts** — it returns 3.6 suggestions instead of
  3.0, so every per-suggestion denominator grows with it. That is instruction
  following drifting, not quality changing.
- **`no_perf` moves coverage and latency**, and leaves correctness and speedup
  alone.
- **Several rows are inputs, not results** — `total_claims`, `facts`,
  `total_weight`, `suggestions`. They are in the figure because a denominator
  moving *is* the finding in several of these cases: see §4.

The honest reading of 320 unmoved comparisons is not "no effect": with five
units per comparison the intervals are wide, and this pilot was sized to
measure interval widths, not to settle every metric. What it does settle is the
handful below.

---

## 3. The resource, and how the analysis reads

This is the report's centrepiece, and it is a paradox read left to right.

| | `base` | `code_only` | difference | interval |
|---|---|---|---|---|
| Resource named (0–2) | 2.00 | **0.00** | −2.00 | −2.00 … −2.00 |
| Claims the sources support | 0.775 | **0.857** | +0.082 | +0.057 … +0.107 |
| Claims about nothing shown | 0.183 | **0.071** | −0.112 | −0.167 … −0.057 |

Strip the telemetry and the reviewer:

- **never once names the right limiting resource.** The interval is degenerate —
  −2.00 to −2.00 — because the score was 0 on both judged units. Not "worse on
  average": wrong every time.
- **writes a better-behaved analysis.** A higher share of its claims are
  supported by what it was shown, and it invents less.

Both are true, and the second is why the first is dangerous. `code_only` makes
fewer claims (7 against 11) and keeps them inside the code it was handed, so it
scores well on groundedness *and* says nothing about where the time went. A
reviewer that hallucinates is detectable. A reviewer that is scrupulous about a
question it cannot see is not.

A bar drawn with a dark outline is one whose paired interval clears zero. The
outline is the only mark used for that, in all figures: direction is the
metric's business, and half of these are rates where down is good.

**Carried by two paired units.** See §7 before leaning on it.

---

## 4. Conditional against global coverage

Left: weighted recall of the reference facts, over two denominators. Right: how
many of the six facts each preset could reach at all.

| Preset | Reachable facts | Conditional recall | Global recall |
|---|---|---|---|
| `base` | 6 | 0.50 | 0.50 |
| `no_timing` | 5 | 0.56 | 0.56 |
| `no_perf` | 4 | 0.71 | **0.81** |
| `code_only` | 3 | **0.80** | 0.44 |

`code_only` is the signature case, and the two columns say opposite things about
it. Conditional recall rises to 0.80 against `base`'s 0.50 — interval
+0.244 … +0.356, clear of zero. Global recall *falls*, to 0.44. It could reach
three of the six facts and covered most of them; `base` could reach all six and
covered half. The ablation did not get better at using evidence: its denominator
halved, and against the facts that actually describe the cell it did worse.

Read conditional alone and `code_only` is the best preset here. That is the
misreading both metrics exist to prevent.

This is the pairing the plan asks for and the reason both metrics exist:

- high conditional, low global → **the model used its context well, and the
  context was not enough**. That is the signature of a source that matters.
- low conditional, low global → the context was there and the model ignored it.

`no_perf` is the oddity: both recalls go *up* against `base` (+0.208 and
+0.306), and both intervals clear zero. Removing the performance summary cannot
have added facts to the payload, so what moved is which facts the judge
credited on two units. This is the one result in the figure the pilot cannot
resolve, and it is listed as a finding rather than quietly dropped because the
interval says it is not noise in the resampling - it is noise in the sample.

---

## 5. What the suggestions did to the runtime

Each dot is one unit's geometric mean speedup over the suggestions it produced;
the box is their spread. The dotted line at 1.0 is "no change" — above it the
suggestions made the cell faster, below it they made it slower.

| Preset | Geometric mean | Paired difference | Interval |
|---|---|---|---|
| `base` | 0.949 | — | — |
| `code_only` | **0.613** | −0.335 | −0.574 … −0.082 |

One fact conditions the whole figure: **`verified_rate` is 0 for every
preset.** Not one suggestion, in any condition, produced an output fingerprint
matching the baseline. So `correctness_conditioned_speedup` - the headline
metric of the plan - is empty for this run, and what is plotted here is speedup
over *all* measured suggestions, correct or not. On this payload the comparison
between presets is still paired and still valid; the absolute speedups are not
a claim that any suggestion was safe to apply.

Two things to take from it:

- **No preset reliably speeds the cell up.** `base` sits just under 1.0. On this
  payload, under this replay mode, the reviewer's suggestions are roughly
  neutral at best — which is a finding about the reviewer, not about the
  ablations.
- **`code_only` is reliably worse.** Its suggestions slow the cell down, and the
  interval of the difference stays clear of zero. `best_speedup.mean_best_speedup`
  agrees (0.784 against 1.000, interval −0.412 … −0.064): even the *best*
  suggestion of a telemetry-blind review is a regression.

This is where §3 lands. The analysis that named the wrong resource produced
optimisations aimed at the wrong thing, and the benchmark measured the cost.

---

## 6. What it cost

Three panels, one axis each, same preset order: input tokens, review latency,
and the speedup those bought.

| Preset | Input tokens | Latency (s) | Geometric mean speedup |
|---|---|---|---|
| `base` | 14,035 | 63.8 | 0.949 |
| `code_only` | **1,838** | **33.7** | **0.613** |
| `no_perf` | — | **36.6** | — |

`code_only` is **7.6× cheaper in input tokens** and **1.9× faster**, and every
one of those savings is real and reliably measured. It is also the preset that
cannot find the bottleneck and whose suggestions regress.

`no_perf` is the interesting middle: it halves the latency (−27.2 s, interval
−37.3 … −16.5) with no measured cost in correctness, speedup or resource
identification. If this held at larger N it would be the one defensible
economy in the set — the raw sampled arrays are most of the prompt and
contributed nothing this pilot could detect.

Cost is never read without the quality beside it, which is why the third panel
is in the same figure rather than its own.

---

## 7. What the judged half rests on

Left: how many paired units each kind of comparison has. Right: judge verdicts
written against gaps left.

- **Deterministic metrics: 5 paired units.** Computed from the records, so every
  unit counts.
- **Judged metrics: 2 paired units.** `sample_per_cell=2` by design — judging is
  a manual agent pass, and the pilot sampled two units per cell.
- **56 verdicts written, 96 gaps.** The gaps are the 12 unjudged units × 8
  judged metrics. Every one is listed in `judge/missing.csv` with a reason;
  none is counted as a zero.

This is the report's main limitation and it is asymmetric. §5 and §6 rest on
five units and are as solid as a five-unit pilot allows. §3 and §4 rest on two,
and the degenerate intervals there (−2.00 … −2.00) are degenerate because two
identical observations have no spread — not because the effect is certain.

The direction of §3 is not in doubt: a resource score of 0 on both judged units,
with a reviewer that never saw a timing number, is what the design predicts.
The *size* of every judged effect should be re-read after the remaining 12 units
are judged.

---

## Bottom line

1. **Telemetry is what makes the review about performance.** Remove it and the
   reviewer still writes a fluent, well-grounded, concise analysis — of the
   wrong resource, with suggestions that make the cell slower. This is the
   failure mode the experiment was built to catch, and it is the one that would
   be invisible to any metric that only checked whether claims were supported.

2. **Groundedness and usefulness came apart.** `code_only` scores *better* on
   supported-claim precision and hallucination rate than `base`. Any evaluation
   that reports those without a bottleneck score would rank the blindest preset
   first.

3. **Conditional coverage must never be read without global.** On its own it
   says `code_only` uses evidence best; with its denominator it says the
   evidence was not there.

4. **The raw performance arrays may be free to drop.** `no_perf` halves latency
   with nothing measurable lost. This is the pilot's one actionable economy and
   the first thing the main run should confirm or kill.

5. **The pilot sized the main run, as intended.** Interval widths here say what
   N buys: the deterministic half is usable at 5 units, and the judged half is
   not — the next run needs either more judged units per cell or fewer judged
   metrics.
