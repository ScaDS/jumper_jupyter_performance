# Paper metrics mapping

## 1. ECCO

| ECCO metric | What it is | Ours |
|---|---|---|
| Speedup | `T(slow input) / T(generated)` on private tests | `mean_median_speedup`, `best_speedup`: `T(original cell) / T(suggestion)`; original cell = slow input |
| Memory Reduction | `M(slow input) / M(generated)` | new `memory_reduction`: `M(original cell) / M(suggestion)`, peak RSS |
| — (ours) | — | new `reference_speedup.geometric_mean_reference_speedup` — **Speedup vs human**: `T(ECCO fast human) / T(suggestion)`, geometric mean over correct suggestions |
| — (ours) | — | new `reference_speedup.best_reference_speedup` — **Best speedup vs human**: the same ratio for the fastest correct suggestion of the whole cell — every generation and repetition of this code pair, not averaged. Same extremum caveat as Best Runtime % |
| — (ours) | — | new `reference_memory_reduction.geometric_mean_reference_memory_reduction` — **Memory reduction vs human**: `M(ECCO fast human) / M(suggestion)`, geometric mean over correct suggestions |
| — (ours) | — | new `reference_memory_reduction.best_reference_memory_reduction` — **Best memory reduction vs human**: the same ratio for the leanest correct suggestion of the whole cell, not averaged. Same extremum caveat as Best Runtime % |
| Runtime % | share of CodeNet submissions with `T(submission) > T(generated)` | new `runtime_percentile.mean_runtime_percentile` — **Mean Runtime %** (reference only): same, over submissions re-run on our inputs; each generation is reduced to its best correct suggestion, and those are averaged — where the option a user would pick lands, typically |
| — (ours) | — | new `runtime_percentile.best_runtime_percentile` — **Best Runtime %**: the largest of those per-generation bests, not averaged — the best the preset was ever seen to do. An extremum, so it grows with the number of generations and is not comparable between presets drawn a different number of times |
| Memory % | share of CodeNet submissions with `M(submission) > M(generated)` | new `memory_percentile.mean_memory_percentile` — **Mean Memory %** (reference only): same, over submissions re-run on our inputs; each generation is reduced to its best correct suggestion, and those are averaged |
| — (ours) | — | new `memory_percentile.best_memory_percentile` — **Best Memory %**: the largest of those per-generation bests, not averaged. Same extremum caveat as Best Runtime % |
| pass@1 | `P(first sample passes all tests)` | `pass_at_k.pass_at_1`: `P(first-ranked suggestion correct)`; correct = answers equal to the human fast solution's, as `cli.verify` measured them (the reviewer's fingerprint only where no verification exists) |

## 2. KernelBench

| KernelBench metric | What it is | Ours |
|---|---|---|
| Speedup vs PyTorch Eager | `T(eager Model) / T(generated)` | `mean_median_speedup`, `best_speedup`: `T(original cell) / T(suggestion)`; original cell = eager |
| Correctness (`fast_0`) | `P(allclose(out_ref, out_gen, atol=rtol=1e-2))` over 5 random inputs; `\|a−b\| ≤ atol + rtol·\|b\|` per element | `allclose_correctness` and `correctness_conditioned_speedup.coverage`: as in paper, via `cli.verify`. `correctness_rates.verified_rate` is the **reviewer's own** check - `isclose(stat_ref, stat_gen, rel_tol=1e-6)` for shape, mean, std, min, max on one input - kept as a diagnostic of the product, not as the experiment's correctness |
| `fast_p` | `P(correct ∧ speedup > p)` | new `fast_p`, `p ∈ {1, 1.5, 2}`; closest existing: `correctness_conditioned_speedup` |
| Speedup vs `torch.compile` | `T(torch.compile Model) / T(generated)` | new `compile_speedup` |
| Geometric mean speedup | `exp(mean(log speedup))` | `mean_median_speedup.geometric_mean_speedup`, `correctness_conditioned_speedup.geometric_mean_speedup` |


>  PyTorch Eager - default PyTorch non-optimized mode of execution
