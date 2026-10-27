# Paper metrics mapping

## 1. ECCO

| ECCO metric | What it is | Ours |
|---|---|---|
| Speedup | `T(slow input) / T(generated)` on private tests | `mean_median_speedup`, `best_speedup`: `T(original cell) / T(suggestion)`; original cell = slow input |
| Memory Reduction | `M(slow input) / M(generated)` | new `memory_reduction`: `M(original cell) / M(suggestion)`, peak RSS |
| — (ours) | — | new `reference_speedup`: `T(ECCO fast human) / T(suggestion)` |
| — (ours) | — | new `reference_memory_reduction`: `M(ECCO fast human) / M(suggestion)` |
| Runtime % | share of CodeNet submissions with `T(submission) > T(generated)` | new `runtime_percentile` (reference only): same, over submissions re-run on our inputs |
| Memory % | share of CodeNet submissions with `M(submission) > M(generated)` | new `memory_percentile` (reference only): same, over submissions re-run on our inputs |
| pass@1 | `P(first sample passes all tests)` | `pass_at_k.pass_at_1`: `P(first-ranked suggestion verified)`; verified = fingerprint match |

## 2. KernelBench

| KernelBench metric | What it is | Ours |
|---|---|---|
| Speedup vs PyTorch Eager | `T(eager Model) / T(generated)` | `mean_median_speedup`, `best_speedup`: `T(original cell) / T(suggestion)`; original cell = eager |
| Correctness (`fast_0`) | `P(allclose(out_ref, out_gen, atol=rtol=1e-2))` over 5 random inputs; `\|a−b\| ≤ atol + rtol·\|b\|` per element | `correctness_rates.verified_rate`: `isclose(stat_ref, stat_gen, rel_tol=1e-6)` for shape, mean, std, min, max on one input; new `allclose_correctness`: as in paper |
| `fast_p` | `P(correct ∧ speedup > p)` | new `fast_p`, `p ∈ {1, 1.5, 2}`; closest existing: `correctness_conditioned_speedup` |
| Speedup vs `torch.compile` | `T(torch.compile Model) / T(generated)` | new `compile_speedup` |
| Geometric mean speedup | `exp(mean(log speedup))` | `mean_median_speedup.geometric_mean_speedup`, `correctness_conditioned_speedup.geometric_mean_speedup` |


>  PyTorch Eager - default PyTorch non-optimized mode of execution
