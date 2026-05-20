# Monitor Benchmark

Compares the three monitor backends (`thread`, `subprocess_python`, `native_c`)
at multiple sampling frequencies under full CPU saturation.  Each configuration
is repeated 10 times (configurable); outliers are removed via IQR and results
are reported as mean ± std.

## Quick start

```bash
# 1. Run the benchmark (3 backends × 5 freqs × 10 repeats × 60s ≈ 2.5 h)
python -m jumper_extension.monitor.benchmark.run_benchmark

# 2. Generate comparison plots
python -m jumper_extension.monitor.benchmark.plot_results
```

## Options

```bash
# Shorter runs for quick testing
python -m jumper_extension.monitor.benchmark.run_benchmark --duration 30 --repeats 3

# Only specific backends / frequencies
python -m jumper_extension.monitor.benchmark.run_benchmark \
    --backends native_c,subprocess_python \
    --frequencies 1,4,16
```

## Output

Results are written to `benchmark/results/`:

| File                     | Description                                       |
|--------------------------|---------------------------------------------------|
| `<backend>_<freq>Hz.csv` | Raw per-sample data (median-representative run)   |
| `summary.csv`            | Aggregated mean ± std statistics per config       |
| `A_run_chart.png`        | Binary hit/miss step plot + moving average        |
| `B_cumulative.png`       | Cumulative sample count vs ideal                  |
| `C_iat_histogram.png`    | Inter-arrival time histogram + KDE                |
| `summary_table.png`      | Tabular overview with mean ± std                  |

## Plots explained

- **A. Run-chart**: Shows *when* samples were missed. The shaded step
  function is 1 (hit) or 0 (miss) at each sample. The smoothed line is
  a moving-average hit rate.

- **B. Cumulative success curve**: Compares actual throughput (samples
  received) against the ideal straight line. The gap between the two is
  the deficit.

- **C. Histogram / KDE of inter-arrival times**: Reveals jitter. A tight
  peak around the target interval is ideal; long tails indicate scheduling
  starvation.
