#!/usr/bin/env python3
"""Visualise monitor benchmark results.

Reads the CSV files written by ``run_benchmark.py`` and produces three
comparison plots:

    A. Run-chart (binary step + moving average)
    B. Cumulative success curve
    C. Histogram / KDE of inter-arrival times

Usage
-----
    python -m jumper_extension.monitor.benchmark.plot_results [--results-dir ...]

Plots are saved as PNG files alongside the result CSVs.
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Consistent colours for backends
BACKEND_COLORS = {
    "thread": "#e74c3c",
    "subprocess_python": "#3498db",
    "native_c": "#2ecc71",
}
BACKEND_LABELS = {
    "thread": "Thread",
    "subprocess_python": "Subprocess (Python)",
    "native_c": "Native C",
}


def load_run_data(results_dir):
    """Load all per-run CSV files into a dict keyed by (backend, freq_hz)."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_*Hz.csv"))):
        fname = os.path.splitext(os.path.basename(path))[0]
        # e.g. "native_c_10Hz" → backend="native_c", freq=10
        parts = fname.rsplit("_", 1)
        freq_str = parts[-1]  # "10Hz"
        freq = float(freq_str.replace("Hz", ""))
        backend = parts[0]
        df = pd.read_csv(path)
        runs[(backend, freq)] = df
    return runs


def load_summary(results_dir):
    path = os.path.join(results_dir, "summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ------------------------------------------------------------------
# Plot A: Run-chart (binary step + moving average)
# ------------------------------------------------------------------

def plot_run_chart(runs, results_dir):
    """One subplot per frequency.  Within each: one line per backend."""
    freqs = sorted({f for _, f in runs.keys()})
    backends = sorted({b for b, _ in runs.keys()})

    fig, axes = plt.subplots(
        len(freqs), 1,
        figsize=(14, 3.5 * len(freqs)),
        sharex=False,
        squeeze=False,
    )

    for row, freq in enumerate(freqs):
        ax = axes[row, 0]
        interval = 1.0 / freq
        for backend in backends:
            key = (backend, freq)
            if key not in runs:
                continue
            df = runs[key]
            if df.empty or "hit" not in df.columns:
                continue

            t = df["time_rel"].values
            hit = df["hit"].astype(float).values

            color = BACKEND_COLORS.get(backend, "gray")
            label = BACKEND_LABELS.get(backend, backend)

            # Moving average (window = 2× frequency, min 5 samples)
            win = max(5, freq * 2)
            if len(hit) >= win:
                ma = pd.Series(hit).rolling(win, min_periods=1).mean()
                ax.plot(t, ma, color=color, linewidth=1.8, label=label)
            else:
                ax.plot(t, hit, color=color, linewidth=1, label=label)

            # Mark individual misses as small markers on the x-axis
            miss_mask = hit < 0.5
            if miss_mask.any():
                ax.scatter(
                    t[miss_mask],
                    np.full(miss_mask.sum(), -0.03),
                    color=color, marker="|", s=30, alpha=0.6,
                )

        ax.set_ylim(-0.08, 1.08)
        ax.set_ylabel("Hit rate (moving avg)")
        ax.set_title(f"{freq} Hz (interval = {interval:.3f}s)")
        ax.legend(loc="lower left", fontsize=8)
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle("A. Run-chart: moving-average hit rate  (ticks = individual misses)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(results_dir, "A_run_chart.png"), dpi=150)
    plt.close(fig)
    print("  → A_run_chart.png")


# ------------------------------------------------------------------
# Plot B: Cumulative success curve
# ------------------------------------------------------------------

def plot_cumulative(runs, results_dir):
    freqs = sorted({f for _, f in runs.keys()})
    backends = sorted({b for b, _ in runs.keys()})

    fig, axes = plt.subplots(
        len(freqs), 1,
        figsize=(14, 3.2 * len(freqs)),
        sharex=False,
        squeeze=False,
    )

    for row, freq in enumerate(freqs):
        ax = axes[row, 0]
        interval = 1.0 / freq

        for backend in backends:
            key = (backend, freq)
            if key not in runs:
                continue
            df = runs[key]
            if df.empty:
                continue

            t = df["time_rel"].values
            actual = np.arange(1, len(t) + 1)
            color = BACKEND_COLORS.get(backend, "gray")
            label = BACKEND_LABELS.get(backend, backend)
            ax.plot(t, actual, color=color, linewidth=1.5, label=label)

        # Ideal line: at time t the ideal count is floor(t * freq) + 1
        # (the +1 accounts for the immediate first sample at t ≈ 0)
        max_t = max(
            (runs[k]["time_rel"].iloc[-1] for k in runs if k[1] == freq),
            default=60,
        )
        t_ideal = np.linspace(0, max_t, 200)
        ideal_count = t_ideal * freq + 1
        ax.plot(t_ideal, ideal_count, color="black",
                linewidth=1, linestyle="--", alpha=0.5, label="Ideal")

        ax.set_ylabel("Cumulative samples")
        ax.set_title(f"{freq} Hz")
        ax.legend(loc="upper left", fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle("B. Cumulative success curve: actual vs ideal throughput",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(results_dir, "B_cumulative.png"), dpi=150)
    plt.close(fig)
    print("  → B_cumulative.png")


# ------------------------------------------------------------------
# Plot C: Histogram / KDE of inter-arrival times
# ------------------------------------------------------------------

def plot_iat_histogram(runs, results_dir):
    freqs = sorted({f for _, f in runs.keys()})
    backends = sorted({b for b, _ in runs.keys()})

    fig, axes = plt.subplots(
        len(freqs), 1,
        figsize=(14, 3.2 * len(freqs)),
        sharex=False,
        squeeze=False,
    )

    for row, freq in enumerate(freqs):
        ax = axes[row, 0]
        interval = 1.0 / freq

        for backend in backends:
            key = (backend, freq)
            if key not in runs:
                continue
            df = runs[key]
            if df.empty or "inter_arrival" not in df.columns:
                continue

            iat = df["inter_arrival"].dropna().values
            if len(iat) < 2:
                continue

            color = BACKEND_COLORS.get(backend, "gray")
            label = BACKEND_LABELS.get(backend, backend)

            # Histogram
            # Clip outlier display at 5× interval for readability
            clip_max = interval * 5
            iat_clipped = np.clip(iat, 0, clip_max)
            bins = np.linspace(0, clip_max, 80)
            ax.hist(
                iat_clipped, bins=bins,
                alpha=0.35, color=color, label=label,
                density=True, edgecolor="none",
            )

            # KDE overlay (using gaussian_kde from scipy if available,
            # otherwise skip)
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(iat_clipped, bw_method=0.1)
                x = np.linspace(0, clip_max, 300)
                ax.plot(x, kde(x), color=color, linewidth=1.5)
            except ImportError:
                pass

        # Ideal interval marker
        ax.axvline(interval, color="black", linewidth=1, linestyle="--",
                    alpha=0.6, label=f"Target ({interval:.3f}s)")

        ax.set_ylabel("Density")
        ax.set_title(f"{freq} Hz — inter-arrival time distribution")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    axes[-1, 0].set_xlabel("Inter-arrival time (s)")
    fig.suptitle("C. Histogram / KDE of inter-arrival times",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(results_dir, "C_iat_histogram.png"), dpi=150)
    plt.close(fig)
    print("  → C_iat_histogram.png")


# ------------------------------------------------------------------
# Summary table plot
# ------------------------------------------------------------------

def plot_summary_table(results_dir):
    summary = load_summary(results_dir)
    if summary is None or summary.empty:
        return

    # Build a display table with "mean ± std" formatted strings
    rows = []
    for _, r in summary.iterrows():
        def _fmt_pct(m, s):
            return f"{m:.1f} ± {s:.1f}%"
        def _fmt_f(m, s, decimals=4):
            return f"{m:.{decimals}f} ± {s:.{decimals}f}"
        def _fmt_int(m, s):
            return f"{m:.0f} ± {s:.0f}"

        rows.append([
            r.get("backend", ""),
            int(r.get("freq_hz", 0)),
            f"{r.get('interval', 0):.3f}",
            _fmt_f(r.get("duration_mean", 0), r.get("duration_std", 0), 1),
            int(r.get("expected", 0)),
            _fmt_int(r.get("actual_mean", 0), r.get("actual_std", 0)),
            _fmt_pct(r.get("hit_rate_mean", 0), r.get("hit_rate_std", 0)),
            _fmt_f(r.get("mean_iat_mean", 0), r.get("mean_iat_std", 0)),
            _fmt_f(r.get("median_iat_mean", 0), r.get("median_iat_std", 0)),
            _fmt_f(r.get("p95_iat_mean", 0), r.get("p95_iat_std", 0)),
            _fmt_f(r.get("max_iat_mean", 0), r.get("max_iat_std", 0)),
            int(r.get("n_runs", 0)),
        ])

    col_labels = [
        "Backend", "Freq\n(Hz)", "Interval\n(s)", "Duration\n(mean±std)",
        "Expected", "Actual\n(mean±std)", "Hit %\n(mean±std)",
        "Mean IAT\n(mean±std)", "Median IAT\n(mean±std)",
        "P95 IAT\n(mean±std)", "Max IAT\n(mean±std)", "Runs",
    ]

    fig, ax = plt.subplots(figsize=(18, 0.5 + 0.45 * len(rows)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(list(range(len(col_labels))))

    fig.suptitle("Benchmark Summary (mean ± std, outliers removed)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(results_dir, "summary_table.png"), dpi=150)
    plt.close(fig)
    print("  → summary_table.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory containing benchmark CSV results",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        sys.exit(1)

    runs = load_run_data(args.results_dir)
    if not runs:
        print("No run data found. Run the benchmark first:")
        print("  python -m jumper_extension.monitor.benchmark.run_benchmark")
        sys.exit(1)

    print(f"Loaded {len(runs)} runs from {args.results_dir}")
    print("Generating plots …")

    #plot_run_chart(runs, args.results_dir)
    plot_cumulative(runs, args.results_dir)
    plot_iat_histogram(runs, args.results_dir)
    plot_summary_table(args.results_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
