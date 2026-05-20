#!/usr/bin/env python3
"""Monitor backend benchmark.

Runs each monitor implementation (thread, subprocess_python, native_c)
at several sampling frequencies while a CPU-heavy workload saturates all
available cores.  Each configuration is repeated multiple times;
outliers are removed and mean ± std are reported.

Results are saved as CSV files and visualised with three plot types:

    A. Run-chart  – binary hit/miss step plot + moving-average overlay
    B. Cumulative success curve – actual vs ideal sample count over time
    C. Histogram / KDE of inter-arrival times – jitter & tail behaviour

Usage
-----
    python -m jumper_extension.monitor.benchmark.run_benchmark [--duration 60]

The script writes its outputs into the ``benchmark/results/`` directory
next to this file.
"""

import argparse
import atexit
import faulthandler
import multiprocessing
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
import psutil


def _cleanup_children():
    """Kill any remaining child processes on exit."""
    for child in psutil.Process().children(recursive=True):
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

atexit.register(_cleanup_children)

# ---------------------------------------------------------------------------
# Monitor factories
# ---------------------------------------------------------------------------

BACKENDS = {
    "thread": lambda: _make_thread_monitor(),
    "subprocess_python": lambda: _make_subprocess_monitor(),
    "native_c": lambda: _make_native_c_monitor(),
}

FREQUENCIES = [1, 2, 4, 8, 16]  # Hz


def _make_thread_monitor():
    from jumper_extension.monitor.backends.thread import PerformanceMonitor
    return PerformanceMonitor()


def _make_subprocess_monitor():
    from jumper_extension.monitor.backends.subprocess_python import (
        SubprocessPerformanceMonitor,
    )
    return SubprocessPerformanceMonitor()


def _make_native_c_monitor():
    from jumper_extension.monitor.backends.native_c import (
        CSubprocessPerformanceMonitor,
    )
    return CSubprocessPerformanceMonitor()


# ---------------------------------------------------------------------------
# CPU workload
# ---------------------------------------------------------------------------

def _cpu_burn(stop_event):
    """Pure-Python busy loop to saturate one core."""
    while not stop_event.is_set():
        s = 0
        for i in range(50_000):
            s += i * i


def _available_cpus():
    """Return the number of CPUs available to this process.

    Respects SLURM's ``SLURM_CPUS_PER_TASK`` / ``SLURM_CPUS_ON_NODE``,
    cgroup limits (``os.sched_getaffinity``), and falls back to
    ``os.cpu_count()`` only as a last resort.
    """
    # 1. SLURM environment (most reliable on shared HPC nodes)
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    # 2. cgroup / taskset affinity
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    # 3. Fallback
    return os.cpu_count() or 4


def _print_cpu_diagnostics():
    """Print all CPU detection methods for debugging."""
    print("\nCPU detection diagnostics:")
    print(f"  SLURM_CPUS_PER_TASK:  {os.environ.get('SLURM_CPUS_PER_TASK', '<not set>')}")
    print(f"  SLURM_CPUS_ON_NODE:   {os.environ.get('SLURM_CPUS_ON_NODE', '<not set>')}")
    print(f"  SLURM_JOB_CPUS_PER_NODE: {os.environ.get('SLURM_JOB_CPUS_PER_NODE', '<not set>')}")
    try:
        aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = '<unavailable>'
    print(f"  os.sched_getaffinity: {aff}")
    print(f"  os.cpu_count():       {os.cpu_count()}")
    print(f"  → _available_cpus():  {_available_cpus()}")
    print()


def start_workload(n_workers=None):
    """Spawn *n_workers* processes (default: available CPUs) doing busy work."""
    if n_workers is None:
        n_workers = _available_cpus()
    stop_event = multiprocessing.Event()
    workers = []
    for _ in range(n_workers):
        p = multiprocessing.Process(target=_cpu_burn, args=(stop_event,), daemon=True)
        p.start()
        workers.append(p)
    return workers, stop_event


def stop_workload(workers, stop_event):
    stop_event.set()
    # Send SIGTERM to all immediately — no need for graceful shutdown
    for p in workers:
        try:
            if p.is_alive():
                p.terminate()
        except (OSError, ValueError):
            pass
    # Give them a brief moment to exit, then force-kill stragglers
    for p in workers:
        try:
            p.join(timeout=0.5)
            if p.is_alive():
                p.kill()
        except (OSError, ValueError):
            pass
    for p in workers:
        try:
            p.join(timeout=0.5)
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Experiment overview helpers
# ---------------------------------------------------------------------------

def _count_level_pids(monitor):
    """Return a dict mapping each active level to its PID count."""
    counts = {}
    for level in getattr(monitor, "levels", []):
        df = monitor.nodes.view(level=level)
        counts[level] = len(df)
    return counts


def _system_task_count_from_loadavg():
    """Read the system-wide task count from ``/proc/loadavg``.

    The fourth field of ``/proc/loadavg`` is ``running/total`` — the
    total being the number of tasks (kernel view, includes threads)
    currently in the system.  Unlike ``/proc/<pid>`` entries this file
    remains readable by unprivileged users even when ``/proc`` is
    mounted with ``hidepid=1``/``hidepid=2`` (common on shared HPC
    nodes).  Returns ``None`` on non-Linux systems or parse errors.
    """
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
        if len(parts) >= 4 and "/" in parts[3]:
            return int(parts[3].split("/", 1)[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _snapshot_process_counts():
    """Snapshot process counts right now (call while workload is running).

    The counters for user/slurm are computed in a single pass over
    ``psutil.process_iter`` with per-process exception handling so that
    one unreadable /proc entry (e.g. a race with a dying process, a
    ``ZombieProcess``, or a ``hidepid``-restricted entry) does not
    invalidate the totals.
    """
    from jumper_extension.utilities import is_slurm_available
    uid = os.getuid()
    slurm_job_id = (
        os.environ.get("SLURM_JOB_ID", "") if is_slurm_available() else ""
    )

    try:
        n_process_tree = 1 + len(psutil.Process().children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        n_process_tree = -1

    n_psutil_visible = 0
    n_user = 0
    n_slurm = 0 if slurm_job_id else None

    try:
        iterator = psutil.process_iter(["pid", "uids"])
    except Exception:
        iterator = iter(())

    while True:
        try:
            p = next(iterator)
        except StopIteration:
            break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            # Catastrophic iterator failure — stop but keep partial counts.
            break

        n_psutil_visible += 1

        # User filter
        try:
            info_uids = p.info.get("uids") if isinstance(p.info, dict) else None
            is_user = bool(info_uids) and info_uids.real == uid
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            is_user = False

        if is_user:
            n_user += 1

        # Slurm filter — reading /proc/<pid>/environ is only possible
        # for our own processes, so skip non-user PIDs to avoid tons of
        # guaranteed-to-fail syscalls on busy nodes.
        if slurm_job_id and is_user:
            try:
                if p.environ().get("SLURM_JOB_ID") == slurm_job_id:
                    n_slurm += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    # psutil.process_iter only returns processes the caller can see via
    # /proc/<pid>.  On systems with hidepid=1/2 this collapses to the
    # current user's processes, which makes n_psutil_visible == n_user.
    # Fall back to the system-wide task count from /proc/loadavg when
    # it is clearly larger than what psutil can see.
    nr_tasks = _system_task_count_from_loadavg()
    if nr_tasks is not None and nr_tasks > n_psutil_visible:
        n_system = nr_tasks
        system_source = "tasks, /proc/loadavg"
    else:
        n_system = n_psutil_visible
        system_source = "processes, psutil"

    return {
        "process_tree": n_process_tree,
        "user": n_user,
        "uid": uid,
        "system": n_system,
        "system_source": system_source,
        "slurm": n_slurm,
    }


def print_experiment_overview(monitor, n_workers, proc_counts):
    """Print PID / process counts for each level."""
    print(f"\n{'─'*60}")
    print("Experiment overview")
    print(f"{'─'*60}")
    print(f"  CPUs (available):       {_available_cpus()}")
    print(f"  CPUs (total on node):   {os.cpu_count()}")
    print(f"  Burn workers:           {n_workers}")
    print(f"  Active levels:          {getattr(monitor, 'levels', '?')}")
    print(f"  Processes per level (during workload):")
    uid = proc_counts.get("uid", "?")
    print(f"    process (PID tree):   {proc_counts.get('process_tree', '?')}")
    print(f"    user    (uid={uid}): {' ' * max(0, 4 - len(str(uid)))}{proc_counts.get('user', '?')}")
    system_source = proc_counts.get("system_source", "")
    system_label = f"system  ({system_source}):" if system_source else "system  (all):"
    print(f"    {system_label:<22s}{proc_counts.get('system', '?')}")
    n_slurm = proc_counts.get("slurm")
    if n_slurm is not None:
        print(f"    slurm   (job={os.environ.get('SLURM_JOB_ID', '?')}): "
              f"{n_slurm}")

    # Per-level sample counts from last run
    print(f"  Samples from last run:")
    for level in getattr(monitor, "levels", []):
        df = monitor.nodes.view(level=level)
        print(f"    {level:>8s}:             {len(df)}")
    print(f"{'─'*60}")


def _proc_in_slurm_job(proc, slurm_job_id):
    """Check if a process belongs to the given SLURM job."""
    if not slurm_job_id:
        return False
    try:
        env = proc.environ()
        return env.get("SLURM_JOB_ID") == slurm_job_id
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return False


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_single(backend_name, freq_hz, duration_sec, n_workers=None):
    """Run one benchmark: backend × frequency.

    Returns a dict with summary statistics and the raw DataFrame, or
    None if no data was collected.
    """
    interval = 1.0 / freq_hz
    expected_samples = int(duration_sec * freq_hz)

    monitor = BACKENDS[backend_name]()

    # Hard timeout
    deadline = duration_sec + 30
    old_alarm = None
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Run exceeded hard deadline of {deadline}s")
    if hasattr(signal, "SIGALRM"):
        old_alarm = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(deadline))

    # Watchdog: if this run is still inside run_single() past the
    # deadline, dump the full stack of every thread to stderr so we can
    # diagnose where it hangs.  Repeats every <deadline>s until the run
    # returns (at which point we cancel it in the finally block).
    # faulthandler.dump_traceback_later is safe from async-signal context
    # and does not interfere with SIGALRM.
    faulthandler.enable()
    try:
        faulthandler.dump_traceback_later(
            timeout=int(deadline),
            repeat=True,
            file=sys.stderr,
        )
    except (RuntimeError, ValueError):
        # Older Pythons or already-armed timer — ignore.
        pass

    workers = None
    stop_event = None
    t_wall_start = time.perf_counter()
    try:
        workers, stop_event = start_workload(n_workers=n_workers)
        print(f"({len(workers)} burn workers) ", end="", flush=True)
        time.sleep(0.5)
        monitor.start(interval=interval)
        t_setup_done = time.perf_counter()

        # --- measurement window ---
        t_start = time.perf_counter()

        # Snapshot process counts while workload is running
        proc_counts = _snapshot_process_counts()

        time.sleep(duration_sec)
        t_end = time.perf_counter()

        # --- teardown ---
        t_teardown_start = time.perf_counter()
        stop_workload(workers, stop_event)
        monitor.stop()
        t_teardown_done = time.perf_counter()
    except TimeoutError as exc:
        t_end = time.perf_counter()
        t_setup_done = t_setup_done if 't_setup_done' in dir() else t_end
        t_teardown_start = time.perf_counter()
        print(f"      ⚠ {exc}")
        if workers and stop_event:
            try:
                stop_workload(workers, stop_event)
            except Exception:
                pass
        try:
            monitor.stop()
        except Exception:
            pass
        t_teardown_done = time.perf_counter()
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_alarm is not None:
                signal.signal(signal.SIGALRM, old_alarm)
        try:
            faulthandler.cancel_dump_traceback_later()
        except (RuntimeError, AttributeError):
            pass
        # Always ensure workers are dead
        if workers and stop_event:
            try:
                stop_workload(workers, stop_event)
            except Exception:
                pass

    # --- inter-run cleanup ---
    # The collector may have reniced our process to +19.  Restore to 0
    # so the next run starts with a clean scheduling state.
    try:
        os.nice(-os.nice(0))  # reset to 0
    except (OSError, PermissionError):
        pass

    # Briefly wait so the OS can fully reclaim child resources and
    # avoid leftover scheduling artifacts leaking into the next run.
    time.sleep(1.0)

    # Extract the "process" level data
    if not monitor.nodes.node_names():
        return None
    df = monitor.nodes.view(level="process")
    if df.empty:
        return None

    df = df.copy()
    t0 = df["time"].iloc[0]
    df["time_rel"] = df["time"] - t0
    df["inter_arrival"] = df["time"].diff()
    df["hit"] = df["inter_arrival"].le(interval * 1.5)
    df.loc[df.index[0], "hit"] = True

    actual_duration = t_end - t_start
    setup_time = t_setup_done - t_wall_start
    teardown_time = t_teardown_done - t_teardown_start
    total_wall = t_teardown_done - t_wall_start
    n_actual = len(df)

    return {
        "backend": backend_name,
        "freq_hz": freq_hz,
        "interval": interval,
        "duration": actual_duration,
        "expected": expected_samples,
        "actual": n_actual,
        "hit_rate": min(100.0, n_actual / expected_samples * 100) if expected_samples else 0,
        "mean_iat": df["inter_arrival"].mean(),
        "median_iat": df["inter_arrival"].median(),
        "p95_iat": df["inter_arrival"].quantile(0.95),
        "p99_iat": df["inter_arrival"].quantile(0.99),
        "max_iat": df["inter_arrival"].max(),
        "setup_time": setup_time,
        "teardown_time": teardown_time,
        "total_wall": total_wall,
        "proc_counts": proc_counts,
        "df": df,
        "monitor": monitor,
    }


# ---------------------------------------------------------------------------
# Outlier removal (IQR on hit_rate)
# ---------------------------------------------------------------------------

def remove_outliers(rows):
    """Drop runs whose hit_rate is an IQR outlier.  Returns filtered list."""
    if len(rows) < 4:
        return rows
    rates = np.array([r["hit_rate"] for r in rows])
    q1, q3 = np.percentile(rates, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    kept = [r for r in rows if lo <= r["hit_rate"] <= hi]
    n_removed = len(rows) - len(kept)
    if n_removed:
        print(f"      (removed {n_removed} outlier(s) by IQR on hit_rate)")
    return kept if kept else rows  # never discard all


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

_SANITY_N_SAMPLES = 3  # collect this many samples at 1 Hz (≈3 s)

# Base metrics that must be present and have no NaN values.
# The first sample is excluded from the zero-check because CPU utilisation
# is always 0 on the very first tick (it's a delta metric).
_REQUIRED_METRICS = [
    "time", "cpu_util_avg", "cpu_util_min", "cpu_util_max",
    "memory",
    "io_read", "io_write", "io_read_count", "io_write_count",
]
# Columns where all-zeros is only acceptable at certain levels.
# IO rates can be zero at process/user/slurm level when idle, but
# system-level IO should always show some activity.
_ZERO_CHECK_SKIP_LEVELS = {
    "cpu_util_min": {"system"},  # system-level min can be 0 if some cores are idle
    # io_read (read_bytes in /proc/<pid>/io) counts physical disk reads.
    # Reads served from page cache register as 0.  Our sanity IO worker
    # writes + fsyncs (→ io_write > 0) then reads back from cache
    # (→ io_read may stay 0).  io_read_count (syscr) is always > 0.
    "io_read": {"process", "user", "slurm"},
}
_GPU_METRICS = [
    "gpu_util_avg", "gpu_util_min", "gpu_util_max",
    "gpu_band_avg", "gpu_band_min", "gpu_band_max",
    "gpu_mem_avg", "gpu_mem_min", "gpu_mem_max",
]


def _sanity_check(backend_name):
    """Run a short sanity check for *backend_name*.

    Starts the monitor (without CPU burn) for a few seconds and verifies
    that the collected data contains the expected columns with non-NaN
    values and that metrics like CPU, memory, and IO are actually being
    retrieved.  Returns True on success, False on failure (with
    diagnostics printed).
    """
    freq_hz = 1
    interval = 1.0 / freq_hz
    monitor = BACKENDS[backend_name]()

    import tempfile, threading

    # Background thread that generates continuous IO so that
    # io_read/io_write counters are non-zero at every level.
    io_stop = threading.Event()
    tmp_path = None

    def _io_worker():
        nonlocal tmp_path
        f = tempfile.NamedTemporaryFile(delete=False, prefix="jumper_sanity_")
        tmp_path = f.name
        buf = b"x" * (64 * 1024)  # 64 KB chunks
        while not io_stop.is_set():
            f.write(buf)
            f.flush()
            os.fsync(f.fileno())
            f.seek(0)
            _ = f.read()
            f.seek(0)
            f.truncate()
        f.close()

    io_thread = threading.Thread(target=_io_worker, daemon=True)
    io_thread.start()

    monitor.start(interval=interval)
    # Wait up to 10s, but stop early once we have enough samples.
    required = _SANITY_N_SAMPLES + 1  # +1 because first sample's CPU delta is 0
    hard_deadline = time.monotonic() + 10.0
    enough = False
    while time.monotonic() < hard_deadline:
        time.sleep(0.5)
        if monitor.nodes.node_names():
            levels = getattr(monitor, "levels", ["process"])
            counts = [
                len(monitor.nodes.view(level=lv))
                for lv in levels
            ]
            if counts and min(counts) >= required:
                enough = True
                break
    monitor.stop()

    io_stop.set()
    io_thread.join(timeout=2)
    if tmp_path:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not monitor.nodes.node_names():
        print("    FAIL: no data collected (monitor has no registered nodes)")
        return False

    if not enough:
        # Print what we got so far for debugging
        levels = getattr(monitor, "levels", ["process"])
        for lv in levels:
            df = monitor.nodes.view(level=lv)
            n = len(df)
            print(f"    DEBUG [{lv}]: got {n}/{required} samples")
            if not df.empty:
                print(f"           columns: {list(df.columns[:10])}...")
                print(f"           first row: {df.iloc[0].to_dict()}")
        print(f"    FAIL: timed out waiting for {required} samples")
        return False

    ok = True
    for level in getattr(monitor, "levels", ["process"]):
        df = monitor.nodes.view(level=level)
        if df.empty:
            print(f"    FAIL [{level}]: no samples collected")
            ok = False
            continue

        n_samples = len(df)
        if n_samples < 2:
            print(f"    FAIL [{level}]: only {n_samples} sample(s), "
                  f"need ≥2 for delta metrics")
            ok = False
            continue

        # For zero-checks, skip the first row (CPU delta is 0, IO rates
        # are meaningless on the very first tick).
        df_check = df.iloc[1:]

        # Check required base metrics
        for col in _REQUIRED_METRICS:
            if col not in df.columns:
                print(f"    FAIL [{level}]: missing column '{col}'")
                ok = False
                continue
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                print(f"    FAIL [{level}]: column '{col}' has "
                      f"{n_nan}/{n_samples} NaN values")
                ok = False
            # All-zeros check (skip "time" — it legitimately starts near 0)
            if col != "time" and len(df_check) > 0:
                skip_levels = _ZERO_CHECK_SKIP_LEVELS.get(col, set())
                if level not in skip_levels and (df_check[col] == 0).all():
                    print(f"    FAIL [{level}]: column '{col}' is all zeros "
                          f"(after skipping first sample)")
                    ok = False

        # Check per-core CPU columns exist
        cpu_cols = [c for c in df.columns if c.startswith("cpu_util_")
                    and c not in ("cpu_util_avg", "cpu_util_min", "cpu_util_max")]
        if not cpu_cols:
            print(f"    FAIL [{level}]: no per-core CPU columns found")
            ok = False

        # Check GPU metrics if GPU is available
        has_gpu = getattr(monitor, "num_gpus", 0) > 0
        if has_gpu:
            for col in _GPU_METRICS:
                if col not in df.columns:
                    print(f"    FAIL [{level}]: GPU detected but missing "
                          f"column '{col}'")
                    ok = False
                    continue
                n_nan = df[col].isna().sum()
                if n_nan > 0:
                    print(f"    FAIL [{level}]: GPU column '{col}' has "
                          f"{n_nan}/{n_samples} NaN values")
                    ok = False

        if ok:
            metrics_summary = (
                f"cpu_avg={df_check['cpu_util_avg'].mean():.1f}%, "
                f"mem={df_check['memory'].mean():.1f}MB, "
                f"io_r={df_check['io_read'].mean():.0f}, "
                f"io_w={df_check['io_write'].mean():.0f}"
            )
            if has_gpu:
                metrics_summary += (
                    f", gpu_util={df_check['gpu_util_avg'].mean():.1f}%"
                )
            print(f"    OK   [{level}]: {n_samples} samples, {metrics_summary}")

    return ok


def run_sanity_checks(backends):
    """Run sanity checks for all requested backends.

    Returns the list of backends that passed.
    """
    print(f"\n{'='*60}")
    print(f"Sanity checks ({_SANITY_N_SAMPLES} samples @ 1 Hz, no CPU burn)")
    print(f"{'='*60}")

    passed = []
    for backend_name in backends:
        print(f"\n  {backend_name}:")
        try:
            if _sanity_check(backend_name):
                passed.append(backend_name)
            else:
                print(f"  → {backend_name} FAILED sanity check, skipping.")
        except Exception as exc:
            print(f"    FAIL: {exc}")
            print(f"  → {backend_name} FAILED sanity check, skipping.")

        # Cool down between checks
        time.sleep(1.0)
        psutil.process_iter.cache_clear()

    print(f"\n  Passed: {len(passed)}/{len(backends)} "
          f"({', '.join(passed) if passed else 'none'})")
    print(f"{'='*60}\n")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Monitor benchmark")
    parser.add_argument("--duration", type=int, default=60,
                        help="Workload duration in seconds (default: 60)")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of repetitions per configuration "
                             "(default: 10)")
    parser.add_argument("--backends", type=str, default=None,
                        help="Comma-separated list of backends to test "
                             "(default: all)")
    parser.add_argument("--frequencies", type=str, default=None,
                        help="Comma-separated list of frequencies in Hz "
                             "(default: 1,2,4,8,16)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU burn workers "
                             "(default: auto-detect from SLURM / affinity)")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Skip the initial sanity checks")
    args = parser.parse_args()

    _print_cpu_diagnostics()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    backends = (
        args.backends.split(",") if args.backends
        else list(BACKENDS.keys())
    )
    frequencies = (
        [float(f) for f in args.frequencies.split(",")]
        if args.frequencies else FREQUENCIES
    )

    n_repeats = args.repeats
    overview_printed = False
    agg_summaries = []

    # --- Sanity checks ---
    if not args.skip_sanity:
        backends = run_sanity_checks(backends)
        if not backends:
            print("All backends failed sanity checks. Aborting.")
            sys.exit(1)
    else:
        print("\n(Sanity checks skipped)\n")

    for backend_name in backends:
        if backend_name not in BACKENDS:
            print(f"Unknown backend: {backend_name!r}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"Backend: {backend_name}")
        print(f"{'='*60}")

        for freq in frequencies:
            interval = 1.0 / freq
            expected = int(args.duration * freq)
            print(f"\n  {freq} Hz (interval={interval:.3f}s), "
                  f"expected≈{expected}, repeats={n_repeats}",
                  flush=True)

            run_rows = []
            all_dfs = []
            last_monitor = None
            for rep in range(1, n_repeats + 1):
                print(f"    run {rep}/{n_repeats} …", end=" ", flush=True)
                try:
                    result = run_single(backend_name, freq, args.duration,
                                         n_workers=args.workers)
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    continue
                if result is None:
                    print("no data")
                    continue

                # Clear psutil's internal cache between runs
                psutil.process_iter.cache_clear()

                n = result["actual"]
                pct = result["hit_rate"]
                dur = result["duration"]
                setup = result["setup_time"]
                td = result["teardown_time"]
                wall = result["total_wall"]
                print(f"{n}/{expected} ({pct:.1f}%) "
                      f"[measure={dur:.1f}s, setup={setup:.1f}s, "
                      f"teardown={td:.1f}s, total={wall:.1f}s]")

                all_dfs.append(result["df"])
                last_monitor = result.pop("monitor")
                last_proc_counts = result.pop("proc_counts", {})
                result.pop("df")
                result["rep"] = rep
                run_rows.append(result)

            if not run_rows:
                print("    ⚠ All runs failed, skipping.")
                continue

            # Print experiment overview once (from the last successful run)
            if not overview_printed and last_monitor is not None:
                print_experiment_overview(
                    last_monitor,
                    _available_cpus(),
                    last_proc_counts,
                )
                overview_printed = True

            # Outlier removal
            kept = remove_outliers(run_rows)

            # Save all per-run raw data (use the median-hit-rate run
            # as the representative for per-sample plots)
            rates = [r["hit_rate"] for r in kept]
            median_idx = int(np.argmin(
                np.abs(np.array(rates) - np.median(rates))
            ))
            # Save representative raw data
            rep_df = all_dfs[kept[median_idx]["rep"] - 1]
            tag = f"{backend_name}_{freq}Hz"
            rep_df.to_csv(
                os.path.join(results_dir, f"{tag}.csv"), index=False
            )

            # Aggregate statistics
            metrics = [
                "hit_rate", "mean_iat", "median_iat",
                "p95_iat", "p99_iat", "max_iat", "actual", "duration",
            ]
            agg = {
                "backend": backend_name,
                "freq_hz": freq,
                "interval": interval,
                "expected": expected,
                "n_runs": len(kept),
            }
            for m in metrics:
                vals = np.array([r[m] for r in kept])
                agg[f"{m}_mean"] = np.mean(vals)
                agg[f"{m}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0
            agg_summaries.append(agg)

            print(f"    → avg hit_rate: "
                  f"{agg['hit_rate_mean']:.1f}% "
                  f"± {agg['hit_rate_std']:.1f}%  "
                  f"({len(kept)} runs)")

    # ---- Final summary ----
    if agg_summaries:
        summary = pd.DataFrame(agg_summaries)
        summary_path = os.path.join(results_dir, "summary.csv")
        summary.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print("Aggregated Summary (mean ± std)")
        print(f"{'='*60}")
        display_cols = [
            "backend", "freq_hz", "expected", "n_runs",
            "actual_mean", "actual_std",
            "hit_rate_mean", "hit_rate_std",
            "duration_mean",
            "mean_iat_mean", "p95_iat_mean", "max_iat_mean",
        ]
        display_cols = [c for c in display_cols if c in summary.columns]
        print(summary[display_cols].to_string(index=False, float_format="%.3f"))
        print(f"\nResults written to: {results_dir}/")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
