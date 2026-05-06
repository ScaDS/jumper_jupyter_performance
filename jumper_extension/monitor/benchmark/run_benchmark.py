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
import random
import signal
import sys
import time
from queue import Empty as QueueEmpty

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
# CPU workload — parallel Monte-Carlo Pi via dart-throwing
# ---------------------------------------------------------------------------
#
# Why dart-throwing instead of an open-ended busy loop?
#
# The previous implementation spawned N infinite ``_cpu_burn`` workers
# and SIGTERM'd them after a fixed wallclock window.  That gives a hit-
# rate measurement (samples observed vs samples expected) but says
# nothing about the *runtime cost* the monitor imposes on the workload.
#
# The new workload is a fixed-size, embarrassingly-parallel Monte Carlo
# Pi estimation.  Each worker throws a fixed number of darts at a unit
# square and counts how many fall inside the unit quarter-circle; the
# parent aggregates the hits into a Pi estimate.  Because the work is
# *finite and identical every run*, we can:
#
#   * calibrate ``n_darts_total`` once (without any monitor running) so
#     the workload takes a target wallclock duration (default ≈ 30 s),
#   * re-run the same workload with a monitor active, and
#   * report ``runtime overhead`` = (t_with_monitor − t_baseline) /
#     t_baseline as an additional benchmark axis next to the hit-rate
#     of achievable sampling frequencies.
#
# Implementation notes:
#   * Pure-Python tight loop saturates one core per worker (the C
#     ``random()`` call still releases the GIL, but each worker is its
#     own process).
#   * Hits are returned via a ``multiprocessing.Queue``; workers
#     self-terminate when done, no SIGTERM dance required.
#   * Seeds are derived from the parent PID + worker index + start
#     time, so two workers in the same run never produce identical
#     streams, but a single calibration is reproducible enough to be
#     stable across repeats on the same node.


def _dart_worker(n_darts: int, seed: int, result_queue) -> None:
    """Throw ``n_darts`` darts at the unit square, push hit count.

    A "hit" is a dart that lands inside the unit quarter-circle
    (``x*x + y*y <= 1``).  The function is a pure-Python tight loop on
    purpose: that is what we want to monitor.
    """
    rng = random.Random(seed)
    rand = rng.random  # local binding speeds the loop up measurably
    hits = 0
    for _ in range(n_darts):
        x = rand()
        y = rand()
        if x * x + y * y <= 1.0:
            hits += 1
    try:
        result_queue.put(hits)
    except Exception:
        # Parent may have torn down the queue already (timeout path);
        # nothing useful we can do from a child here.
        pass


def _spawn_dart_workers(n_darts_total: int, n_workers: int):
    """Fork ``n_workers`` dart workers and return ``(workers, queue, per_worker)``.

    Splits the dart budget evenly; any remainder is dropped (≤ n_workers
    darts) so each worker does exactly the same amount of work, keeping
    the CPU load symmetric and the runtime tight.
    """
    per_worker = max(1, n_darts_total // n_workers)
    queue: multiprocessing.Queue = multiprocessing.Queue()
    workers = []
    base_seed = (os.getpid() << 16) ^ int(time.perf_counter_ns() & 0xFFFFFFFF)
    for i in range(n_workers):
        p = multiprocessing.Process(
            target=_dart_worker,
            args=(per_worker, base_seed ^ (i + 1), queue),
            daemon=True,
        )
        p.start()
        workers.append(p)
    return workers, queue, per_worker


def _await_dart_workers(workers, queue, per_worker, hard_timeout=None):
    """Drain ``queue`` until every worker reports, then reap children.

    Returns ``(pi_estimate, n_workers_reported, total_hits, total_darts)``.
    A ``QueueEmpty`` from the timeout aborts early; remaining workers
    are SIGKILL'd to keep the system clean.
    """
    n_workers = len(workers)
    deadline = (time.perf_counter() + hard_timeout) if hard_timeout else None
    total_hits = 0
    n_done = 0
    for _ in range(n_workers):
        if deadline is not None:
            timeout = max(0.0, deadline - time.perf_counter())
        else:
            timeout = None
        try:
            total_hits += queue.get(timeout=timeout)
            n_done += 1
        except QueueEmpty:
            break
        except (EOFError, OSError):
            break
    # Reap workers (they should already be exiting on their own).
    for p in workers:
        try:
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
                p.join(timeout=1)
        except (OSError, ValueError):
            pass
    total_darts = per_worker * n_workers
    pi = (4.0 * total_hits / total_darts) if total_darts else float("nan")
    return pi, n_done, total_hits, total_darts


def _run_dart_workload(n_darts_total, n_workers, hard_timeout=None):
    """Run one full dart-throwing workload, return ``(pi, elapsed, n_done)``."""
    workers, queue, per_worker = _spawn_dart_workers(n_darts_total, n_workers)
    t0 = time.perf_counter()
    pi, n_done, _hits, _total = _await_dart_workers(
        workers, queue, per_worker, hard_timeout=hard_timeout
    )
    elapsed = time.perf_counter() - t0
    return pi, elapsed, n_done


# ---------------------------------------------------------------------------
# Calibration: choose n_darts so the workload takes ~target_sec, then
# measure the unmonitored baseline runtime over several repeats.
# ---------------------------------------------------------------------------

PROBE_DARTS_PER_WORKER = 200_000
# Calibration is considered converged once a full-sized run lands
# within ``CALIB_TOLERANCE`` of the requested target.  Anything tighter
# would just be noise from jitter between repeats.
CALIB_TOLERANCE = 0.15  # 15 %
CALIB_MAX_ITER = 3


def _round_per_worker(n_darts_total, n_workers):
    """Round so each worker gets the same tidy (multiple-of-1000) count."""
    per = max(1000, ((n_darts_total // n_workers) // 1000) * 1000)
    return per * n_workers, per


def calibrate_workload(n_workers, target_sec=30.0, n_repeats=10):
    """Find ``n_darts_total`` so a single workload takes ≈ ``target_sec``.

    Returns ``(n_darts_total, baseline_mean, baseline_std, baseline_times)``.

    A tiny probe on its own is not enough: with N workers ≫ CPU cores
    (as in the Slurm script) the probe is fork-bound and overestimates
    steady-state throughput by several ×, producing dart counts whose
    real runtime is far above ``target_sec``.  We therefore iterate:

    1. **Probe** — very cheap, gives a first-order rate estimate.
    2. **Calibration loop** — run one *full-sized* workload; if its
       runtime is outside ±``CALIB_TOLERANCE`` of ``target_sec``, rescale
       ``n_darts_total`` by the observed ratio and try again (up to
       ``CALIB_MAX_ITER`` times).  Each iteration runs the actual
       target-sized workload, so it directly measures the steady-state
       rate under the real scheduling regime.
    3. **Baseline** — ``n_repeats`` full-size runs form the unmonitored
       reference used to compute per-run overhead percentages.
    """
    print(f"\n{'='*60}")
    print(f"Calibration  (target≈{target_sec:.0f}s, "
          f"workers={n_workers}, repeats={n_repeats})")
    print(f"{'='*60}")

    hard = target_sec * 3 + 30

    # 1) Probe -----------------------------------------------------------
    probe_total = PROBE_DARTS_PER_WORKER * n_workers
    print(f"  probe: {probe_total:,} darts … ", end="", flush=True)
    pi_probe, t_probe, n_done = _run_dart_workload(
        probe_total, n_workers, hard_timeout=120
    )
    if n_done < n_workers or t_probe <= 0:
        raise RuntimeError(
            f"probe failed: only {n_done}/{n_workers} workers reported "
            f"after {t_probe:.2f}s"
        )
    rate = probe_total / t_probe
    print(f"{t_probe:.2f}s  ({rate:.2e} darts/s, π≈{pi_probe:.4f})")
    print(f"  (probe rate is a first-order estimate; the next step "
          f"validates it at full size)")

    # 2) Iterative correction at full size -------------------------------
    # ``rate`` initialises n_darts_total; subsequent iterations refine
    # it until a full-sized run lands within the tolerance band.
    n_darts_total, per = _round_per_worker(
        max(1000 * n_workers, int(rate * target_sec)), n_workers
    )
    last_t = None
    for it in range(1, CALIB_MAX_ITER + 1):
        print(f"  calib {it}/{CALIB_MAX_ITER}: {n_darts_total:,} darts "
              f"({per:,}/worker) … ", end="", flush=True)
        pi_cal, t_cal, n_done = _run_dart_workload(
            n_darts_total, n_workers, hard_timeout=hard,
        )
        if n_done < n_workers or t_cal <= 0:
            raise RuntimeError(
                f"calibration iteration {it} failed: only "
                f"{n_done}/{n_workers} workers reported after {t_cal:.2f}s"
            )
        err = (t_cal - target_sec) / target_sec
        print(f"{t_cal:6.2f}s  (π≈{pi_cal:.4f}, error {err:+.1%})")
        last_t = t_cal
        if abs(err) <= CALIB_TOLERANCE:
            print(f"  → converged within ±{CALIB_TOLERANCE:.0%}")
            break
        # Rescale proportionally to hit the target on the next try.
        scale = target_sec / t_cal
        n_darts_total, per = _round_per_worker(
            max(1000 * n_workers, int(n_darts_total * scale)),
            n_workers,
        )
    else:
        print(f"  ⚠ calibration did not converge after {CALIB_MAX_ITER} "
              f"iterations (last={last_t:.2f}s vs target {target_sec:.0f}s); "
              f"proceeding with current dart count.")

    print(f"  target: {n_darts_total:,} darts "
          f"({per:,}/worker × {n_workers} workers)")

    # 3) Baseline repeats (no monitor) -----------------------------------
    times = []
    for r in range(1, n_repeats + 1):
        pi, t, n_done = _run_dart_workload(
            n_darts_total, n_workers, hard_timeout=hard,
        )
        times.append(t)
        ok = "ok" if n_done == n_workers else f"only {n_done}/{n_workers}"
        print(f"    baseline {r:2d}/{n_repeats}: {t:6.2f}s  "
              f"(π≈{pi:.4f}, {ok})")

    arr = np.array(times)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    print(f"  → baseline = {mean:.2f}s ± {std:.2f}s "
          f"(min {arr.min():.2f}, max {arr.max():.2f})")
    if abs(mean - target_sec) / target_sec > CALIB_TOLERANCE:
        print(f"  ⚠ baseline mean is {((mean-target_sec)/target_sec):+.1%} "
              f"off the {target_sec:.0f}s target — consider rerunning "
              f"with more baseline repeats or a different worker count.")
    print(f"{'='*60}\n")
    return n_darts_total, mean, std, times


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


# ---------------------------------------------------------------------------
# Experiment overview helpers
# ---------------------------------------------------------------------------

def _count_level_pids(monitor):
    """Return a dict mapping each active level to its PID count."""
    counts = {}
    for level in getattr(monitor, "levels", []):
        df = monitor.data.data.get(level, pd.DataFrame())
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
        df = monitor.data.data.get(level, pd.DataFrame())
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

def run_single(backend_name, freq_hz, n_darts_total, baseline_time, n_workers):
    """Run one benchmark: ``backend × frequency`` over the calibrated
    Pi-via-darts workload.

    Parameters
    ----------
    backend_name
        Key into :data:`BACKENDS`.
    freq_hz
        Sampling frequency for the monitor under test.
    n_darts_total
        Total number of darts (calibrated once per process to give a
        wallclock duration ≈ ``--duration``).
    baseline_time
        Mean unmonitored wallclock runtime of the same workload, used
        to compute the runtime overhead the monitor imposes.
    n_workers
        Number of dart-throwing worker processes.

    Returns ``dict`` with summary statistics and the raw DataFrame, or
    ``None`` if no usable data was collected.
    """
    interval = 1.0 / freq_hz

    # Hard timeout: 3× the unmonitored baseline plus a generous fixed
    # margin for monitor setup/teardown and Slurm/SSH stalls.  The
    # alarm raises ``TimeoutError`` from the main thread.
    deadline = max(60, int(baseline_time * 3 + 30))
    old_alarm = None
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Run exceeded hard deadline of {deadline}s")
    if hasattr(signal, "SIGALRM"):
        old_alarm = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(deadline)

    monitor = None
    workers = []
    queue = None
    per_worker = max(1, n_darts_total // n_workers)
    pi_estimate = float("nan")
    n_done = 0
    proc_counts = {}

    t_wall_start = time.perf_counter()
    t_workload_start = t_workload_end = t_wall_start
    t_teardown_done = t_wall_start

    try:
        # 1) Fork the dart workers FIRST.  Constructing the monitor
        #    afterwards keeps NVML's "do not fork after nvmlInit"
        #    contract intact (otherwise the parent segfaults at
        #    teardown — see the previous fix).
        workers, queue, per_worker = _spawn_dart_workers(
            n_darts_total, n_workers
        )
        print(f"({len(workers)} dart workers, {per_worker:,}/worker) ",
              end="", flush=True)

        # 2) Construct + start the monitor.  This is the moment from
        #    which the workload is being observed; we measure
        #    ``t_workload_start`` here so the reported duration
        #    matches what the monitor saw.
        monitor = BACKENDS[backend_name]()
        monitor.start(interval=interval)
        t_workload_start = time.perf_counter()

        # 3) Snapshot process counts once while the workload is running.
        proc_counts = _snapshot_process_counts()

        # 4) Block until every worker reports its hit count, or the
        #    SIGALRM deadline fires.
        deadline_abs = time.perf_counter() + deadline
        total_hits = 0
        for _ in range(n_workers):
            timeout = max(0.0, deadline_abs - time.perf_counter())
            try:
                total_hits += queue.get(timeout=timeout)
                n_done += 1
            except QueueEmpty:
                break
            except (EOFError, OSError):
                break
        t_workload_end = time.perf_counter()
        total_darts = per_worker * n_workers
        pi_estimate = (
            4.0 * total_hits / total_darts if total_darts else float("nan")
        )

        # 5) Stop the monitor before reaping children so the collector
        #    thread is guaranteed to have left its critical section
        #    before any data extraction below.
        monitor.stop()
        for p in workers:
            try:
                p.join(timeout=2)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=1)
            except (OSError, ValueError):
                pass
        t_teardown_done = time.perf_counter()
    except TimeoutError as exc:
        t_workload_end = time.perf_counter()
        print(f"      ⚠ {exc}")
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass
        for p in workers:
            try:
                if p.is_alive():
                    p.kill()
            except Exception:
                pass
        t_teardown_done = time.perf_counter()
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_alarm is not None:
                signal.signal(signal.SIGALRM, old_alarm)
        # Last-ditch: ensure no worker outlives this run.
        for p in workers:
            try:
                if p.is_alive():
                    p.kill()
                    p.join(timeout=1)
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

    # --- Extract the "process" level data --------------------------------
    if monitor is None or monitor.data is None:
        return None
    # If the collector thread did not terminate (slow tick under heavy
    # contention), reading the per-level DataFrames now would race with
    # pandas internals on the collector side and can segfault inside
    # ``concatenate_managers``.  Skip this run defensively.
    mon_thread = getattr(monitor, "monitor_thread", None)
    if mon_thread is not None and mon_thread.is_alive():
        print("      ⚠ collector thread still alive after stop(); "
              "skipping run to avoid pandas race")
        return None
    df = monitor.data.data.get("process", pd.DataFrame())
    if df.empty:
        return None

    df = df.copy()
    t0 = df["time"].iloc[0]
    df["time_rel"] = df["time"] - t0
    df["inter_arrival"] = df["time"].diff()
    df["hit"] = df["inter_arrival"].le(interval * 1.5)
    df.loc[df.index[0], "hit"] = True

    workload_duration = max(0.0, t_workload_end - t_workload_start)
    setup_time = max(0.0, t_workload_start - t_wall_start)
    teardown_time = max(0.0, t_teardown_done - t_workload_end)
    total_wall = max(0.0, t_teardown_done - t_wall_start)
    n_actual = len(df)

    # Expected samples = how many ticks should fit in the *actual*
    # workload window (monitor was active for exactly this long).
    expected_samples = max(1, int(round(workload_duration * freq_hz)))

    # Runtime overhead introduced by the monitor.  ``baseline_time`` is
    # the mean unmonitored runtime measured during calibration.
    overhead_s = workload_duration - baseline_time
    overhead_pct = (
        100.0 * overhead_s / baseline_time if baseline_time > 0 else float("nan")
    )

    return {
        "backend": backend_name,
        "freq_hz": freq_hz,
        "interval": interval,
        "duration": workload_duration,
        "baseline": baseline_time,
        "overhead_s": overhead_s,
        "overhead_pct": overhead_pct,
        "n_darts": n_darts_total,
        "pi_estimate": pi_estimate,
        "workers_finished": n_done,
        "expected": expected_samples,
        "actual": n_actual,
        "hit_rate": min(100.0, n_actual / expected_samples * 100),
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
        if monitor.data is not None:
            levels = getattr(monitor, "levels", ["process"])
            counts = [
                len(monitor.data.data.get(lv, pd.DataFrame()))
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

    if monitor.data is None:
        print(f"    FAIL: monitor.data is None")
        return False

    if not enough:
        # Print what we got so far for debugging
        levels = getattr(monitor, "levels", ["process"])
        for lv in levels:
            df = monitor.data.data.get(lv, pd.DataFrame())
            n = len(df)
            print(f"    DEBUG [{lv}]: got {n}/{required} samples")
            if not df.empty:
                print(f"           columns: {list(df.columns[:10])}...")
                print(f"           first row: {df.iloc[0].to_dict()}")
        print(f"    FAIL: timed out waiting for {required} samples")
        return False

    ok = True
    for level in getattr(monitor, "levels", ["process"]):
        df = monitor.data.data.get(level, pd.DataFrame())
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

def _install_global_watchdog(results_dir, period_sec=60):
    """Install a line-buffered watchdog that dumps tracebacks periodically.

    Writes to ``<results_dir>/watchdog.log`` using line-buffered IO so
    that entries show up promptly on shared filesystems, independent of
    Slurm's stdio capture.  Additionally registers ``SIGUSR1`` as a
    manual trigger so we can force a stack dump from outside the job::

        scancel --signal=USR1 <jobid>

    The ``SIGUSR1`` handler is chained, so it does not interfere with
    the default behaviour if other code also registers it.
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "watchdog.log")
    # Line-buffered so every written line hits the page cache immediately.
    wd_file = open(log_path, "a", buffering=1)
    wd_file.write(
        f"\n---- watchdog armed at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        f" pid={os.getpid()} period={period_sec}s ----\n"
    )
    wd_file.flush()

    faulthandler.enable(file=wd_file)
    # NOTE: ``faulthandler.dump_traceback_later(repeat=True)`` runs a
    # C-level watchdog thread that walks every Python thread's
    # ``PyThreadState`` from *outside* the GIL.  On CPython ≤3.10 this
    # has a known race against heavily-threaded psutil workloads (and
    # against process trees with many forks): the watchdog can observe
    # a thread mid-state-swap, after which that thread's next pure-
    # Python step crashes with "PyThreadState_Get: the function must
    # be called with the GIL held, but the GIL is released (the
    # current Python thread state is NULL)".  We therefore do *not*
    # arm the periodic dump; ``faulthandler.enable()`` above is still
    # active so genuine segfaults are reported with tracebacks, and
    # the per-run ``SIGALRM`` in ``run_single`` already provides a
    # hard timeout.  Manual stack dumps are still available via
    # SIGUSR1 (registered below).
    _ = period_sec  # kept in signature for API compatibility

    def _disarm_watchdog():
        try:
            wd_file.flush()
        except (OSError, ValueError):
            pass

    atexit.register(_disarm_watchdog)

    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(
                signal.SIGUSR1, file=wd_file, chain=False
            )
        except (RuntimeError, ValueError, OSError):
            pass

    print(f"[watchdog] segfault tracebacks → {log_path} "
          f"(periodic dump disabled to avoid CPython 3.10 tstate race)",
          flush=True)
    print(f"[watchdog] manual dump: scancel --signal=USR1 $SLURM_JOB_ID",
          flush=True)
    return wd_file


def main():
    parser = argparse.ArgumentParser(description="Monitor benchmark")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Target wallclock duration of the unmonitored "
                             "workload, in seconds.  The number of darts is "
                             "calibrated once at startup so a single run takes "
                             "≈ this long (default: 30).")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of monitored repetitions per "
                             "(backend, frequency) pair (default: 10)")
    parser.add_argument("--baseline-repeats", type=int, default=10,
                        help="Number of *unmonitored* repetitions used to "
                             "establish the runtime baseline against which "
                             "the monitor overhead is computed (default: 10)")
    parser.add_argument("--backends", type=str, default=None,
                        help="Comma-separated list of backends to test "
                             "(default: all)")
    parser.add_argument("--frequencies", type=str, default=None,
                        help="Comma-separated list of frequencies in Hz "
                             "(default: 1,2,4,8,16)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of dart-throwing worker processes "
                             "(default: auto-detect from SLURM / affinity)")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Skip the initial sanity checks")
    args = parser.parse_args()

    _print_cpu_diagnostics()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    _install_global_watchdog(results_dir, period_sec=60)

    backends = (
        args.backends.split(",") if args.backends
        else list(BACKENDS.keys())
    )
    frequencies = (
        [float(f) for f in args.frequencies.split(",")]
        if args.frequencies else FREQUENCIES
    )

    n_repeats = args.repeats
    n_workers = args.workers if args.workers is not None else _available_cpus()
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

    # --- Calibrate the workload (no monitor active) -------------------
    # The dart count is fixed for the entire run; ``baseline_time`` is
    # the reference against which every monitored run's duration is
    # compared to derive the runtime overhead %.
    n_darts_total, baseline_time, baseline_std, baseline_times = (
        calibrate_workload(
            n_workers=n_workers,
            target_sec=float(args.duration),
            n_repeats=int(args.baseline_repeats),
        )
    )
    # Persist the baseline so the plotter / downstream tools can read it.
    pd.DataFrame({
        "rep": list(range(1, len(baseline_times) + 1)),
        "duration_s": baseline_times,
    }).to_csv(os.path.join(results_dir, "baseline.csv"), index=False)
    with open(os.path.join(results_dir, "baseline.txt"), "w") as f:
        f.write(
            f"n_workers={n_workers}\n"
            f"n_darts_total={n_darts_total}\n"
            f"per_worker={n_darts_total // n_workers}\n"
            f"baseline_mean_s={baseline_time:.6f}\n"
            f"baseline_std_s={baseline_std:.6f}\n"
            f"baseline_min_s={min(baseline_times):.6f}\n"
            f"baseline_max_s={max(baseline_times):.6f}\n"
        )

    for backend_name in backends:
        if backend_name not in BACKENDS:
            print(f"Unknown backend: {backend_name!r}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"Backend: {backend_name}")
        print(f"{'='*60}")

        for freq in frequencies:
            interval = 1.0 / freq
            # Expected sample count assuming the workload still takes
            # ≈ baseline_time; per-run actuals are reported alongside.
            expected = max(1, int(round(baseline_time * freq)))
            print(f"\n  {freq} Hz (interval={interval:.3f}s), "
                  f"expected≈{expected} samples / "
                  f"baseline≈{baseline_time:.1f}s, "
                  f"repeats={n_repeats}",
                  flush=True)

            run_rows = []
            all_dfs = []
            last_monitor = None
            for rep in range(1, n_repeats + 1):
                print(f"    run {rep}/{n_repeats} …", end=" ", flush=True)
                try:
                    result = run_single(
                        backend_name, freq,
                        n_darts_total=n_darts_total,
                        baseline_time=baseline_time,
                        n_workers=n_workers,
                    )
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
                ovh = result["overhead_pct"]
                wall = result["total_wall"]
                pi = result["pi_estimate"]
                print(f"{n}/{expected} ({pct:.1f}%)  "
                      f"t={dur:.2f}s vs base {baseline_time:.2f}s "
                      f"(overhead {ovh:+.1f}%)  "
                      f"π≈{pi:.4f}  total={wall:.1f}s")

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
                    n_workers,
                    last_proc_counts,
                )
                overview_printed = True

            # Outlier removal (still on hit_rate; overhead follows
            # roughly linearly so outliers cluster together).
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
                "overhead_s", "overhead_pct",
            ]
            agg = {
                "backend": backend_name,
                "freq_hz": freq,
                "interval": interval,
                "expected": expected,
                "baseline_s": baseline_time,
                "n_darts": n_darts_total,
                "n_workers": n_workers,
                "n_runs": len(kept),
            }
            for m in metrics:
                vals = np.array([r[m] for r in kept], dtype=float)
                agg[f"{m}_mean"] = float(np.mean(vals))
                agg[f"{m}_std"] = (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                )
            agg_summaries.append(agg)

            print(f"    → avg hit_rate: "
                  f"{agg['hit_rate_mean']:.1f}% "
                  f"± {agg['hit_rate_std']:.1f}%   "
                  f"avg overhead: "
                  f"{agg['overhead_pct_mean']:+.1f}% "
                  f"± {agg['overhead_pct_std']:.1f}%   "
                  f"({len(kept)} runs)")

    # ---- Final summary ----
    if agg_summaries:
        summary = pd.DataFrame(agg_summaries)
        summary_path = os.path.join(results_dir, "summary.csv")
        summary.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print("Aggregated Summary (mean ± std)")
        print(f"  baseline (no monitor): "
              f"{baseline_time:.2f}s ± {baseline_std:.2f}s "
              f"over {len(baseline_times)} repeats, "
              f"{n_darts_total:,} darts on {n_workers} workers")
        print(f"{'='*60}")
        display_cols = [
            "backend", "freq_hz", "expected", "n_runs",
            "actual_mean", "actual_std",
            "hit_rate_mean", "hit_rate_std",
            "duration_mean", "duration_std",
            "overhead_pct_mean", "overhead_pct_std",
            "mean_iat_mean", "p95_iat_mean", "max_iat_mean",
        ]
        display_cols = [c for c in display_cols if c in summary.columns]
        print(summary[display_cols].to_string(index=False, float_format="%.3f"))
        print(f"\nResults written to: {results_dir}/")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
