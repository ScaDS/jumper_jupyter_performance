"""Runtime sanity check for performance monitor backends.

The check starts an unstarted monitor for a few seconds while a
background thread generates I/O activity, then validates that the
collected data contains the expected columns, is not all-NaN, and not
all-zero.

.. important::

    This check was tailored for the ``thread``, ``subprocess_python``
    and ``native_c`` backends.  Other monitors (e.g. ``slurm_multinode``
    or user-provided implementations) are unlikely to populate all the
    metrics/levels the check expects and are therefore expected to
    fail.  Callers should warn the user beforehand via
    :func:`is_supported_monitor`.
"""

import logging
import os
import tempfile
import threading
import time

import pandas as pd

logger = logging.getLogger("extension")

_SANITY_N_SAMPLES = 3

_REQUIRED_METRICS = [
    "time", "cpu_util_avg", "cpu_util_min", "cpu_util_max",
    "memory",
    "io_read", "io_write", "io_read_count", "io_write_count",
]
# Columns whose "all zeros" result is acceptable at specific levels.
_ZERO_CHECK_SKIP_LEVELS = {
    "cpu_util_min": {"system"},
    "io_read": {"process", "user", "slurm"},
}
_GPU_METRICS = [
    "gpu_util_avg", "gpu_util_min", "gpu_util_max",
    "gpu_band_avg", "gpu_band_min", "gpu_band_max",
    "gpu_mem_avg", "gpu_mem_min", "gpu_mem_max",
]


def is_supported_monitor(monitor) -> bool:
    """Return ``True`` if *monitor* is a backend the sanity check supports.

    Only ``thread`` (``PerformanceMonitor``), ``subprocess_python``
    (``SubprocessPerformanceMonitor``) and ``native_c``
    (``CSubprocessPerformanceMonitor``, a subclass of the subprocess
    monitor) are supported.
    """
    try:
        from jumper_extension.monitor.backends.thread import PerformanceMonitor
        from jumper_extension.monitor.backends.subprocess_python import (
            SubprocessPerformanceMonitor,
        )
    except ImportError:
        return False
    return isinstance(monitor, (PerformanceMonitor, SubprocessPerformanceMonitor))


def run_sanity_check(monitor, interval: float = 1.0, timeout: float = 10.0) -> bool:
    """Run a short sanity check on an *unstarted* monitor.

    Collects ~3 samples at ``interval`` seconds while a background IO
    worker keeps I/O counters non-zero, then validates the collected
    data.  Returns ``True`` on success, ``False`` otherwise.  Progress
    is printed to stdout so the user sees it from a notebook cell.
    """
    required = _SANITY_N_SAMPLES + 1  # +1: first sample's CPU delta is 0

    io_stop = threading.Event()
    tmp_path_holder = [None]

    def _io_worker():
        f = tempfile.NamedTemporaryFile(delete=False, prefix="jumper_sanity_")
        tmp_path_holder[0] = f.name
        buf = b"x" * (64 * 1024)
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

    print(f"[JUmPER] sanity check: collecting {_SANITY_N_SAMPLES} samples "
          f"at {1.0 / interval:.1f} Hz...")

    try:
        monitor.start(interval=interval)
    except Exception as exc:
        io_stop.set()
        io_thread.join(timeout=2)
        _cleanup_tmp(tmp_path_holder[0])
        print(f"[JUmPER] sanity check FAILED: monitor did not start: {exc}")
        return False

    deadline = time.monotonic() + timeout
    enough = False
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if monitor.data is not None:
            levels = getattr(monitor, "levels", ["process"])
            counts = [
                len(monitor.data.data.get(lv, pd.DataFrame())) for lv in levels
            ]
            if counts and min(counts) >= required:
                enough = True
                break

    monitor.stop()
    io_stop.set()
    io_thread.join(timeout=2)
    _cleanup_tmp(tmp_path_holder[0])

    if monitor.data is None or not enough:
        print(f"[JUmPER] sanity check FAILED: not enough samples collected "
              f"(needed {required}).")
        return False

    ok = True
    for level in getattr(monitor, "levels", ["process"]):
        df = monitor.data.data.get(level, pd.DataFrame())
        if df.empty or len(df) < 2:
            print(f"[JUmPER] sanity check [{level}]: "
                  f"only {len(df)} sample(s), need >=2.")
            ok = False
            continue

        df_check = df.iloc[1:]  # skip first tick (CPU delta is 0)

        for col in _REQUIRED_METRICS:
            if col not in df.columns:
                print(f"[JUmPER] sanity check [{level}]: missing '{col}'.")
                ok = False
                continue
            if df[col].isna().any():
                print(f"[JUmPER] sanity check [{level}]: "
                      f"NaN values in '{col}'.")
                ok = False
            if col != "time" and len(df_check) > 0:
                skip = _ZERO_CHECK_SKIP_LEVELS.get(col, set())
                if level not in skip and (df_check[col] == 0).all():
                    print(f"[JUmPER] sanity check [{level}]: "
                          f"'{col}' is all zeros.")
                    ok = False

        cpu_cols = [
            c for c in df.columns
            if c.startswith("cpu_util_")
            and c not in ("cpu_util_avg", "cpu_util_min", "cpu_util_max")
        ]
        if not cpu_cols:
            print(f"[JUmPER] sanity check [{level}]: "
                  f"no per-core CPU columns found.")
            ok = False

        if getattr(monitor, "num_gpus", 0) > 0:
            for col in _GPU_METRICS:
                if col not in df.columns:
                    print(f"[JUmPER] sanity check [{level}]: "
                          f"GPU column '{col}' missing.")
                    ok = False
                elif df[col].isna().any():
                    print(f"[JUmPER] sanity check [{level}]: "
                          f"NaN values in GPU column '{col}'.")
                    ok = False

    if ok:
        print("[JUmPER] sanity check PASSED.")
    else:
        print("[JUmPER] sanity check FAILED.")
    return ok


def _cleanup_tmp(path):
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass
