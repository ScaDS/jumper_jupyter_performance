"""Local monitoring collector that runs in a child process.

This module is executed as
``python -m jumper_extension.monitor.backends.subprocess_python._collector``
by :class:`SubprocessPerformanceMonitor`.  It instantiates the default
:class:`PerformanceMonitor`, collects metrics at the requested interval,
and writes one JSON object per sample to *stdout* (one line per object).

The parent process reads these lines to populate its
:class:`PerformanceData` container.

Protocol (stdout, one JSON line per message)::

    {"status": "ready", "pid": <int>, "columns_by_level": {…}, …}  # handshake
    {"level": "<level>", "wallclock": <float>, "sample": {…}}       # data
"""

import ctypes
import ctypes.util
import json
import os
import signal
import struct
import sys
import time
from typing import List, Optional

import psutil

_SCHED_BATCH = 3
_SCHED_OTHER = 0


def _set_sched_batch(pid: int) -> bool:
    try:
        _libc_name = ctypes.util.find_library("c")
        if not _libc_name:
            return False
        _libc = ctypes.CDLL(_libc_name, use_errno=True)
        param = struct.pack("i", 0)
        buf = ctypes.create_string_buffer(param)
        rc = _libc.sched_setscheduler(pid, _SCHED_BATCH, buf)
        return rc == 0
    except (OSError, AttributeError):
        return False


def _set_sched_other(pid: int) -> bool:
    try:
        _libc_name = ctypes.util.find_library("c")
        if not _libc_name:
            return False
        _libc = ctypes.CDLL(_libc_name, use_errno=True)
        param = struct.pack("i", 0)
        buf = ctypes.create_string_buffer(param)
        rc = _libc.sched_setscheduler(pid, _SCHED_OTHER, buf)
        return rc == 0
    except (OSError, AttributeError):
        return False


def _run_collector(
    interval: float,
    levels: Optional[List[str]] = None,
    target_pid: Optional[int] = None,
) -> None:
    _elevated = False
    try:
        os.nice(-10)
        _elevated = True
    except PermissionError:
        pass

    _renice_log = open("/tmp/jumper_renice.log", "a")
    _my_nice = os.getpriority(os.PRIO_PROCESS, 0)
    _renice_log.write(
        f"[{time.strftime('%H:%M:%S')}] collector start: "
        f"own nice={_my_nice} elevated={_elevated} "
        f"pid={os.getpid()} target_pid={target_pid}\n"
    )
    _renice_log.flush()

    _RENICE_VALUE = 19
    _reniced_pids: set = set()
    _my_pid = os.getpid()

    def _renice_target_pids(pids):
        for pid in pids:
            if pid == _my_pid or pid in _reniced_pids:
                continue
            try:
                old_nice = os.getpriority(os.PRIO_PROCESS, pid)
                os.setpriority(os.PRIO_PROCESS, pid, _RENICE_VALUE)
                new_nice = os.getpriority(os.PRIO_PROCESS, pid)
                batch_ok = _set_sched_batch(pid)
                _reniced_pids.add(pid)
                _renice_log.write(
                    f"[{time.strftime('%H:%M:%S')}] reniced pid={pid} "
                    f"nice {old_nice}->{new_nice} "
                    f"sched_batch={'ok' if batch_ok else 'FAIL'}\n"
                )
                _renice_log.flush()
            except (PermissionError, ProcessLookupError, OSError) as exc:
                _renice_log.write(
                    f"[{time.strftime('%H:%M:%S')}] renice pid={pid} FAILED: {exc}\n"
                )
                _renice_log.flush()

    def _restore_target_pids():
        for pid in list(_reniced_pids):
            try:
                os.setpriority(os.PRIO_PROCESS, pid, 0)
                _set_sched_other(pid)
            except (PermissionError, ProcessLookupError, OSError):
                pass
        _reniced_pids.clear()

    import io
    import contextlib
    import logging

    log_capture = io.StringIO()
    log_handler = logging.StreamHandler(log_capture)
    log_handler.setLevel(logging.WARNING)
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.WARNING)

    temp_stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(temp_stdout):
            from jumper_extension.monitor.backends.thread import PerformanceMonitor
            monitor = PerformanceMonitor()
    except Exception as e:
        root_logger.removeHandler(log_handler)
        root_logger.setLevel(original_level)
        error_msg = {"status": "error", "pid": os.getpid(), "error": str(e)}
        sys.stderr.write(f"[SubprocessCollector] init error: {e}\n")
        sys.stderr.flush()
        sys.stdout.write(json.dumps(error_msg) + "\n")
        sys.stdout.flush()
        return

    root_logger.removeHandler(log_handler)
    root_logger.setLevel(original_level)

    if target_pid is not None:
        monitor.pid = target_pid
        monitor.process = psutil.Process(target_pid)

    for label, buf in [("init logs", log_capture), ("init stdout", temp_stdout)]:
        text = buf.getvalue()
        if text:
            sys.stderr.write(f"[SubprocessCollector {label}] {text}")
            sys.stderr.flush()

    if levels is None:
        levels = monitor.levels

    # Derive per-level column schema from a bootstrap collect call.
    # PerformanceMonitor.__init__() already bootstrapped the IO state,
    # so this call produces real columns without resetting anything.
    bootstrap_rows = monitor._collect_metrics()
    columns_by_level = {
        level: list(row.keys())
        for level, row in zip(monitor.levels, bootstrap_rows)
    }

    hardware = monitor.nodes.hardware["local"]
    ready_msg = {
        "status": "ready",
        "pid": os.getpid(),
        "num_cpus": hardware.num_cpus,
        "num_system_cpus": hardware.num_system_cpus,
        "num_gpus": hardware.num_gpus,
        "gpu_memory": hardware.gpu_memory,
        "gpu_name": hardware.gpu_name,
        "memory_limits": hardware.memory_limits,
        "cpu_handles": hardware.cpu_handles,
        "levels": levels,
        "columns_by_level": columns_by_level,
    }
    sys.stdout.write(json.dumps(ready_msg) + "\n")
    sys.stdout.flush()

    running = True

    def _shutdown(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    monitor.interval = interval
    monitor.start_time = time.perf_counter()
    monitor.wallclock_start_time = time.time()
    monitor.running = True

    next_tick = time.perf_counter()
    _tick_count = 0
    try:
        while running:
            _t0 = time.perf_counter()
            try:
                process_pids = monitor._process_backend.get_process_pids()
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                next_tick += interval
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
                continue
            _t1 = time.perf_counter()

            _renice_target_pids(process_pids)
            _t2 = time.perf_counter()

            try:
                rows = monitor._collect_metrics()
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                next_tick += interval
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
                continue
            _t3 = time.perf_counter()

            for level, row in zip(monitor.levels, rows):
                if level not in levels:
                    continue
                msg = {
                    "level": level,
                    "wallclock": time.time(),
                    "sample": row,
                }
                sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()
            _t4 = time.perf_counter()

            _tick_count += 1
            if _tick_count <= 5 or _tick_count % 10 == 0:
                _renice_log.write(
                    f"[{time.strftime('%H:%M:%S')}] tick={_tick_count} "
                    f"npids={len(process_pids)} "
                    f"get_pids={_t1-_t0:.3f}s "
                    f"renice={_t2-_t1:.3f}s "
                    f"collect={_t3-_t2:.3f}s "
                    f"emit={_t4-_t3:.3f}s "
                    f"total={_t4-_t0:.3f}s\n"
                )
                _renice_log.flush()

            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()

    except BrokenPipeError:
        sys.stderr.write("[SubprocessCollector] Broken pipe — parent exited\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[SubprocessCollector] Error in main loop: {e}\n")
        sys.stderr.flush()
    finally:
        _restore_target_pids()
        _renice_log.write(f"[{time.strftime('%H:%M:%S')}] collector stop\n")
        _renice_log.close()
        monitor.running = False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="JUmPER local subprocess monitoring collector"
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--target-pid", type=int, default=None)
    parser.add_argument("--levels", type=str, default=None)
    args = parser.parse_args()
    levels = args.levels.split(",") if args.levels else None
    _run_collector(args.interval, levels, target_pid=args.target_pid)


if __name__ == "__main__":
    main()
