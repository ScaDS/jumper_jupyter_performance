"""Remote monitoring collector that runs on each SLURM node.

Protocol (stdout, one JSON line per message)::

    {"status": "ready", "node": "<hostname>", "columns_by_level": {…}, …}
    {"level": "<level>", "node": "<hostname>", "wallclock": <float>, "sample": {…}}
"""

import json
import os
import signal
import socket
import sys
import time
from typing import List, Optional

from jumper_extension.monitor.backends.thread import PerformanceMonitor


def _run_collector(interval: float, levels: Optional[List[str]] = None) -> None:
    hostname = socket.gethostname()

    try:
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
        with contextlib.redirect_stdout(temp_stdout):
            monitor = PerformanceMonitor()

        root_logger.removeHandler(log_handler)
        root_logger.setLevel(original_level)

        log_output = log_capture.getvalue()
        if log_output:
            sys.stderr.write(f"[Collector init logs] {log_output}")
            sys.stderr.flush()

        stdout_output = temp_stdout.getvalue()
        if stdout_output:
            sys.stderr.write(f"[Collector init stdout] {stdout_output}")
            sys.stderr.flush()

        if levels is None:
            levels = monitor.levels

        # Bootstrap column schema
        bootstrap_rows = monitor._collect_metrics()
        columns_by_level = {
            level: list(row.keys())
            for level, row in zip(monitor.levels, bootstrap_rows)
        }

        hardware = monitor.nodes.hardware["local"]
        ready_msg = {
            "status": "ready",
            "node": hostname,
            "num_cpus": hardware.num_cpus,
            "num_system_cpus": hardware.num_system_cpus,
            "num_gpus": hardware.num_gpus,
            "gpu_memory": hardware.gpu_memory,
            "gpu_name": hardware.gpu_name,
            "levels": levels,
            "pid": os.getpid(),
            "columns_by_level": columns_by_level,
        }
        sys.stdout.write(json.dumps(ready_msg) + "\n")
        sys.stdout.flush()

    except Exception as e:
        error_msg = {
            "status": "error",
            "node": hostname,
            "error": str(e),
            "pid": os.getpid(),
        }
        sys.stderr.write(f"[Collector error] {e}\n")
        sys.stderr.flush()
        sys.stdout.write(json.dumps(error_msg) + "\n")
        sys.stdout.flush()
        return

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

    sys.stderr.write(f"[Collector {hostname}] Starting main loop with interval {interval}s\n")
    sys.stderr.flush()

    try:
        while running:
            t0 = time.perf_counter()
            rows = monitor._collect_metrics()

            for level, row in zip(monitor.levels, rows):
                if level not in levels:
                    continue
                msg = {
                    "node": hostname,
                    "level": level,
                    "wallclock": time.time(),
                    "sample": row,
                }
                sys.stdout.write(json.dumps(msg) + "\n")
                sys.stderr.write(f"[Collector {hostname}] Sent sample for level {level}\n")
                sys.stderr.flush()
            sys.stdout.flush()

            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
    except BrokenPipeError:
        sys.stderr.write(f"[Collector {hostname}] Broken pipe - exiting\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[Collector {hostname}] Error in main loop: {e}\n")
        sys.stderr.flush()
    finally:
        monitor.running = False
        sys.stderr.write(f"[Collector {hostname}] Exiting main loop\n")
        sys.stderr.flush()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="JUmPER remote node monitoring collector"
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--levels", type=str, default=None)
    args = parser.parse_args()
    levels = args.levels.split(",") if args.levels else None
    _run_collector(args.interval, levels)


if __name__ == "__main__":
    main()
