"""Remote monitoring collector that runs on each SLURM node.

This module is executed as ``python -m jumper_extension.monitor.backends.slurm_multinode._collector``
on each remote node via SSH.  It instantiates the default
:class:`PerformanceMonitor`, collects metrics at the requested interval,
and writes one JSON object per sample to *stdout* (one line per object).

The orchestrator on the head node reads these lines to aggregate results.

Protocol (stdout, one JSON line per sample)::

    {"node": "<hostname>", "time": <float>, "level": "<level>", "sample": {…}}

A special ``{"status": "ready", "node": "<hostname>", ...}`` line is
emitted once the monitor is initialised so the orchestrator knows the
collector is alive.

The collector stops gracefully when *stdin* is closed or when it receives
a SIGTERM / SIGINT.
"""

import json
import os
import signal
import socket
import sys
import time
from typing import List, Optional

# Make sure the package is importable even when invoked stand-alone on
# the remote node.  The orchestrator ensures the correct PYTHONPATH.
from jumper_extension.monitor.backends.thread import PerformanceMonitor


def _run_collector(interval: float, levels: Optional[List[str]] = None) -> None:
    hostname = socket.gethostname()
    
    try:
        # Temporarily redirect stdout and logging to prevent log messages from breaking JSON protocol
        import io
        import contextlib
        import logging
        
        # Create a custom logger handler to capture log messages
        log_capture = io.StringIO()
        log_handler = logging.StreamHandler(log_capture)
        log_handler.setLevel(logging.WARNING)
        
        # Get the root logger and add our capture handler
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.WARNING)
        
        # Capture stdout as well
        temp_stdout = io.StringIO()
        with contextlib.redirect_stdout(temp_stdout):
            monitor = PerformanceMonitor()
        
        # Remove our temporary handler and restore logging
        root_logger.removeHandler(log_handler)
        root_logger.setLevel(original_level)
        
        # Check if any log messages were captured and write them to stderr
        log_output = log_capture.getvalue()
        if log_output:
            sys.stderr.write(f"[Collector init logs] {log_output}")
            sys.stderr.flush()
        
        # Check if any stdout was captured and write it to stderr
        stdout_output = temp_stdout.getvalue()
        if stdout_output:
            sys.stderr.write(f"[Collector init stdout] {stdout_output}")
            sys.stderr.flush()

        if levels is None:
            levels = monitor.levels

        # Emit "ready" handshake
        ready_msg = {
            "status": "ready",
            "node": hostname,
            "num_cpus": monitor.num_cpus,
            "num_system_cpus": monitor.num_system_cpus,
            "num_gpus": monitor.num_gpus,
            "gpu_memory": monitor.gpu_memory,
            "gpu_name": monitor.gpu_name,
            "levels": levels,
            "pid": os.getpid(),
        }
        sys.stdout.write(json.dumps(ready_msg) + "\n")
        sys.stdout.flush()
        
    except Exception as e:
        # If anything fails during initialization, send error message
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

    # Debug: Let us know we're starting the main loop
    sys.stderr.write(f"[Collector {hostname}] Starting main loop with interval {interval}s\n")
    sys.stderr.flush()

    try:
        while running:
            t0 = time.perf_counter()
            monitor.process_pids = monitor._get_process_pids()
            metrics = monitor._collect_metrics()

            for level, data_tuple in zip(monitor.levels, metrics):
                if level not in levels:
                    continue
                (
                    time_mark,
                    cpu_util,
                    memory,
                    gpu_util,
                    gpu_band,
                    gpu_mem,
                    io_counters,
                ) = data_tuple

                sample = {
                    "node": hostname,
                    "time": time_mark,
                    "wallclock": time.time(),
                    "level": level,
                    "sample": {
                        "cpu_util": cpu_util,
                        "memory": memory,
                        "gpu_util": gpu_util,
                        "gpu_band": gpu_band,
                        "gpu_mem": gpu_mem,
                        "io_counters": io_counters,
                    },
                }
                sys.stdout.write(json.dumps(sample) + "\n")
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
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated list of levels to monitor (default: all available)",
    )
    args = parser.parse_args()
    levels = args.levels.split(",") if args.levels else None
    _run_collector(args.interval, levels)


if __name__ == "__main__":
    main()
