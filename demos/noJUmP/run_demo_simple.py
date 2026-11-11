#!/usr/bin/env python3
"""Minimal "hello world" smoke test for JUmPER without Jupyter.

Only requires Python and ``pip install .`` of this repository.  No PyTorch,
CUDA, environment modules or large memory/disk usage are needed.  The
workload consists of short sleeps, a little CPU burn and a small memory
allocation, each wrapped in ``service.monitored()`` so it appears as a
separate "cell".

Usage (direct):
    python run_demo_simple.py

Usage (SLURM):
    sbatch run_demo_simple.sbatch
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import time

os.environ.setdefault("MPLBACKEND", "Agg")

from jumper_extension.core.service import build_perfmonitor_service
from jumper_extension.utilities import is_slurm_available

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("noJUmP-simple")


def cpu_burn(seconds: float) -> int:
    """Busy-loop for ``seconds`` seconds and return the iteration count."""
    end = time.time() + seconds
    count = 0
    while time.time() < end:
        count += sum(i * i for i in range(1000)) % 7
    return count


def cpu_burn_parallel(seconds: float, processes: int) -> None:
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(cpu_burn, [seconds] * processes)
    print(f"CPU burn on {processes} process(es) done, checksum={sum(results)}")


def allocate_memory(mb: int, hold_seconds: float) -> None:
    print(f"Allocating {mb} MB...")
    buf = bytearray(mb * 1024 * 1024)
    for i in range(0, len(buf), 4096):
        buf[i] = 1
    time.sleep(hold_seconds)
    del buf
    print("Memory released")


def main() -> int:
    logger.info("=== noJUmP simple demo ===")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    export_level = "slurm" if is_slurm_available() else "user"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_simple_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    jpeg_path = os.path.join(output_dir, f"perf_plot_{export_level}.jpeg")
    pickle_path = os.path.join(output_dir, f"perf_data_{export_level}.pkl")
    csv_path = os.path.join(output_dir, f"perf_data_{export_level}.csv")
    session_path = os.path.join(output_dir, "session")

    if hasattr(os, "sched_getaffinity"):
        n_cpus = len(os.sched_getaffinity(0))
    else:
        n_cpus = multiprocessing.cpu_count()

    service = build_perfmonitor_service()
    service.start_monitoring(interval=1.0)
    service.show_resources()

    with service.monitored("Cell 0: idle sleep"):
        logger.info("Cell 0: sleeping 5 s")
        time.sleep(5)

    with service.monitored("Cell 1: CPU burn (1 process)"):
        logger.info("Cell 1: CPU burn on 1 process for 10 s")
        cpu_burn_parallel(10, 1)

    with service.monitored("Cell 2: CPU burn (all CPUs)"):
        logger.info(f"Cell 2: CPU burn on {n_cpus} processes for 10 s")
        cpu_burn_parallel(10, n_cpus)

    with service.monitored("Cell 3: allocate 500 MB"):
        logger.info("Cell 3: allocate 500 MB and hold for 5 s")
        allocate_memory(500, 5)

    with service.monitored("Cell 4: idle sleep"):
        logger.info("Cell 4: sleeping 5 s")
        time.sleep(5)

    logger.info("Exporting performance data...")

    service.plot_performance(level=export_level, save_jpeg=jpeg_path)
    logger.info(f"JPEG plot ({export_level}) saved to: {jpeg_path}")

    perf_frames = service.export_perfdata(level=export_level)
    if isinstance(perf_frames, dict):
        serializable = {
            k: v.to_dict(orient="records") if hasattr(v, "to_dict") else v
            for k, v in perf_frames.items()
        }
    elif hasattr(perf_frames, "to_dict"):
        serializable = perf_frames.to_dict(orient="records")
    else:
        serializable = perf_frames
    with open(pickle_path, "wb") as f:
        pickle.dump(serializable, f)
    logger.info(f"Pickle data ({export_level}) saved to: {pickle_path}")

    service.export_perfdata(file=csv_path, level=export_level)
    logger.info(f"CSV perfdata ({export_level}) saved to: {csv_path}")

    service.export_session(path=session_path)
    logger.info(f"Session exported to: {session_path}")

    service.stop_monitoring()
    service.close()

    logger.info("=== Simple demo complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
