#!/usr/bin/env python3
"""Minimal multi-node smoke test for JUmPER without Jupyter.

Multi-node counterpart of ``run_demo_simple.py``.  Only requires Python and
``pip install .`` of this repository (no PyTorch, CUDA or modules).  Uses
the ``slurm_multinode`` monitor backend.

Each allocated node runs, one after another: a short sleep, a CPU burn on
all its CPUs and a 500 MB allocation.  A final cell runs the CPU burn on
all nodes at the same time.  Every phase is a separate "cell".

Usage (via SLURM):
    sbatch run_demo_multinode_simple.sbatch

Usage (direct, inside a SLURM allocation):
    python run_demo_multinode_simple.py
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

from jumper_extension.core.service import build_perfmonitor_service
from jumper_extension.utilities import is_slurm_available

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("noJUmP-multinode-simple")

PHASES = {
    "sleep": "import time; time.sleep(5)",
    "cpu_burn": (
        "import os; from run_demo_simple import cpu_burn_parallel; "
        "cpu_burn_parallel(10, len(os.sched_getaffinity(0)))"
    ),
    "memory": (
        "from run_demo_simple import allocate_memory; "
        "allocate_memory(500, 5)"
    ),
}


def run_phase(script_dir: str, phase: str, hostname: str = None,
              num_nodes: int = 1) -> None:
    """Run a workload phase via srun on one node or on ``num_nodes`` nodes."""
    cmd = [
        "srun",
        "--overlap",
        f"--nodes={num_nodes}",
        f"--ntasks={num_nodes}",
        "--ntasks-per-node=1",
        "--unbuffered",
        f"--chdir={script_dir}",
    ]
    if hostname:
        cmd.append(f"--nodelist={hostname}")
    cmd += [sys.executable, "-c", PHASES[phase]]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.warning(
            f"srun for phase '{phase}' exited with code {result.returncode}"
        )


def main() -> int:
    logger.info("=== noJUmP simple multi-node demo ===")

    if not is_slurm_available():
        logger.error("This demo requires a SLURM allocation. Run via sbatch.")
        return 1

    base_dir = os.path.dirname(os.path.abspath(__file__))
    export_level = "slurm"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_multinode_simple_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    jpeg_path = os.path.join(output_dir, f"perf_plot_{export_level}.jpeg")
    pickle_path = os.path.join(output_dir, f"perf_data_{export_level}.pkl")
    csv_path = os.path.join(output_dir, f"perf_data_{export_level}.csv")
    session_path = os.path.join(output_dir, "session")

    service = build_perfmonitor_service()
    service.start_monitoring(interval=1.0, monitor_type="slurm_multinode")
    service.show_resources()

    from jumper_extension.monitor.backends.slurm_multinode._node_discovery import get_slurm_nodes
    nodes = get_slurm_nodes()
    logger.info(f"Allocated nodes: {nodes}")

    phase_labels = [
        ("sleep", "idle sleep 5 s"),
        ("cpu_burn", "CPU burn 10 s (all CPUs)"),
        ("memory", "allocate 500 MB"),
    ]

    for node_idx, hostname in enumerate(nodes):
        for phase, label in phase_labels:
            with service.monitored(f"Node {node_idx} ({hostname}): {label}"):
                logger.info(f"{label} on {hostname}")
                run_phase(base_dir, phase, hostname=hostname)

    with service.monitored("All nodes: CPU burn 10 s (all CPUs)"):
        logger.info(f"CPU burn on all {len(nodes)} nodes in parallel")
        run_phase(base_dir, "cpu_burn", num_nodes=len(nodes))

    logger.info("Exporting performance data...")

    cells = service.cell_history.view()
    cell_range = (int(cells["cell_index"].min()), int(cells["cell_index"].max()))
    service.plot_performance(
        level=export_level, save_jpeg=jpeg_path, cell_range=cell_range
    )
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

    logger.info("=== Simple multi-node demo complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
