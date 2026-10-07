#!/usr/bin/env python3
"""Multi-node performance monitoring demo without Jupyter.

This script replicates the workload from ``run_demo.py`` but runs across
multiple SLURM-allocated nodes.  It uses the ``slurm_multinode`` monitor
backend, which launches a collector on every allocated node via ``srun``
and aggregates their samples.

The workload is executed **sequentially** on each node: first on node 1,
then on node 2, etc.  Each node's workload phases are tracked as separate
"cells" in the performance data, labelled with the node hostname.

After running, performance data is exported in the same formats as the
single-node demo.

Usage (via SLURM):
    sbatch run_demo_multinode.sbatch

Usage (direct, inside a SLURM allocation):
    python run_demo_multinode.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from random import random

from jumper_extension.core.service import build_perfmonitor_service
from jumper_extension.utilities import is_slurm_available

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("noJUmP-multinode")


# ---------------------------------------------------------------------------
# Workload functions (shared with single-node demo)
# ---------------------------------------------------------------------------

def throw_dart(iterations: int) -> int:
    hits = 0
    for i in range(iterations):
        x = random()
        y = random()
        if (x * x) + (y * y) <= 1:
            hits += 1
    return hits


def compute_pi(iterations: int, process_count: int) -> None:
    from multiprocessing import Pool
    pool = Pool(processes=process_count)
    trials_per_process = [int(iterations / process_count)] * process_count
    hits = pool.map(throw_dart, trials_per_process)
    pi = (sum(hits) * 4) / iterations
    print(f"Pi approximation: {pi}")


def fill_memory_2_5gb() -> list:
    """Fill 2.5 GB of main memory."""
    import gc
    import numpy as np
    print("Filling 2.5GB of main memory...")
    gb_size = 2.5 * 1024 * 1024 * 1024
    num_elements = int(gb_size / 8)
    print(f"Allocating {num_elements} float64 elements (~2.50 GB)")
    arrays = []
    try:
        for i in range(10):
            arr = np.ones(num_elements // 10, dtype=np.float64)
            arrays.append(arr)
            print(f"Created array {i + 1}")
        gc.collect()
        print("2.5GB memory allocation completed")
        return arrays
    except MemoryError:
        print("Memory allocation failed!")
        return []


def write_and_read_1gb_forced() -> None:
    """Write 1 GB to disk, wait, then read it back."""
    filename = "large_data_forced.bin"
    data_size = 1024 * 1024 * 1024
    chunk_size = 1024 * 1024
    target_rate_mb_per_sec = 100
    target_time_per_chunk = 1.0 / target_rate_mb_per_sec

    print("Starting 1GB write operation...")
    start_time = time.time()
    written_bytes = 0
    chunk_count = 0

    with open(filename, "wb") as f:
        f.flush()
        while written_bytes < data_size:
            chunk = os.urandom(chunk_size)
            f.write(chunk)
            written_bytes += len(chunk)
            chunk_count += 1
            f.flush()
            os.fsync(f.fileno())
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                current_speed = written_bytes / (1024 * 1024) / elapsed_time
                print(
                    f"Writing: {written_bytes / (1024**3):.2f} GB "
                    f"({chunk_count} chunks), Speed: {current_speed:.2f} MB/s",
                    end="\r",
                )
            time.sleep(target_time_per_chunk)

    print(f"\nWritten {os.path.getsize(filename) / (1024**3):.2f} GB to disk")
    print("Forcing OS to flush all buffers...")
    os.sync()
    print("Waiting 10 seconds...")
    time.sleep(10)

    print("Reading 1GB data back (forced cache bypass)...")
    start_time = time.time()
    read_bytes = 0
    read_chunk_count = 0

    try:
        with open(filename, "rb") as f:
            while read_bytes < data_size:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                read_bytes += len(chunk)
                read_chunk_count += 1
                try:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except Exception:
                    pass
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    current_speed = read_bytes / (1024 * 1024) / elapsed_time
                    print(
                        f"Reading: {read_bytes / (1024**3):.2f} GB "
                        f"({read_chunk_count} chunks), Speed: {current_speed:.2f} MB/s",
                        end="\r",
                    )
                time.sleep(target_time_per_chunk)
    except Exception as e:
        print(f"\nError during read: {e}")
        with open(filename, "rb") as f:
            while read_bytes < data_size:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                read_bytes += len(chunk)
                read_chunk_count += 1
                time.sleep(target_time_per_chunk)

    print(f"\nRead {read_bytes / (1024**3):.2f} GB from disk")
    os.sync()
    os.remove(filename)
    print("1GB forced write/read operation completed")


def cuda_memory_operations() -> None:
    """Fill 2 GB of CUDA memory, wait, then free it."""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed, skipping CUDA operations")
        return

    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA memory operations")
        return

    print("CUDA available, performing memory operations...")
    device = torch.device("cuda")
    gb_size = 2 * 1024 * 1024 * 1024
    num_elements = int(gb_size / 4)
    print(f"Allocating 2GB of CUDA memory ({num_elements} elements)")
    tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
    print(f"Allocated tensor size: {tensor.numel() * 4 / (1024**3):.2f} GB")
    print("Waiting 10 seconds...")
    time.sleep(10)
    del tensor
    torch.cuda.empty_cache()
    print("CUDA memory freed")


def cuda_heavy_computation() -> None:
    """Perform heavy computation on CUDA device for 20 seconds."""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed, skipping CUDA heavy computation")
        return

    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA heavy computation")
        return

    print("CUDA available, performing heavy computation...")
    device = torch.device("cuda")
    size = 10000
    iterations = 0
    print("Starting heavy computation on CUDA for 20 seconds...")
    start_time = time.time()
    while time.time() - start_time < 20:
        a = torch.randn(size, size, dtype=torch.float32, device=device)
        b = torch.randn(size, size, dtype=torch.float32, device=device)
        c = torch.matmul(a, b)
        d = torch.sin(c) + torch.cos(c)
        e = torch.exp(d) * torch.log(d + 1e-8)
        iterations += 1
        if iterations % 10 == 0:
            torch.cuda.empty_cache()
    print(f"Heavy computation completed in 20 seconds with {iterations} iterations")


# ---------------------------------------------------------------------------
# Remote workload execution
# ---------------------------------------------------------------------------

def _srun_on_node(hostname: str, script_dir: str, python_code: str) -> None:
    """Run a Python snippet on a specific node via srun."""
    cmd = [
        "srun",
        "--overlap",
        "--nodes=1",
        f"--nodelist={hostname}",
        "--ntasks=1",
        "--unbuffered",
        "bash", "-c",
        f"cd {script_dir} && {sys.executable} -c \"{python_code}\"",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.warning(f"srun on {hostname} exited with code {result.returncode}")


def run_phase_on_node(hostname: str, script_dir: str, phase: str) -> None:
    """Execute a single workload phase on a specific node."""
    phases = {
        "compute_pi": (
            f"from run_demo_multinode import compute_pi; "
            f"import multiprocessing; "
            f"compute_pi(10**8, multiprocessing.cpu_count())"
        ),
        "fill_memory": (
            f"from run_demo_multinode import fill_memory_2_5gb; "
            f"import time; "
            f"fill_memory_2_5gb(); time.sleep(5)"
        ),
        "io": (
            f"from run_demo_multinode import write_and_read_1gb_forced; "
            f"write_and_read_1gb_forced()"
        ),
        "cuda_mem": (
            f"from run_demo_multinode import cuda_memory_operations; "
            f"cuda_memory_operations()"
        ),
        "cuda_compute": (
            f"from run_demo_multinode import cuda_heavy_computation; "
            f"cuda_heavy_computation()"
        ),
    }
    code = phases[phase]
    logger.info(f"  Phase '{phase}' on {hostname}")
    _srun_on_node(hostname, script_dir, code)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("=== noJUmP multi-node demo ===")

    if not is_slurm_available():
        logger.error("This demo requires a SLURM allocation. Run via sbatch.")
        return 1

    base_dir = os.path.dirname(os.path.abspath(__file__))
    export_level = "slurm"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_multinode_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    jpeg_path = os.path.join(output_dir, f"perf_plot_{export_level}.jpeg")
    pickle_path = os.path.join(output_dir, f"perf_data_{export_level}.pkl")
    csv_path = os.path.join(output_dir, f"perf_data_{export_level}.csv")
    session_path = os.path.join(output_dir, "session")

    service = build_perfmonitor_service()
    service.start_monitoring(interval=2.0, monitor_type="slurm_multinode")
    service.show_resources()

    # Discover allocated nodes
    from jumper_extension.monitor.backends.slurm_multinode._node_discovery import get_slurm_nodes
    nodes = get_slurm_nodes()
    logger.info(f"Allocated nodes: {nodes}")

    # Run the workload sequentially on each node
    phase_labels = [
        ("compute_pi",    "compute_pi with all CPUs"),
        ("fill_memory",   "fill 2.5 GB memory"),
        ("io",            "write/read 1 GB"),
        ("cuda_mem",      "CUDA memory operations"),
        ("cuda_compute",  "CUDA heavy computation"),
    ]

    cell_idx = 0
    for node_idx, hostname in enumerate(nodes):
        for phase, label in phase_labels:
            with service.monitored(f"Node {node_idx} ({hostname}): {label}"):
                logger.info(f"Cell {cell_idx}: {label} on {hostname}")
                run_phase_on_node(hostname, base_dir, phase)
                cell_idx += 1

    # --- Export ---
    logger.info("Exporting performance data...")

    service.plot_performance(
        level=export_level,
        save_jpeg=jpeg_path,
    )
    logger.info(f"JPEG plot ({export_level}) saved to: {jpeg_path}")

    perf_frames = service.export_perfdata(level=export_level)
    import pickle as _pickle
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
        _pickle.dump(serializable, f)
    logger.info(f"Pickle data ({export_level}) saved to: {pickle_path}")

    service.export_perfdata(
        file=csv_path,
        level=export_level,
    )
    logger.info(f"CSV perfdata ({export_level}) saved to: {csv_path}")

    service.export_session(path=session_path)
    logger.info(f"Session exported to: {session_path}")

    service.stop_monitoring()
    service.close()

    logger.info("=== Multi-node demo complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
