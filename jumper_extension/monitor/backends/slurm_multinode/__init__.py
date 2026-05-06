"""SLURM multi-node performance monitoring package.

Provides a monitor that discovers SLURM job nodes, connects via SSH,
runs per-node performance agents, and aggregates results into a log file.
"""

from jumper_extension.monitor.backends.slurm_multinode.monitor import SlurmMultinodeMonitor

__all__ = ["SlurmMultinodeMonitor"]
