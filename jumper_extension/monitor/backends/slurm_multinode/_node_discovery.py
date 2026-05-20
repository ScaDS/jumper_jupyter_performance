"""Discover SLURM job nodes from environment variables."""

import os
import re
import subprocess
import logging
from typing import List

logger = logging.getLogger("extension")


def expand_nodelist(nodelist: str) -> List[str]:
    """Expand a SLURM compact nodelist into individual hostnames.

    Uses ``scontrol show hostnames`` when available, falls back to a
    simple bracket-expansion parser otherwise.

    Args:
        nodelist: Compact nodelist string, e.g. ``"node[01-03,05]"``
            or ``"node01,node02"``.

    Returns:
        List of individual node hostnames.
    """
    # Try scontrol first – it handles all SLURM nodelist formats
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: simple bracket expansion
    return _expand_brackets(nodelist)


def _expand_brackets(nodelist: str) -> List[str]:
    """Expand ``prefix[01-03,05]`` style nodelists without scontrol."""
    nodes: List[str] = []
    # Split on comma that is NOT inside brackets
    parts = re.split(r",(?![^\[]*\])", nodelist)
    for part in parts:
        m = re.match(r"^(.+?)\[(.+)\]$", part)
        if not m:
            nodes.append(part.strip())
            continue
        prefix = m.group(1)
        for spec in m.group(2).split(","):
            if "-" in spec:
                lo, hi = spec.split("-", 1)
                width = len(lo)
                for i in range(int(lo), int(hi) + 1):
                    nodes.append(f"{prefix}{str(i).zfill(width)}")
            else:
                nodes.append(f"{prefix}{spec}")
    return nodes


def get_slurm_nodes() -> List[str]:
    """Return the list of nodes allocated to the current SLURM job.

    Reads ``SLURM_JOB_NODELIST`` (or ``SLURM_NODELIST``) from the
    environment.

    Raises:
        RuntimeError: If no SLURM nodelist environment variable is set.
    """
    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get(
        "SLURM_NODELIST", ""
    )
    if not nodelist:
        raise RuntimeError(
            "SLURM_JOB_NODELIST / SLURM_NODELIST is not set. "
            "Are you running inside a SLURM job?"
        )
    nodes = expand_nodelist(nodelist)
    logger.info(f"[JUmPER]: Discovered SLURM nodes: {nodes}")
    return nodes
