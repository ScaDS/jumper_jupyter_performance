import os
import logging
from typing import Optional, Dict, Callable, List

import json
import pandas as pd
import psutil

from jumper_extension.core.messages import (
    ExtensionErrorCode,
    EXTENSION_ERROR_MESSAGES,
)

logger = logging.getLogger("extension")


def filter_perfdata(cell_history_data, perfdata, compress_idle=True):
    """Filter performance data to remove idle periods if requested"""
    if cell_history_data is None or cell_history_data.empty:
        return perfdata.iloc[0:0]

    if compress_idle:
        # Remove idle periods between cells
        # Create time masks for each cell's execution period
        masks = []
        for _, cell in cell_history_data.iterrows():
            mask = (perfdata["time"] >= cell["start_time"]) & (
                perfdata["time"] <= cell["end_time"]
            )
            masks.append(mask)

        if masks:
            combined_mask = pd.concat(masks, axis=1).any(axis=1)
            # Use .values to avoid IndexingError when perfdata has
            # a non-unique or misaligned index (e.g. multi-level data
            # at high sampling frequencies).
            return perfdata.loc[combined_mask.values]
        else:
            return perfdata.iloc[0:0]
    else:
        """Get start time from first cell and end time from last cell in the
        range"""
        start_time = cell_history_data.iloc[0]["start_time"]
        end_time = cell_history_data.iloc[-1]["end_time"]
        return perfdata[
            (perfdata["time"] >= start_time) & (perfdata["time"] <= end_time)
        ]


def is_slurm_available():
    """Check if SLURM is available by checking for SLURM_JOB_ID environment
    variable"""
    return os.environ.get("SLURM_JOB_ID") is not None


def get_available_levels():
    """Get list of available performance monitoring levels"""
    base_levels = ["user", "process", "system"]
    if is_slurm_available():
        base_levels.append("slurm")
    return base_levels


def detect_cgroup_version():
    """Detect if system is using cgroup v1 or v2"""
    return (
        "v2" if os.path.exists("/sys/fs/cgroup/cgroup.controllers") else "v1"
    )


def detect_memory_limit(level, uid, slurm_job):
    """Detect memory limit for a given level"""
    system_mem = round(psutil.virtual_memory().total / (1024**3), 2)

    if level == "slurm":
        paths = (
            [
                f"/sys/fs/cgroup/memory/slurm/uid_{uid}/job_{slurm_job}/"
                "memory.limit_in_bytes"
            ]
            if detect_cgroup_version() == "v1"
            else [
                f"/sys/fs/cgroup/system.slice/slurmstepd.scope/"
                f"job_{slurm_job}/memory.max",
                f"/sys/fs/cgroup/system.slice/slurm.service/job_{slurm_job}/"
                "memory.max",
                f"/sys/fs/cgroup/slurm/uid_{uid}/job_{slurm_job}/memory.max",
            ]
        )

        for path in paths:
            if os.path.exists(path):
                with open(path) as f:
                    limit = f.read().strip()
                    if limit != "max":
                        return round(int(limit) / (1024**3), 2)
    elif level == "process":
        try:
            import resource

            rlimit = resource.getrlimit(resource.RLIMIT_AS)[0]
            if rlimit != resource.RLIM_INFINITY:
                return round(rlimit / (1024**3), 2)
        except Exception:
            pass

    return system_mem


def load_dataframe_from_file(
    filename: str,
    readers: Dict[str, Callable],
    required_columns: List[str],
    entity_name: str = "data",
) -> Optional[pd.DataFrame]:
    """Load a DataFrame from CSV or JSON file with validation.

    Args:
        filename: Path to the file to load
        readers: Dict mapping format (e.g., 'csv', 'json') to reader functions
        required_columns: List of column names that must be present
        entity_name: Human-readable name for logging (e.g., 'performance data')

    Returns:
        DataFrame if successful, None otherwise
    """
    if not filename:
        return None

    _, ext = os.path.splitext(filename)
    file_format = ext.lower().lstrip(".")

    try:
        reader = readers.get(file_format)
        if reader is None:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.UNSUPPORTED_FORMAT
                ].format(
                    format=file_format or "",
                    supported_formats=", ".join(readers.keys()),
                )
            )
            return None
        df = reader(filename)
    except Exception as e:
        logger.warning(f"[JUmPER]: Failed to load {entity_name}: {e}")
        return None

    # Validate required columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.warning(
            f"[JUmPER]: Cannot load {entity_name}. "
            f"Missing required columns: {', '.join(missing)}"
        )
        return None

    return df

def save_perfdata_to_disk(pid, data):
    """Save performance data to disk by PID and level"""
    perfdata_dir = f"perfdata_results/{pid}"
    os.makedirs(perfdata_dir, exist_ok=True)

    for level in data.levels:
        df = data.view(level=level)
        if not df.empty:
            filepath = os.path.join(perfdata_dir, f"perfdata_{level}.csv")
            df.to_csv(filepath, index=False)


def save_cell_history_to_disk(pid, cell_history):
    """Save cell history to disk by PID"""
    perfdata_dir = f"perfdata_results/{pid}"
    os.makedirs(perfdata_dir, exist_ok=True)

    filepath = os.path.join(perfdata_dir, "cell_history.json")
    with open(filepath, "w") as f:
        json.dump(cell_history.data.to_dict("records"), f, indent=2)


def load_perfdata_from_disk(pid, levels):
    """Load performance data from disk by PID"""
    perfdata_dir = f"perfdata_results/{pid}"
    perfdata_by_level = {}

    for level in levels:
        filepath = os.path.join(perfdata_dir, f"perfdata_{level}.csv")
        if os.path.exists(filepath):
            perfdata_by_level[level] = pd.read_csv(filepath)
        else:
            perfdata_by_level[level] = pd.DataFrame()

    return perfdata_by_level


def load_cell_history_from_disk(pid):
    """Load cell history from disk by PID"""
    filepath = f"perfdata_results/{pid}/cell_history.json"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return pd.DataFrame()


def save_monitor_metadata_to_disk(pid, monitor):
    """Save monitor metadata to disk"""
    perfdata_dir = f"perfdata_results/{pid}"
    os.makedirs(perfdata_dir, exist_ok=True)

    metadata = {
        "num_cpus": monitor.num_cpus,
        "num_system_cpus": monitor.num_system_cpus,
        "num_gpus": monitor.num_gpus,
        "gpu_memory": monitor.gpu_memory,
        "start_time": monitor.start_time,
        "memory_limits": monitor.memory_limits,
    }

    filepath = os.path.join(perfdata_dir, "monitor_metadata.json")
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)


def load_monitor_metadata_from_disk(pid):
    """Load monitor metadata from disk"""
    filepath = f"perfdata_results/{pid}/monitor_metadata.json"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None