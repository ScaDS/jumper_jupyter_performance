"""Disk-backed performance visualizer (kept as a backup).

This backend was used historically to replay sessions whose performance
data and monitor metadata had been persisted under
``perfdata_results/<pid>/``.  It is not wired into
``build_performance_visualizer`` because there is no current consumer,
but it is preserved here so it can be revived later without going back
through the legacy ``adapters/visualizer.py`` module.

Inherits from :class:`MatplotlibPerformanceVisualizer` so that BALI
rendering, cell boundaries and the interactive panel UI keep working.
"""

import logging

import pandas as pd

from jumper_extension.adapters.visualizer.backends.matplotlib import (
    MatplotlibPerformanceVisualizer,
)
from jumper_extension.bali_adapter import BaliVisualizationMixin
from jumper_extension.utilities import (
    get_available_levels,
    load_monitor_metadata_from_disk,
    load_perfdata_from_disk,
)

logger = logging.getLogger("extension")


class DiskPerformanceVisualizer(MatplotlibPerformanceVisualizer):
    """Performance visualizer that loads data from disk instead of memory."""

    def __init__(self, pid, cell_history, bali_adapter=None):
        self.pid = pid
        self.cell_history = cell_history
        self.figsize = (5, 3)
        self.min_duration = 1.0
        self._io_window = 1

        # Load monitor metadata from disk
        metadata = load_monitor_metadata_from_disk(pid)
        if metadata is None:
            logger.warning(
                f"No monitor metadata found for PID {pid}, using defaults"
            )
            metadata = {
                "num_cpus": 8,
                "num_system_cpus": 8,
                "num_gpus": 1,
                "gpu_memory": 30.0,
                "start_time": 0,
                "memory_limits": {
                    level: 100.0 for level in get_available_levels()
                },
            }

        # Create monitor object with loaded metadata
        class MockMonitor:
            def __init__(self, pid, metadata):
                self.pid = pid
                self.num_cpus = metadata["num_cpus"]
                self.num_system_cpus = metadata["num_system_cpus"]
                self.num_gpus = metadata["num_gpus"]
                self.gpu_memory = metadata["gpu_memory"]
                self.start_time = metadata["start_time"]
                self.memory_limits = metadata["memory_limits"]

        self.monitor = MockMonitor(pid, metadata)

        # Initialize BALI functionality
        BaliVisualizationMixin.__init__(self, bali_adapter=bali_adapter)

        # Load perfdata from disk
        self.perfdata_by_level = load_perfdata_from_disk(
            pid, get_available_levels()
        )

        # Initialize subsets (copy from parent class)
        self.subsets = {
            "cpu_all": {
                "cpu": {
                    "type": "multi_series",
                    "prefix": "cpu_util_",
                    "title": "CPU Utilization (%) - Across Cores",
                    "ylim": (0, 100),
                    "label": "CPU Utilization (All Cores)",
                }
            },
            "gpu_all": {
                "gpu_util": {
                    "type": "multi_series",
                    "prefix": "gpu_util_",
                    "title": "GPU Utilization (%) - Across GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Utilization (All GPUs)",
                },
                "gpu_band": {
                    "type": "multi_series",
                    "prefix": "gpu_band_",
                    "title": "GPU Bandwidth Usage (%) - Across GPUs",
                    "ylim": (0, 100),
                    "label": "GPU Bandwidth (All GPUs)",
                },
                "gpu_mem": {
                    "type": "multi_series",
                    "prefix": "gpu_mem_",
                    "title": "GPU Memory Usage (GB) - Across GPUs",
                    "ylim": (0, self.monitor.gpu_memory),
                    "label": "GPU Memory (All GPUs)",
                },
                "gpu_power": {
                    "type": "multi_series",
                    "prefix": "gpu_power_",
                    "title": "GPU Power Usage (W) - Across GPUs",
                    "ylim": None,
                    "label": "GPU Power (All GPUs)",
                },
            },
            "cpu": {
                "cpu_summary": {
                    "type": "summary_series",
                    "columns": [
                        "cpu_util_min",
                        "cpu_util_avg",
                        "cpu_util_max",
                    ],
                    "title": (
                        f"CPU Utilization (%) - {self.monitor.num_cpus} CPUs"
                    ),
                    "ylim": (0, 100),
                    "label": "CPU Utilization Summary",
                }
            },
            "gpu": {
                "gpu_util_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_util_min",
                        "gpu_util_avg",
                        "gpu_util_max",
                    ],
                    "title": (
                        f"GPU Utilization (%) - {self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": (0, 100),
                    "label": "GPU Utilization Summary",
                },
                "gpu_band_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_band_min",
                        "gpu_band_avg",
                        "gpu_band_max",
                    ],
                    "title": (
                        f"GPU Bandwidth Usage (%) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": (0, 100),
                    "label": "GPU Bandwidth Summary",
                },
                "gpu_mem_summary": {
                    "type": "summary_series",
                    "columns": ["gpu_mem_min", "gpu_mem_avg", "gpu_mem_max"],
                    "title": (
                        f"GPU Memory Usage (GB) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": (0, self.monitor.gpu_memory),
                    "label": "GPU Memory Summary",
                },
                "gpu_power_summary": {
                    "type": "summary_series",
                    "columns": [
                        "gpu_power_min",
                        "gpu_power_avg",
                        "gpu_power_max",
                    ],
                    "title": (
                        f"GPU Power Usage (W) - "
                        f"{self.monitor.num_gpus} GPUs"
                    ),
                    "ylim": None,
                    "label": "GPU Power Summary",
                },
            },
            "mem": {
                "memory": {
                    "type": "single_series",
                    "column": "memory",
                    "title": "Memory Usage (GB)",
                    "ylim": None,
                    "label": "Memory Usage",
                }
            },
            "io": {
                "io_read": {
                    "type": "single_series",
                    "column": "io_read",
                    "title": "I/O Read (MB/s)",
                    "ylim": None,
                    "label": "IO Read MB/s",
                },
                "io_write": {
                    "type": "single_series",
                    "column": "io_write",
                    "title": "I/O Write (MB/s)",
                    "ylim": None,
                    "label": "IO Write MB/s",
                },
                "io_read_count": {
                    "type": "single_series",
                    "column": "io_read_count",
                    "title": "I/O Read Operations (ops/s)",
                    "ylim": None,
                    "label": "IO Read Ops",
                },
                "io_write_count": {
                    "type": "single_series",
                    "column": "io_write_count",
                    "title": "I/O Write Operations (ops/s)",
                    "ylim": None,
                    "label": "IO Write Ops",
                },
            },
        }

    def plot(
        self,
        metric_subsets=("cpu", "mem", "io"),
        cell_range=None,
        show_idle=False,
        show_bali=True,
    ):
        """Plot performance metrics using disk data.

        BALI segments are shown by default for the disk-replay path
        because the typical reason for ``%perfmonitor_plot --from-disk``
        is to inspect previously computed BALI segments. The default
        ``cell_range`` covers the *full* loaded history rather than the
        last "long" cell, so BALI segments that ran in earlier cells are
        included on the initial render.
        """
        if any(not df.empty for df in self.perfdata_by_level.values()):
            if cell_range is None:
                try:
                    valid_cells = self.cell_history.view()
                    if len(valid_cells) > 0:
                        cell_range = (
                            int(valid_cells["cell_index"].min()),
                            int(valid_cells["cell_index"].max()),
                        )
                except Exception:  # noqa: BLE001 - defensive default
                    cell_range = None
            # Override the plot method to use disk data
            self._plot_with_disk_data(
                metric_subsets, cell_range, show_idle, show_bali
            )
        else:
            logger.warning("No performance data found on disk")

    def _plot_with_disk_data(
        self, metric_subsets, cell_range, show_idle, show_bali=True
    ):
        """Modified plot method that uses pre-loaded disk data."""
        # Use the parent class plot method but override data access

        class MockData:
            def view(self, level):
                return self.perfdata_by_level.get(level, pd.DataFrame())

        mock_data = MockData()
        mock_data.perfdata_by_level = self.perfdata_by_level
        self.monitor.data = mock_data

        # Call parent plot method
        super().plot(
            metric_subsets, cell_range, show_idle, show_bali=show_bali
        )
