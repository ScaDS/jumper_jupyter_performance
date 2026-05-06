"""Configuration and state models for the JUmPER core.

This module defines dataclasses that hold runtime configuration for
monitoring, performance reports, and names of exported or loaded
variables.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExportVars:
    """Names of variables used when exporting data frames.

    Attributes:
        perfdata: Variable name for exported performance data.
        cell_history: Variable name for exported cell history.
    """

    perfdata: str = "perfdata_df"
    cell_history: str = "cell_history_df"


@dataclass
class LoadedVars:
    """Names of variables used when loading data frames.

    Attributes:
        perfdata: Variable name for loaded performance data.
        cell_history: Variable name for loaded cell history.
    """

    perfdata: str = "loaded_perfdata_df"
    cell_history: str = "loaded_cell_history_df"


@dataclass
class PerfomanceReports:
    """Configuration for automatic per-cell performance reports.

    Attributes:
        enabled: Whether per-cell reports are enabled.
        level: Monitoring level used when generating reports.
        text: If True, use text reports instead of HTML.
    """

    enabled: bool = False
    level: str = "process"
    text: bool = False


@dataclass
class PerformanceMonitoring:
    """Configuration for the performance monitoring loop.

    Attributes:
        default_interval: Default sampling interval in seconds.
        user_interval: User-provided interval overriding the default.
        running: Whether monitoring is currently running.
    """

    default_interval: float = 1.0
    user_interval: Optional[float] = None
    running: bool = False


@dataclass
class Settings:
    """Top-level configuration container for the extension.

    Groups performance reports, monitoring configuration, and variable
    names used when exporting or loading data.

    Attributes:
        perfreports: Settings for per-cell performance reports.
        monitoring: Settings for the monitoring loop.
        export_vars: Names for exported data variables.
        loaded_vars: Names for loaded data variables.
        visualizer_backend: Default backend used for plotting.
    """

    perfreports: PerfomanceReports = field(default_factory=PerfomanceReports)
    monitoring: PerformanceMonitoring = field(default_factory=PerformanceMonitoring)
    export_vars: ExportVars = field(default_factory=ExportVars)
    loaded_vars: LoadedVars = field(default_factory=LoadedVars)
    visualizer_backend: str = "matplotlib"

    def snapshot(self) -> "Settings":
        """Return a deep copy of the current settings.

        Returns:
            Settings: Independent copy of the current configuration.
        """
        return copy.deepcopy(self)
