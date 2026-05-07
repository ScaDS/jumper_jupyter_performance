import json
import os
import sys
import time
import zipfile
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from jumper_extension.adapters.data import aggregate_node_info
from jumper_extension.monitor.common import OfflinePerformanceMonitor
from jumper_extension.core.messages import (
    ExtensionInfoCode,
    EXTENSION_INFO_MESSAGES,
)


class SessionExporter:
    """Handles exporting a monitoring session to directory/ZIP."""

    def __init__(self, monitor, cell_history, visualizer, reporter, logger):
        self.monitor = monitor
        self.cell_history = cell_history
        self.visualizer = visualizer
        self.reporter = reporter
        self.logger = logger

    def export(self, path: Optional[str] = None) -> str:
        """Export session to a directory or, if path ends with .zip, to a zip archive.

        Returns the path to the exported directory or created archive.
        """
        export_dir, zip_target = self._determine_export_paths(path)
        os.makedirs(export_dir, exist_ok=True)

        schemas_perf = self._export_performance_data(export_dir)
        ch_df = self._export_cell_history(export_dir)
        manifest = self._build_manifest(schemas_perf, ch_df)
        self._write_manifest(export_dir, manifest)

        if zip_target:
            return self._create_zip_archive(export_dir, zip_target)

        self.logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXPORT_SUCCESS].format(
                filename=export_dir
            )
        )
        return export_dir

    def _determine_export_paths(self, path: Optional[str]) -> tuple:
        """Determine the export directory and optional ZIP target path.

        Returns:
            tuple: (export_dir, zip_target) where zip_target is None if not creating a ZIP
        """
        zip_target = None

        if path and path.lower().endswith(".zip"):
            export_dir = tempfile.mkdtemp(prefix="jumper-session-")
            zip_target = path
        else:
            export_dir = os.path.abspath(path or self._default_session_dirname())

        return export_dir, zip_target

    def _export_performance_data(self, export_dir: str) -> Dict[str, List[str]]:
        """Export performance data CSVs for each monitoring level.

        Args:
            export_dir: Directory to write CSV files to

        Returns:
            Dict mapping level names to their column schemas
        """
        schemas_perf: Dict[str, List[str]] = {}
        level_filenames = {
            "process": "perf_process.csv",
            "user": "perf_user.csv",
            "system": "perf_system.csv",
            "slurm": "perf_slurm.csv",
        }

        for level in self.monitor.nodes.levels:
            try:
                df_out = self.monitor.nodes.view(level=level, cell_history=self.cell_history)
            except Exception:
                df_out = pd.DataFrame()
            if not df_out.empty:
                schemas_perf[level] = list(df_out.columns)
                fname = level_filenames.get(level, f"perf_{level}.csv")
                df_out.to_csv(os.path.join(export_dir, fname), index=False)

        return schemas_perf

    def _export_cell_history(self, export_dir: str) -> pd.DataFrame:
        """Export cell history to CSV.

        Args:
            export_dir: Directory to write the cell history CSV to

        Returns:
            DataFrame containing the cell history
        """
        ch_df = self.cell_history.view()
        if not ch_df.empty:
            ch_df.to_csv(os.path.join(export_dir, "cell_history.csv"), index=False)
        return ch_df

    def _build_manifest(self, schemas_perf: Dict[str, List[str]], ch_df: pd.DataFrame) -> dict:
        """Build the manifest dictionary containing session metadata.

        Args:
            schemas_perf: Performance data schemas by level
            ch_df: Cell history DataFrame

        Returns:
            Manifest dictionary
        """
        hardware = aggregate_node_info(self.monitor.nodes.hardware)
        return {
            "version": "1.0",
            "app": {"name": "JUmPER", "version": self._app_version()},
            "monitor": {
                "interval": getattr(self.monitor, "interval", 1.0),
                "start_time": getattr(self.monitor, "start_time", None),
                "stop_time": getattr(self.monitor, "stop_time", None),
                "wallclock_start_time": getattr(self.monitor, "wallclock_start_time", None),
                "wallclock_stop_time": getattr(self.monitor, "wallclock_stop_time", None),
                "num_cpus": hardware.num_cpus,
                "num_system_cpus": hardware.num_system_cpus,
                "num_gpus": hardware.num_gpus,
                "gpu_memory": hardware.gpu_memory,
                "gpu_name": hardware.gpu_name,
                "memory_limits": hardware.memory_limits,
                "cpu_handles": hardware.cpu_handles,
                "pid": getattr(self.monitor, "pid", None),
                "uid": getattr(self.monitor, "uid", None),
                "slurm_job": getattr(self.monitor, "slurm_job", None),
                "os": os.name,
                "python": sys.version.split(" ")[0],
            },
            "levels": self.monitor.nodes.levels,
            "schemas": {
                "perf": schemas_perf,
                "cell_history": list(ch_df.columns),
            },
            "visualizer": {
                "default_metric_subsets": list(
                    getattr(self.visualizer, "default_subsets", ("cpu", "mem", "io"))
                ) + (["gpu", "gpu_all"] if hardware.num_gpus else []),
                "figsize": list(getattr(self.visualizer, "figsize", (5, 3))),
                "io_window": getattr(self.visualizer, "_io_window", None),
                "last_state": {},
            },
            "reporter": {
                "level": getattr(self.reporter, "level", "process") if hasattr(self.reporter, "level") else "process",
                "format": "text",
                "thresholds": getattr(self.reporter.printer.analyzer, "thresholds", {}),
            },
            "time_origin": "perf_counter",
            "timezone": time.tzname[0] if time.tzname else "",
        }

    def _write_manifest(self, export_dir: str, manifest: dict) -> None:
        """Write the manifest JSON file to the export directory.

        Args:
            export_dir: Directory to write the manifest to
            manifest: Manifest dictionary to serialize
        """
        with open(os.path.join(export_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _create_zip_archive(self, export_dir: str, zip_target: str) -> str:
        """Create a ZIP archive from the export directory and clean up.

        Args:
            export_dir: Directory containing exported files
            zip_target: Path to the ZIP file to create

        Returns:
            Path to the created ZIP file
        """
        with zipfile.ZipFile(zip_target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(export_dir):
                for name in files:
                    ap = os.path.join(root, name)
                    rel = os.path.relpath(ap, export_dir)
                    zf.write(ap, rel)

        self.logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXPORT_SUCCESS].format(
                filename=zip_target
            )
        )

        # Clean up temp dir
        try:
            shutil.rmtree(export_dir)
        except Exception:
            pass

        return zip_target

    def _default_session_dirname(self) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"jumper-session-{ts}"

    def _app_version(self) -> str:
        try:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pyproject = os.path.join(here, "..", "pyproject.toml")
            pyproject = os.path.normpath(pyproject)
            if os.path.exists(pyproject):
                with open(pyproject, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith("version") and "=" in s:
                            val = s.split("=", 1)[1].strip().strip('"')
                            if val:
                                return val
        except Exception:
            pass
        return "unknown"


class SessionImporter:
    """Handles importing a monitoring session from directory/ZIP."""

    def __init__(self, logger):
        self.logger = logger

    def import_(self, path: str, service) -> bool:
        """Import a session into the given service. Returns True on success."""
        if not path:
            return False

        work_dir, cleanup_dir = self._prepare_work_directory(path)

        try:
            manifest = self._load_manifest(work_dir)
            self._load_cell_history(work_dir, service)
            perf_dfs = self._load_performance_data(work_dir)

            self._setup_offline_monitor(manifest, perf_dfs, service, source=path)
            self._setup_reporter(manifest, service)
            self._apply_visualizer_settings(manifest, service)

            return True
        finally:
            if cleanup_dir and work_dir and os.path.isdir(work_dir):
                try:
                    shutil.rmtree(work_dir)
                except Exception:
                    pass

    def _prepare_work_directory(self, path: str) -> tuple:
        """Prepare the work directory from ZIP or direct path.

        Args:
            path: Path to ZIP file or directory

        Returns:
            tuple: (work_dir, cleanup_dir) where cleanup_dir indicates if temp dir should be cleaned
        """
        if path.lower().endswith(".zip"):
            work_dir = tempfile.mkdtemp(prefix="jumper-session-import-")
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(work_dir)
            return work_dir, True
        else:
            return path, False

    def _load_manifest(self, work_dir: str) -> dict:
        """Load the manifest JSON file from the work directory.

        Args:
            work_dir: Directory containing the manifest file

        Returns:
            Manifest dictionary
        """
        manifest_path = os.path.join(work_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_cell_history(self, work_dir: str, service) -> None:
        """Load cell history data from CSV into the service.

        Args:
            work_dir: Directory containing the cell history CSV
            service: Service object to load data into
        """
        ch_csv = os.path.join(work_dir, "cell_history.csv")
        if os.path.exists(ch_csv):
            try:
                service.cell_history.data = pd.read_csv(ch_csv)
            except Exception:
                pass

    def _load_performance_data(self, work_dir: str) -> Dict[str, pd.DataFrame]:
        """Load performance data CSVs from the work directory.

        Args:
            work_dir: Directory containing performance CSV files

        Returns:
            Dict mapping level names to their DataFrames
        """
        level_files = {
            "process": "perf_process.csv",
            "user": "perf_user.csv",
            "system": "perf_system.csv",
            "slurm": "perf_slurm.csv",
        }
        perf_dfs: Dict[str, pd.DataFrame] = {}

        for level, fname in level_files.items():
            fpath = os.path.join(work_dir, fname)
            if os.path.exists(fpath):
                try:
                    perf_dfs[level] = pd.read_csv(fpath)
                except Exception:
                    continue

        return perf_dfs

    def _setup_offline_monitor(self, manifest: dict, perf_dfs: Dict[str, pd.DataFrame], service, source: Optional[str]) -> None:
        """Create and attach an offline performance monitor to the service.

        Args:
            manifest: Manifest dictionary with monitor configuration
            perf_dfs: Performance data DataFrames by level
            service: Service object to attach monitor to
        """
        offline = OfflinePerformanceMonitor(
            manifest=manifest,
            perf_dfs=perf_dfs,
            source=source,
        )
        service.monitor = offline
        service.visualizer.attach(service.monitor)

    def _setup_reporter(self, manifest: dict, service) -> None:
        """Rebuild and attach the performance reporter with thresholds from manifest.

        Args:
            manifest: Manifest dictionary with reporter configuration
            service: Service object to attach reporter to
        """
        thresholds = None
        try:
            thresholds = manifest.get("reporter", {}).get("thresholds")
        except Exception:
            thresholds = None

        from jumper_extension.adapters.reporter import build_performance_reporter

        service.reporter = build_performance_reporter(
            service.cell_history,
            display_disabled=False,
            display_disabled_reason="Display not available.",
            thresholds=thresholds,
        )
        service.reporter.attach(service.monitor)

    def _apply_visualizer_settings(self, manifest: dict, service) -> None:
        """Apply visualizer settings from the manifest to the service.

        Args:
            manifest: Manifest dictionary with visualizer configuration
            service: Service object with visualizer to configure
        """
        try:
            viz = manifest.get("visualizer", {})
            if isinstance(viz.get("figsize"), list) and len(viz.get("figsize")) == 2:
                service.visualizer.figsize = (viz["figsize"][0], viz["figsize"][1])
            if viz.get("io_window"):
                try:
                    service.visualizer._io_window = int(viz.get("io_window"))
                except Exception:
                    pass
        except Exception:
            pass
