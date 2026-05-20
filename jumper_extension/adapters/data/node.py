import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from jumper_extension.adapters.data.data import PerformanceData
from jumper_extension.utilities import get_available_levels


@dataclass
class NodeInfo:
    node: str
    num_cpus: int
    num_system_cpus: int
    num_gpus: int
    gpu_memory: float
    gpu_name: str
    memory_limits: Dict[str, float]
    cpu_handles: List[int] = field(default_factory=list)


class NodeDataStore:
    """Single source of truth for per-node hardware metadata and time-series data.

    ``register_node(info)`` stores both the ``NodeInfo`` and the corresponding
    ``PerformanceData`` container under the same node key.

    Access patterns
    ---------------
    ``store.hardware``                     – Dict[str, NodeInfo] (metadata)
    ``store.view(level)``                  – aggregate DataFrame across all nodes
    ``store.view(level, node=n)``          – single-node DataFrame
    ``store.add_sample(node, level, row)`` – append one flat-dict sample
    """

    def __init__(self) -> None:
        self._info: Dict[str, NodeInfo] = {}
        self._nodes: Dict[str, PerformanceData] = {}

    # ------------------------------------------------------------------ #
    # Registration                                                        #
    # ------------------------------------------------------------------ #

    def register_node(self, info: NodeInfo) -> None:
        self._info[info.node] = info
        self._nodes[info.node] = PerformanceData()

    @property
    def hardware(self) -> Dict[str, NodeInfo]:
        return self._info

    def node_names(self) -> List[str]:
        return list(self._info.keys())

    @property
    def levels(self) -> List[str]:
        if not self._nodes:
            return get_available_levels()
        return list(next(iter(self._nodes.values())).levels)

    # ------------------------------------------------------------------ #
    # Writing                                                             #
    # ------------------------------------------------------------------ #

    def add_sample(self, node: str, level: str, row: dict) -> None:
        perf_data = self._nodes.get(node)
        if perf_data is None:
            return
        perf_data.add_sample(level, row)

    def init_node_schema(
        self, node: str, columns_by_level: Dict[str, List[str]]
    ) -> None:
        """Store per-level column lists so view() can return a correctly shaped
        empty DataFrame before the first sample arrives."""
        perf_data = self._nodes.get(node)
        if perf_data is None:
            return
        perf_data._schema_columns = dict(columns_by_level)

    def load_frames(self, node: str, frames: Dict[str, pd.DataFrame]) -> None:
        """Inject pre-loaded DataFrames into a registered node's data container.

        Used by offline (imported) monitors to populate data without going
        through the live add_sample() path.
        """
        perf_data = self._nodes.get(node)
        if perf_data is None:
            return
        for level, df in frames.items():
            perf_data._rows[level] = df.to_dict("records")

    # ------------------------------------------------------------------ #
    # Reading                                                             #
    # ------------------------------------------------------------------ #

    def view(
        self,
        level: str = "process",
        node: Optional[str] = None,
        slice_=None,
        cell_history=None,
    ) -> pd.DataFrame:
        if not self._nodes:
            return pd.DataFrame()

        if node is not None:
            perf_data = self._nodes.get(node)
            if perf_data is None:
                return pd.DataFrame()
            return perf_data.view(level=level, slice_=slice_, cell_history=cell_history)

        if len(self._nodes) == 1:
            return next(iter(self._nodes.values())).view(
                level=level, slice_=slice_, cell_history=cell_history
            )

        return self._aggregate(level, cell_history)

    def _aggregate(self, level: str, cell_history=None) -> pd.DataFrame:
        node_dfs: Dict[str, pd.DataFrame] = {}
        for n, perf in self._nodes.items():
            df = perf.view(level)
            if not df.empty:
                node_dfs[n] = df

        if not node_dfs:
            return pd.DataFrame()
        if len(node_dfs) == 1:
            df = next(iter(node_dfs.values()))
            return self._attach_cell_index(df, cell_history) if cell_history else df

        min_len = min(len(df) for df in node_dfs.values())
        if min_len == 0:
            return pd.DataFrame()

        frames = [
            df.iloc[:min_len].reset_index(drop=True)
            for df in node_dfs.values()
        ]
        result = frames[0].copy()

        self._aggregate_memory(frames, result)
        self._aggregate_io(frames, result)
        self._aggregate_cpu(frames, result)
        self._aggregate_gpu(frames, result)

        if cell_history is not None:
            result = self._attach_cell_index(result, cell_history)
        return result

    def _aggregate_memory(
        self, frames: List[pd.DataFrame], result: pd.DataFrame
    ) -> None:
        if all("memory" in f.columns for f in frames):
            result["memory"] = sum(f["memory"] for f in frames)

    def _aggregate_io(
        self, frames: List[pd.DataFrame], result: pd.DataFrame
    ) -> None:
        for col in ("io_read", "io_write", "io_read_count", "io_write_count"):
            if all(col in f.columns for f in frames):
                result[col] = sum(f[col] for f in frames)

    def _aggregate_cpu(
        self, frames: List[pd.DataFrame], result: pd.DataFrame
    ) -> None:
        if all("cpu_util_avg" in f.columns for f in frames):
            result["cpu_util_avg"] = sum(f["cpu_util_avg"] for f in frames) / len(frames)
        if all("cpu_util_min" in f.columns for f in frames):
            result["cpu_util_min"] = pd.concat(
                [f["cpu_util_min"] for f in frames], axis=1
            ).min(axis=1)
        if all("cpu_util_max" in f.columns for f in frames):
            result["cpu_util_max"] = pd.concat(
                [f["cpu_util_max"] for f in frames], axis=1
            ).max(axis=1)
        drop = [c for c in result.columns if re.match(r"cpu_util_\d+$", c)]
        result.drop(columns=drop, errors="ignore", inplace=True)

    def _aggregate_gpu(
        self, frames: List[pd.DataFrame], result: pd.DataFrame
    ) -> None:
        for metric in ("util", "band", "mem"):
            avg_col = f"gpu_{metric}_avg"
            vals = [f[avg_col] for f in frames if avg_col in f.columns]
            if vals:
                result[avg_col] = sum(vals) / len(vals)

            min_col = f"gpu_{metric}_min"
            vals = [f[min_col] for f in frames if min_col in f.columns]
            if vals:
                result[min_col] = pd.concat(vals, axis=1).min(axis=1)

            max_col = f"gpu_{metric}_max"
            vals = [f[max_col] for f in frames if max_col in f.columns]
            if vals:
                result[max_col] = pd.concat(vals, axis=1).max(axis=1)

        drop = [c for c in result.columns if re.match(r"gpu_(util|band|mem)_\d+$", c)]
        result.drop(columns=drop, errors="ignore", inplace=True)

    def _attach_cell_index(self, df: pd.DataFrame, cell_history) -> pd.DataFrame:
        result = df.copy()
        result["cell_index"] = pd.NA
        times = result["time"].to_numpy()
        for row in cell_history.data.itertuples(index=False):
            mask = (times >= row.start_time) & (times <= row.end_time)
            result.loc[mask, "cell_index"] = row.cell_index
        return result

    # ------------------------------------------------------------------ #
    # Export / load (delegate to primary node)                           #
    # ------------------------------------------------------------------ #

    def export(
        self,
        filename: str = "performance_data.csv",
        level: str = "process",
        cell_history=None,
    ) -> None:
        if not self._nodes:
            return
        df = self.view(level=level, cell_history=cell_history)
        if df.empty:
            return
        first = next(iter(self._nodes.values()))
        _, ext = os.path.splitext(filename)
        format = ext.lower().lstrip(".") or "csv"
        if not format:
            format = "csv"
            filename += ".csv"
        writer = first._file_writers.get(format)
        if writer:
            writer(filename, df)

    def load(self, filename: str) -> Optional[pd.DataFrame]:
        if not self._nodes:
            return None
        return next(iter(self._nodes.values())).load(filename)


def aggregate_node_info(hardware: Dict[str, NodeInfo]) -> NodeInfo:
    """Return a synthetic NodeInfo aggregating all nodes in *hardware*.

    CPUs/GPUs are summed, gpu_memory takes the max, memory_limits are
    summed per level, gpu_name is taken from the first node that has one.
    Used by reporter/service/session to get a single summary view.
    """
    nodes = list(hardware.values())
    if not nodes:
        return NodeInfo(
            node="aggregate", num_cpus=0, num_system_cpus=0, num_gpus=0,
            gpu_memory=0.0, gpu_name="", memory_limits={},
        )
    all_levels = {lvl for n in nodes for lvl in n.memory_limits}
    return NodeInfo(
        node="aggregate",
        num_cpus=sum(n.num_cpus for n in nodes),
        num_system_cpus=sum(n.num_system_cpus for n in nodes),
        num_gpus=sum(n.num_gpus for n in nodes),
        gpu_memory=max(n.gpu_memory for n in nodes),
        gpu_name=next((n.gpu_name for n in nodes if n.gpu_name), ""),
        memory_limits={
            lvl: sum(n.memory_limits.get(lvl, 0.0) for n in nodes)
            for lvl in all_levels
        },
        cpu_handles=[h for n in nodes for h in n.cpu_handles],
    )
