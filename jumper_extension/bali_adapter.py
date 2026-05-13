import pandas as pd
from typing import List, Dict, Tuple, Any
import numpy as np
from itables import show
import logging
from jumper_extension.bali_hook import BaliResultsParser
from jumper_extension.core.messages import EXTENSION_INFO_MESSAGES, ExtensionInfoCode

logger = logging.getLogger("extension")


class BaliAdapter:
    """
    Adapter class that provides a clean interface for BALI functionality.
    """

    def __init__(self):
        self.parser = BaliResultsParser()
        self._segments_df = pd.DataFrame(
            columns=[
                "model",
                "framework",
                "batch_size",
                "input_len",
                "output_len",
                "num_samples",
                "iteration",
                "start_time",
                "end_time",
                "duration",
                "duration_text_gen",
                "start_text_gen",
                "tokens_per_sec",
                "is_error",
                "error_message",
            ]
        )

    def get_segments_dataframe(self) -> pd.DataFrame:
        """Get the current BALI segments as a DataFrame."""
        return self._segments_df.copy()

    def refresh_segments_from_disk(self, pid: int) -> int:
        """
        Refresh BALI segments from disk for the given process ID.
        """
        segments = self.parser.collect_all_bali_segments(pid)

        # Build DataFrame directly from segments and align to canonical column order
        df = pd.DataFrame(segments)
        self._segments_df = df.reindex(columns=self._segments_df.columns)
        
        return len(self._segments_df)

    def get_segments_for_visualization(self, pid: int) -> List[Dict]:
        """
        Get BALI segments in the format needed for visualization.

        Always re-reads from disk so new BALI runs (additional result
        directories created after the first one was cached) are picked up.
        """
        self.refresh_segments_from_disk(pid)

        df = self._segments_df
        df = df[df["start_time"].notna() & df["end_time"].notna()]
        return df.to_dict(orient="records")

    def get_tokens_per_sec_range(
            self, segments: List[Dict]
    ) -> Tuple[float, float]:
        """Get the min/max tokens per second range for coloring."""
        return self.parser.get_tokens_per_sec_range(segments)
    
    def get_energy_efficiency_range(
           self, segments: List[Dict]
    ) -> Tuple[float, float]: 
        """Get the min/max tokens per second range for coloring."""
        return self.parser.get_energy_efficiency_range(segments)

    def get_color_for_tokens_per_sec(
            self, tokens_per_sec: float, vmin: float, vmax: float
    ) -> Tuple[float, float, float, float]:
        """Get color for a given tokens per second value."""
        return self.parser.get_color_for_tokens_per_sec(
            tokens_per_sec, vmin, vmax
        )
    def get_color_for_energy_efficiency(self, tokens_per_sec: float, vmin: float, vmax: float
    ) -> Tuple[float, float, float, float]:
        return self.parser.get_color_for_energy_efficiency(
            tokens_per_sec, vmin, vmax
        )

    def get_colormap(self):
        """Get the colormap used for visualization."""
        return self.parser.colormap
    
    def get_energy_colormap(self):
        return self.parser.colormap_energy

    def add_llm_performance_info(self, segment: Dict, perfdata) -> Dict:
        """Compute energy / tokens-per-joule for a segment.

        ``segment["start_time"]`` / ``segment["end_time"]`` are already in
        the same compressed-time coordinate system as ``perfdata["time"]``
        (i.e. the x-axis used by the plots), so no further normalization is
        required here.
        """
        seg_start = segment.get("start_time")
        seg_end = segment.get("end_time")
        seg_text_start = segment.get("start_text_gen")
        total_tokens = segment.get("total_tokens") or 0

        def _trapz(values):
            if values.empty or "gpu_power_avg" not in values.columns:
                return 0.0
            times = np.asarray(values["time"], dtype=float)
            powers = np.asarray(values["gpu_power_avg"], dtype=float)
            if len(times) < 2:
                return 0.0
            return float(np.trapz(powers, times))

        def _safe_div(a, b):
            return a / b if b else None

        full_values = perfdata[
            (perfdata["time"] >= seg_start) & (perfdata["time"] <= seg_end)
        ]
        if seg_text_start is not None:
            text_values = perfdata[
                (perfdata["time"] >= seg_text_start)
                & (perfdata["time"] <= seg_end)
            ]
        else:
            text_values = full_values.iloc[0:0]

        total_energy = _trapz(full_values)
        text_gen_energy = _trapz(text_values)

        return {
            "total_energy": total_energy,
            "text_gen_energy": text_gen_energy,
            "energy_per_token_full_segment": _safe_div(total_energy, total_tokens),
            "token_per_joule_full_segment": _safe_div(total_tokens, total_energy),
            "energy_per_token_text_gen": _safe_div(text_gen_energy, total_tokens),
            "token_per_joule_text_gen": _safe_div(total_tokens, text_gen_energy),
        }


    def compress_segments(
            self,
            segments: List[Dict],
            cell_range: Tuple[int, int],
            perfdata: Any,
            cell_history: Any,
            compressed_cell_boundaries: List[Dict] = None,
            current_time_offset: float = 0,
    ) -> List[Dict]:
        """Place BALI segments on the compressed plot x-axis.

        The compressed time axis starts at 0 and stitches together the
        rendered cells (no idle gaps). For each cell in the visible range we
        find BALI segments that overlap that cell using ``perf_counter``
        time, then position them at
        ``compressed_cell_start + (segment_perf_start - cell_perf_start)``.

        ``compressed_cell_boundaries`` (optional) is the list of rendered
        cells with their compressed ``start_time``. When omitted, cells are
        stitched in order using their raw ``duration`` from ``cell_history``.
        """
        if not segments:
            return []

        start_idx, end_idx = cell_range
        cell_data = cell_history.view(start_idx, end_idx + 1)

        # Map cell_index -> compressed start_time on the plot axis.
        compressed_by_index = {}
        if compressed_cell_boundaries:
            for cb in compressed_cell_boundaries:
                compressed_by_index[int(cb["cell_index"])] = float(
                    cb["start_time"]
                )
        else:
            running = float(current_time_offset)
            for _, cell in cell_data.iterrows():
                compressed_by_index[int(cell["cell_index"])] = running
                try:
                    running += float(cell["duration"])
                except Exception:
                    pass

        compressed = []
        for _, cell in cell_data.iterrows():
            cell_idx = int(cell["cell_index"])
            if cell_idx not in compressed_by_index:
                # Cell was filtered out (no perfdata), so it isn't drawn.
                continue
            compressed_cell_start = compressed_by_index[cell_idx]
            try:
                cell_perf_start = float(cell["start_time"])
                cell_perf_end = float(cell["end_time"])
            except Exception:
                continue

            for seg in segments:
                # BALI runs inside the kernel process, so its ``start_time``
                # and ``end_time`` come from the same ``time.perf_counter()``
                # clock as ``cell_history``.
                seg_perf_start = seg.get("start_time")
                seg_dur = seg.get("duration")
                if seg_perf_start is None or seg_dur is None:
                    continue
                seg_perf_end = seg_perf_start + seg_dur

                # perf_counter overlap test
                if (
                    seg_perf_end <= cell_perf_start
                    or seg_perf_start >= cell_perf_end
                ):
                    continue

                dt_in_cell = seg_perf_start - cell_perf_start
                seg_start_compressed = compressed_cell_start + dt_in_cell
                seg_end_compressed = seg_start_compressed + seg_dur

                start_text_gen_compressed = None
                if (
                    seg.get("start_text_gen") is not None
                    and seg.get("start_time") is not None
                ):
                    delta_text_gen = (
                        seg["start_text_gen"] - seg["start_time"]
                    )
                    start_text_gen_compressed = (
                        seg_start_compressed + delta_text_gen
                    )

                input_len = seg.get("input_len") or 0
                output_len = seg.get("output_len") or 0
                num_samples = seg.get("num_samples") or 0
                total_tokens = (input_len + output_len) * num_samples
                dur_text = seg.get("duration_text_gen")

                entry = {
                    "cell_index": cell_idx,
                    "start_time": seg_start_compressed,
                    "end_time": seg_end_compressed,
                    "start_text_gen": start_text_gen_compressed,
                    "duration": seg_dur,
                    "duration_text_gen": dur_text,
                    "total_tokens": total_tokens,
                    "tokens_per_sec": seg.get("tokens_per_sec"),
                    "segment_throughput": (
                        total_tokens / seg_dur if seg_dur else None
                    ),
                    "text_gen_throughput": (
                        total_tokens / dur_text if dur_text else None
                    ),
                    "framework": seg.get("framework"),
                    "iteration": seg.get("iteration"),
                    "num_samples": num_samples,
                    "model": seg.get("model"),
                    "batch_size": seg.get("batch_size"),
                    "input_len": input_len,
                    "output_len": output_len,
                    "is_error": seg.get("is_error", False),
                    "error_message": seg.get("error_message"),
                }
                entry.update(self.add_llm_performance_info(entry, perfdata))
                compressed.append(entry)

        compressed.sort(key=lambda r: r["start_time"])
        return compressed


class BaliVisualizationMixin:
    """
    Mixin class that provides BALI visualization capabilities.

    This mixin can be added to visualization classes to provide BALI
    functionality without directly coupling the core visualization code.
    """

    def __init__(self, *args, bali_adapter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bali_adapter = bali_adapter or BaliAdapter()
        self._compressed_bali_segments = []
        self._cached_bali_segments = None

    def _load_bali_segments(self) -> List[Dict]:
        """Load BALI segments, using cache if available."""
        if self._cached_bali_segments is None:
            self._cached_bali_segments = self.bali_adapter.get_segments_for_visualization(
                self.monitor.bali_pid_directory)
            logging.info(f"cached segments: {self._cached_bali_segments}")
        return self._cached_bali_segments

    def _invalidate_bali_cache(self):
        """Invalidate cached BALI segments so the next load fetches from disk."""
        self._cached_bali_segments = None


class BaliMagicsMixin:
    """
    Mixin class that provides BALI magic commands.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bali_adapter = BaliAdapter()

    def _bali_refresh_from_disk(self):
        """Collect BALI segments from disk."""
        pid = getattr(self.monitor, "pid", 0) if self.monitor else 0
        return self.bali_adapter.refresh_segments_from_disk(pid)

    def _bali_segments(self, line: str):
        """Handle the bali_segments magic command."""
        bali_segments = self.bali_adapter.get_segments_dataframe()
        if bali_segments.empty:
            self._bali_refresh_from_disk()
            bali_segments = self.bali_adapter.get_segments_dataframe()

        if bali_segments.empty:
            return print("No BALI segments to display.")

        show(
            bali_segments,
            layout={"topStart": "search", "topEnd": None},
        )

    def _bali_run(self, line: str):
        """Handle the bali_run magic command."""
        count = self._bali_refresh_from_disk()
        print(f"BALI segments: {count} rows")
