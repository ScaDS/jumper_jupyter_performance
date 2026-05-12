import os
import json
import glob
from typing import List, Dict, Tuple
import matplotlib as mpl
import logging

logger = logging.getLogger("extension")

class BaliResultsParser:
    def __init__(self, base_search_path: str = "."):
        self.base_search_path = base_search_path
        self.colormap = mpl.colors.LinearSegmentedColormap.from_list(
            "blue_to_orange",
            ['#51829B','#9BB0C1','#F6995C']
        )
        self.colormap_energy = mpl.colors.LinearSegmentedColormap.from_list(
            "yellow_to_red",
            ['#EADFB4','#F6995C','#874C62']
        )

    def _find_bali_directories(self, pid: int) -> List[str]:
        pid_dir = os.path.join(self.base_search_path, "bali_results", str(pid))
        idx_dirs = [
            d
            for d in glob.glob(os.path.join(pid_dir, "*"))
            if os.path.isdir(d)
        ]
        return sorted(
            idx_dirs,
            key=lambda x: (
                int(os.path.basename(x))
                if os.path.basename(x).isdigit()
                else 0
            ),
        )

    def _load_json(self, filepath: str) -> Dict:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def extract_segment(
        self, benchmark_data: Dict, config_data: Dict
    ) -> List[Dict]:
        segments = []
        if benchmark_data:
            # Get the single framework (first and only key)
            framework = next(iter(benchmark_data))
            framework_data = benchmark_data[framework]

            for iteration_key, iteration_data in framework_data.items():
                start_time = iteration_data.get("start_time")
                end_time = iteration_data.get("end_time")
                # ``generation_time`` is the duration of the text-generation
                # phase; ``tokenize_time`` and ``setup_time`` are also
                # durations (not absolute timestamps).
                generation_time = iteration_data.get("generation_time")
                duration = (
                    (end_time - start_time)
                    if (start_time is not None and end_time is not None)
                    else None
                )
                start_text_gen = (
                    end_time - generation_time
                    if (end_time is not None and generation_time is not None)
                    else None
                )

                segments.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_text_gen": start_text_gen,
                        "duration": duration,
                        "duration_text_gen": generation_time,
                        "tokens_per_sec": iteration_data.get("token_per_sec"),
                        "framework": framework,
                        "iteration": iteration_key,
                        "model": config_data.get("model_name"),
                        "num_samples": config_data.get("num_samples"),
                        "batch_size": config_data.get("batch_size"),
                        "input_len": config_data.get("input_len"),
                        "output_len": config_data.get("output_len"),
                        "is_error": False,
                    }
                )
        else:
            # Create segment with missing timing data when benchmark_data
            # doesn't exist
            segments.append(
                {
                    "start_time": None,
                    "end_time": None,
                    "start_text_gen":None,
                    "duration": None,
                    "duration_text_gen": None,
                    "tokens_per_sec": None,
                    "framework": "unknown",
                    "iteration": "0",
                    "model": config_data.get("model_name"),
                    "batch_size": config_data.get("batch_size"),
                    "input_len": config_data.get("input_len"),
                    "output_len": config_data.get("output_len"),
                    "is_error": False,
                }
            )
        return segments

    def extract_error_segments(self, error_data: Dict, config_data: Dict) -> List[Dict]:
        """Extract error segments from errors.json file."""
        segments = []
        if error_data:
            for framework, error_info in error_data.items():
                start_time = error_info.get("start_time")
                end_time = error_info.get("end_time")
                
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_text_gen": None,
                    "duration": error_info.get("duration"),
                    "tokens_per_sec": None,  # No tokens for failed segments
                    "framework": framework,
                    "iteration": "error",
                    "model": config_data.get("model_name"),
                    "batch_size": config_data.get("batch_size"),
                    "input_len": config_data.get("input_len"),
                    "output_len": config_data.get("output_len"),
                    "is_error": True,
                    "error_message": error_info.get("error", "Unknown error"),
                })
        return segments

    def collect_all_bali_segments(self, pid: int) -> List[Dict]:
        result_dirs = self._find_bali_directories(pid)
        if not result_dirs:
            return []

        segments = []
        logger.info(f"\nBALI result_dirs to plot:{result_dirs}")
        for directory in result_dirs:
            for config_path in glob.glob(
                os.path.join(directory, "*/*/*/*/*/config.json")
            ):
                config_data = self._load_json(config_path)
                benchmark_path = os.path.join(
                    os.path.dirname(config_path), "benchmark_results.json"
                )
                benchmark_data = self._load_json(benchmark_path)

                segments.extend(
                    self.extract_segment(benchmark_data, config_data)
                )
                
                # Also check for error segments
                error_path = os.path.join(
                    os.path.dirname(config_path), "errors.json"
                )
                error_data = self._load_json(error_path)
                if error_data:
                    segments.extend(
                        self.extract_error_segments(error_data, config_data)
                    )
        return sorted(
            [s for s in segments if s["start_time"]],
            key=lambda x: x["start_time"],
        )

    def get_tokens_per_sec_range(
        self, segments: List[Dict]
    ) -> Tuple[float, float]:
        values = [s["tokens_per_sec"] for s in segments if s["tokens_per_sec"]]
        return (min(values), max(values)) if values else (0.0, 100.0)
    
    def get_energy_efficiency_range(
            self, segments: List[Dict]
    ) -> Tuple[float, float]:
        values = [s["token_per_joule_full_segment"] for s in segments if s["token_per_joule_full_segment"]]
        return (min(values), max(values)) if values else (0.0, 100.0)

    def get_color_for_tokens_per_sec(
        self, tokens_per_sec: float, vmin: float, vmax: float
    ) -> Tuple[float, float, float, float]:
        # Handle None tokens_per_sec (for error segments or missing data)

        if vmax == vmin:
            return self.colormap(0.5)
        normalized = max(
            0.0, min(1.0, (tokens_per_sec - vmin) / (vmax - vmin))
        )
        return self.colormap(normalized)
    
    def get_color_for_energy_efficiency(
        self, tokens_per_sec: float, vmin: float, vmax: float
    ) -> Tuple[float, float, float, float]:
         
        if vmax == vmin:
            return self.colormap(0.5)
        normalized = max(
            0.0, min(1.0, (tokens_per_sec - vmin) / (vmax - vmin))
        )
        return self.colormap_energy(normalized)