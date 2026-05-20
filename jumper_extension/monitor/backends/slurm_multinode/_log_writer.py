"""Log writer for multi-node monitoring results.

Writes per-node, per-sample performance data to a structured log file
(JSON Lines format) so it can be inspected or post-processed without
touching the visualizer.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Dict, TextIO, Optional

logger = logging.getLogger("extension")


class MultinodeLogWriter:
    """Appends JSON-Lines entries for every sample received from any node.

    Each line is a self-contained JSON object::

        {
            "node": "node01",
            "timestamp": 1714000000.123,
            "level": "process",
            "sample_index": 42,
            "cpu_util": [...],
            "memory": 1.23,
            "gpu_util": [...],
            "gpu_band": [...],
            "gpu_mem": [...],
            "io_counters": [...]
        }

    The writer is thread-safe; multiple collector threads may call
    :meth:`write_sample` concurrently.
    """

    def __init__(self, log_path: str = "jumper_multinode.jsonl"):
        self._log_path = log_path
        self._lock = threading.Lock()
        self._file: Optional[TextIO] = None
        self._sample_counters: Dict[str, int] = {}

    @property
    def log_path(self) -> str:
        return self._log_path

    def open(self) -> None:
        """Open the log file for appending."""
        directory = os.path.dirname(self._log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._file = open(self._log_path, "a")
        logger.info(
            f"[JUmPER]: Multinode log file opened: {self._log_path}"
        )

    def write_sample(
        self,
        node: str,
        level: str,
        wallclock: float,
        perf_time: float,
        cpu_util: list[float],
        memory: float,
        gpu_util: list[float],
        gpu_band: list[float],
        gpu_mem: list[float],
        io_counters: list[float],
    ) -> None:
        """Append one sample to the log file."""
        if self._file is None:
            return

        with self._lock:
            key = f"{node}:{level}"
            idx = self._sample_counters.get(key, 0)
            self._sample_counters[key] = idx + 1

            entry = {
                "node": node,
                "timestamp": wallclock,
                "perf_time": perf_time,
                "level": level,
                "sample_index": idx,
                "cpu_util": cpu_util,
                "memory": memory,
                "gpu_util": gpu_util,
                "gpu_band": gpu_band,
                "gpu_mem": gpu_mem,
                "io_counters": io_counters,
            }
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            logger.info(
                f"[JUmPER]: Multinode log file closed: {self._log_path}"
            )
