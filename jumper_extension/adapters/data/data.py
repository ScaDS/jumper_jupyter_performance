import json
import os
from typing import Optional

import pandas as pd
import logging

from jumper_extension.utilities import get_available_levels, load_dataframe_from_file
from jumper_extension.core.messages import (
    ExtensionErrorCode,
    ExtensionInfoCode,
    EXTENSION_ERROR_MESSAGES,
    EXTENSION_INFO_MESSAGES,
)

logger = logging.getLogger("extension")


class PerformanceData:
    def __init__(self):
        self.levels = get_available_levels()
        # Base columns used only for offline load() validation
        self._base_columns = [
            "time",
            "memory",
            "io_read_count",
            "io_write_count",
            "io_read",
            "io_write",
            "cpu_util_avg",
            "cpu_util_min",
            "cpu_util_max",
        ]
        # O(1) append buffer; DataFrame built lazily in view()
        self._rows: dict[str, list[dict]] = {level: [] for level in self.levels}
        # Optional per-level column schema for empty-DataFrame shape
        self._schema_columns: Optional[dict[str, list[str]]] = None
        self._file_writers = {
            "json": self._write_json,
            "csv": self._write_csv,
        }
        self._file_readers = {
            "json": pd.read_json,
            "csv": pd.read_csv,
        }

    def _validate_level(self, level):
        if level not in self.levels:
            raise ValueError(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.INVALID_LEVEL
                ].format(level=level, levels=self.levels)
            )

    def _attach_cell_index(self, df, cell_history) -> pd.DataFrame:
        result = df.copy()
        result["cell_index"] = pd.NA
        if "time" not in result.columns:
            return result
        times = result["time"].to_numpy()
        for row in cell_history.data.itertuples(index=False):
            mask = (times >= row.start_time) & (times <= row.end_time)
            result.loc[mask, "cell_index"] = row.cell_index
        return result

    def _write_json(self, filename: str, df: pd.DataFrame) -> None:
        with open(filename, "w") as f:
            json.dump(df.to_dict("records"), f, indent=2)

    def _write_csv(self, filename: str, df: pd.DataFrame) -> None:
        df.to_csv(filename, index=False)

    def view(self, level="process", slice_=None, cell_history=None):
        self._validate_level(level)
        rows = self._rows.get(level, [])
        if slice_ is not None:
            rows = rows[slice_[0]: slice_[1] + 1]
        df = pd.DataFrame(rows)
        if df.empty and self._schema_columns and level in self._schema_columns:
            df = pd.DataFrame(columns=self._schema_columns[level])
        return (
            self._attach_cell_index(df, cell_history)
            if cell_history is not None
            else df
        )

    def add_sample(self, level: str, row: dict) -> None:
        self._validate_level(level)
        self._rows[level].append(row)

    def export(
        self,
        filename="performance_data.csv",
        level="process",
        cell_history=None,
    ):
        self._validate_level(level)
        df_to_write = self.view(level=level, cell_history=cell_history)
        if df_to_write.empty:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[ExtensionErrorCode.NO_PERFORMANCE_DATA]
            )
            return
        _, ext = os.path.splitext(filename)
        format = ext.lower().lstrip(".") or "csv"
        if not format:
            format = "csv"
            filename += ".csv"
        writer = self._file_writers.get(format)
        if writer is None:
            logger.warning(
                EXTENSION_ERROR_MESSAGES[
                    ExtensionErrorCode.UNSUPPORTED_FORMAT
                ].format(
                    format=format,
                    supported_formats=", ".join(["json", "csv"]),
                )
            )
            return
        writer(filename, df_to_write)
        logger.info(
            EXTENSION_INFO_MESSAGES[ExtensionInfoCode.EXPORT_SUCCESS].format(
                filename=filename
            )
        )

    def load(self, filename: str) -> Optional[pd.DataFrame]:
        return load_dataframe_from_file(
            filename,
            self._file_readers,
            self._base_columns,
            entity_name="performance data",
        )
