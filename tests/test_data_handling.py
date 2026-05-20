import os
import logging
from unittest.mock import patch

import pytest
import pandas as pd

from jumper_extension.adapters.cell_history import CellHistory
from jumper_extension.adapters.data import PerformanceData


def test_performance_data(temp_dir):
    """Test PerformanceData functionality"""
    # Test initialization and empty dataframe
    data = PerformanceData()
    assert len(data._rows["system"]) == 0
    assert len(data.view()) == 0

    # Test add_sample and view
    data.add_sample("system", {
        "time": 1234567890,
        "cpu_util_avg": 27.5,
        "cpu_util_min": 25.0,
        "cpu_util_max": 30.0,
        "memory": 4.0,
        "io_read_count": 100,
        "io_write_count": 50,
        "io_read": 1024,
        "io_write": 512,
    })
    assert len(data._rows["system"]) == 1
    df = data.view("system")
    assert len(df) == 1 and df["cpu_util_avg"].iloc[0] == 27.5

    # Test CSV export
    csv_file = os.path.join(temp_dir, "test.csv")
    data.export(csv_file, level="system")
    assert os.path.exists(csv_file) and len(pd.read_csv(csv_file)) == 1


def test_performance_data_gpu():
    """Test GPU functionality and slicing"""
    data = PerformanceData()
    data.add_sample("system", {
        "time": 1234567890,
        "cpu_util_avg": 27.5,
        "cpu_util_min": 25.0,
        "cpu_util_max": 30.0,
        "memory": 4.0,
        "gpu_util_avg": 75.0,
        "gpu_band_avg": 20.0,
        "gpu_mem_avg": 60.0,
        "io_read_count": 100,
        "io_write_count": 50,
        "io_read": 1024,
        "io_write": 512,
    })
    data.add_sample("system", {
        "time": 1234567891,
        "cpu_util_avg": 35.0,
        "cpu_util_min": 25.0,
        "cpu_util_max": 40.0,
        "memory": 5.0,
        "gpu_util_avg": 80.0,
        "gpu_band_avg": 25.0,
        "gpu_mem_avg": 65.0,
        "io_read_count": 200,
        "io_write_count": 60,
        "io_read": 2048,
        "io_write": 1024,
    })

    df = data.view("system")
    assert len(df) == 2 and all(
        col in df.columns
        for col in ["gpu_util_avg", "gpu_band_avg", "gpu_mem_avg"]
    )
    assert len(data.view("system", slice_=(0, 0))) == 1


def test_performance_data_multi_level():
    """Test multi-level functionality"""
    data = PerformanceData()

    # Add data to different levels
    data.add_sample("user", {
        "time": 1234567890,
        "cpu_util_avg": 12.5,
        "cpu_util_min": 10.0,
        "cpu_util_max": 15.0,
        "memory": 1.0,
        "io_read_count": 50,
        "io_write_count": 25,
        "io_read": 512,
        "io_write": 256,
    })
    data.add_sample("process", {
        "time": 1234567890,
        "cpu_util_avg": 22.5,
        "cpu_util_min": 20.0,
        "cpu_util_max": 25.0,
        "memory": 2.0,
        "io_read_count": 75,
        "io_write_count": 35,
        "io_read": 768,
        "io_write": 384,
    })
    data.add_sample("system", {
        "time": 1234567890,
        "cpu_util_avg": 32.5,
        "cpu_util_min": 30.0,
        "cpu_util_max": 35.0,
        "memory": 3.0,
        "io_read_count": 100,
        "io_write_count": 50,
        "io_read": 1024,
        "io_write": 512,
    })

    # Test individual level views
    user_df = data.view("user")
    process_df = data.view("process")
    system_df = data.view("system")

    assert len(user_df) == 1 and user_df["cpu_util_avg"].iloc[0] == 12.5
    assert len(process_df) == 1 and process_df["cpu_util_avg"].iloc[0] == 22.5
    assert len(system_df) == 1 and system_df["cpu_util_avg"].iloc[0] == 32.5

    # Test export for specific levels
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        user_file = os.path.join(temp_dir, "user_test.csv")
        system_file = os.path.join(temp_dir, "system_test.csv")

        data.export(user_file, level="user")
        data.export(system_file, level="system")

        # Check that files were created
        assert os.path.exists(user_file)
        assert os.path.exists(system_file)

        # Verify content
        import pandas as pd

        user_data = pd.read_csv(user_file)
        assert (
            len(user_data) == 1 and user_data["cpu_util_avg"].iloc[0] == 12.5
        )


# === Test CellHistory functionality ===
@pytest.fixture
def simple_history():
    history = CellHistory()
    history.start_cell("print('hello')", [])
    history.end_cell(None)
    return history


def test_start_current_end_cell():
    history = CellHistory()
    history.start_cell("print('hello')", [])
    assert history.current_cell["cell_index"] == 0
    assert history.current_cell["wallclock_start_time"] is not None
    history.end_cell(None)
    assert len(history.data) == 1
    row = history.data.iloc[0]
    assert row["wallclock_start_time"] is not None
    assert row["wallclock_end_time"] is not None
    assert row["wallclock_end_time"] >= row["wallclock_start_time"]


def test_view_method(simple_history, capsys, caplog):
    df = simple_history.view()
    assert len(df) == 1
    assert df.iloc[0]["cell_index"] == 0
    assert df.iloc[0]["raw_cell"] == "print('hello')"
    assert df.iloc[0]["start_time"] < df.iloc[0]["end_time"]

    # Test print method
    caplog.set_level(logging.INFO, logger="extension")
    simple_history.print()
    out = capsys.readouterr().out
    assert ("Cell #0" in out) or ("Cell #0" in caplog.text)


def test_show_itable(simple_history):
    with patch("jumper_extension.adapters.cell_history.show") as mock_show:
        simple_history.show_itable()
        assert mock_show.called, "Expected show() to be called"

        df_arg = mock_show.call_args[0][0]  # Get pd.DataFrame
        assert isinstance(df_arg, pd.DataFrame)
        assert "Code" in df_arg.columns
        assert df_arg.loc[0, "Code"] == "print('hello')"


def test_export_method(simple_history, tmp_path):
    json_file = tmp_path / "history.json"
    simple_history.export(str(json_file))
    assert json_file.exists()


def test_csv_export_functionality(simple_history, temp_dir):
    csv_file = os.path.join(temp_dir, "history.csv")
    simple_history.export(csv_file)
    assert os.path.exists(csv_file)


def test_view_operations(simple_history):
    assert not simple_history.data.empty
    assert "start_time" in simple_history.data.columns
    assert "end_time" in simple_history.data.columns
    assert "duration" in simple_history.data.columns
    assert "raw_cell" in simple_history.data.columns
    assert "cell_index" in simple_history.data.columns
    assert "wallclock_start_time" in simple_history.data.columns
    assert "wallclock_end_time" in simple_history.data.columns


def test_is_duration_calculated_correctly(simple_history):
    df = simple_history.view()
    assert (
        df.iloc[0]["duration"]
        == df.iloc[0]["end_time"] - df.iloc[0]["start_time"]
    )
