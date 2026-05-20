import logging
from unittest.mock import patch

import pandas as pd

from jumper_extension.adapters.visualizer.backends.matplotlib import MatplotlibPerformanceVisualizer
from jumper_extension.adapters.visualizer.backends.plotly import PlotlyPerformanceVisualizer
from jumper_extension.ipython.magics import PerfmonitorMagics
from jumper_extension.core.service import build_perfmonitor_magic_adapter
from jumper_extension.ipython.extension import (
    load_ipython_extension,
    unload_ipython_extension,
)


def test_initialization_and_basic_operations(ipython, mock_cpu_only, caplog):
    """Test initialization, start/stop, and basic operations"""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())
    assert not magics.magic_adapter.service.monitor.running

    # Test start/stop cycle with valid interval parsing
    magics.perfmonitor_start("0.5")
    assert magics.magic_adapter.service.monitor.interval == 0.5
    magics.perfmonitor_stop("")

    # Test already running
    magics.perfmonitor_start("")
    magics.perfmonitor_start("")  # Already running
    magics.perfmonitor_stop("")
    assert not magics.magic_adapter.service.monitor.running

    # Test invalid interval (no monitor running)
    caplog.set_level(logging.WARNING, logger="extension")
    magics.perfmonitor_start("invalid")  # Invalid interval
    assert "Invalid interval value: invalid" in caplog.text
    assert not magics.magic_adapter.service.monitor.running


def test_no_monitor_error_cases(ipython, mock_cpu_only):
    """Test commands that require active monitor"""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())
    magics.perfmonitor_resources("")
    magics.perfmonitor_plot("")
    magics.perfmonitor_perfreport("")
    magics.perfmonitor_export_perfdata("")


def test_resources_and_gpu(ipython, mock_cpu_gpu):
    """Test resources display with GPU"""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())
    magics.perfmonitor_start("")
    magics.perfmonitor_resources("")
    magics.perfmonitor_stop("")


def test_cell_operations(ipython, mock_cpu_only):
    """Test cell history and reports"""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())

    # Test cell history tracking and command
    cell_info = type("Info", (), {"raw_cell": "test"})()
    magics.pre_run_cell(cell_info)
    result = type("Result", (), {"result": None})()
    magics.post_run_cell(result)

    with patch.object(magics.magic_adapter.service.cell_history, "print"):
        magics.show_cell_history("")

    # Test auto-reports
    magics.perfmonitor_start("")
    magics.perfmonitor_enable_perfreports("")
    with patch.object(magics.magic_adapter.service.reporter, "print"):
        magics.post_run_cell(result)
    magics.perfmonitor_disable_perfreports("")

    # Test auto-reports with level option
    magics.perfmonitor_enable_perfreports("--level user")
    assert magics.magic_adapter.service.settings.perfreports.level == "user"
    # First call to post_run_cell resets _skip_report flag
    magics.post_run_cell(result)
    with patch.object(
            magics.magic_adapter.service.reporter,
            "display"
    ) as mock_display:
        magics.post_run_cell(result)
        # Verify that the reporter.print was called with the correct level
        mock_display.assert_called_with(cell_range=None, level="user")
    magics.perfmonitor_disable_perfreports("")
    magics.perfmonitor_stop("")


def test_plot_scenarios(ipython, mock_cpu_only):
    """Test plotting with different data scenarios"""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())
    magics.perfmonitor_start("")

    # Test invalid cell
    magics.perfmonitor_plot("--cell invalid")

    # Test empty data
    with patch.object(
        magics.magic_adapter.service.monitor.nodes,
        "view",
        return_value=pd.DataFrame(columns=["time"]),
    ):
        magics.perfmonitor_plot("")

    # Test with data
    df = pd.DataFrame({"time": [1.0, 2.0], "cpu_util_avg": [50.0, 60.0]})
    with patch.object(
        magics.magic_adapter.service.monitor.nodes, "view", return_value=df
    ), patch.object(magics.magic_adapter.service.visualizer, "plot"), patch.object(
        magics.magic_adapter.service.monitor, "start_time", 0.0
    ):
        magics.perfmonitor_plot("")

    # Test with cell filter
    with patch("time.time", side_effect=[1.0, 2.0]):
        cell_info = type("Info", (), {"raw_cell": "test"})()
        magics.pre_run_cell(cell_info)
        magics.post_run_cell(type("Result", (), {"result": None})())

    with patch.object(
        magics.magic_adapter.service.monitor.nodes, "view", return_value=df
    ), patch.object(magics.magic_adapter.service.visualizer, "plot"), patch.object(
        magics.magic_adapter.service.monitor, "start_time", 0.0
    ):
        magics.perfmonitor_plot("--cell 0")

    magics.perfmonitor_stop("")


def test_plot_backend_selection_via_magic(ipython, mock_cpu_only):
    """Test selecting matplotlib/plotly backend via %perfmonitor_plot."""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())
    magics.perfmonitor_start("")

    # Add one executed cell so --cell 0 is valid and filter_perfdata has range
    cell_info = type("Info", (), {"raw_cell": "x = 1"})()
    magics.pre_run_cell(cell_info)
    magics.post_run_cell(type("Result", (), {"result": None})())

    ch = magics.magic_adapter.service.cell_history.view()
    start_t = float(ch.iloc[0]["start_time"])
    end_t = float(ch.iloc[0]["end_time"])
    if end_t <= start_t:
        end_t = start_t + 0.01

    df = pd.DataFrame(
        {
            "time": [start_t, end_t],
            "cpu_util_min": [10.0, 20.0],
            "cpu_util_avg": [30.0, 40.0],
            "cpu_util_max": [50.0, 60.0],
        }
    )

    service = magics.magic_adapter.service
    monitor = service.monitor

    with patch.object(monitor.nodes, "view", return_value=df), patch.object(
        PlotlyPerformanceVisualizer, "_render_direct_plot"
    ) as mock_plotly_render:
        magics.perfmonitor_plot(
            "--backend plotly --metrics cpu_summary --level process --cell 0"
        )
        assert isinstance(service.visualizer, PlotlyPerformanceVisualizer)
        assert service.settings.visualizer_backend == "plotly"
        assert mock_plotly_render.called

    # No --backend: should use default from settings ("plotly")
    with patch.object(monitor.nodes, "view", return_value=df), patch.object(
        PlotlyPerformanceVisualizer, "_render_direct_plot"
    ) as mock_plotly_default_render:
        magics.perfmonitor_plot("--metrics cpu_summary --level process --cell 0")
        assert isinstance(service.visualizer, PlotlyPerformanceVisualizer)
        assert mock_plotly_default_render.called

    with patch.object(monitor.nodes, "view", return_value=df), patch.object(
        MatplotlibPerformanceVisualizer, "_render_direct_plot"
    ) as mock_matplotlib_render:
        magics.perfmonitor_plot(
            "--backend matplotlib --metrics cpu_summary --level process --cell 0"
        )
        assert isinstance(service.visualizer, MatplotlibPerformanceVisualizer)
        assert service.settings.visualizer_backend == "matplotlib"
        assert mock_matplotlib_render.called

    # No --backend again: should keep using settings default ("matplotlib")
    with patch.object(monitor.nodes, "view", return_value=df), patch.object(
        MatplotlibPerformanceVisualizer, "_render_direct_plot"
    ) as mock_matplotlib_default_render:
        magics.perfmonitor_plot("--metrics cpu_summary --level process --cell 0")
        assert isinstance(service.visualizer, MatplotlibPerformanceVisualizer)
        assert mock_matplotlib_default_render.called

    magics.perfmonitor_stop("")


def test_perfreport_scenarios(ipython, mock_cpu_only):
    """Test performance reporting scenarios"""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())

    # Test no monitor
    magics.perfmonitor_perfreport("")

    magics.perfmonitor_start("")

    # Test invalid cell for perfreport command
    magics.perfmonitor_perfreport("--cell invalid")

    # Add cell to history
    with patch("time.time", side_effect=[1.0, 2.0]):
        cell_info = type("Info", (), {"raw_cell": "test"})()
        magics.pre_run_cell(cell_info)
        magics.post_run_cell(type("Result", (), {"result": None})())

    # Test empty data
    with patch.object(
        magics.magic_adapter.service.monitor.nodes,
        "view",
        return_value=pd.DataFrame(columns=["time"]),
    ):
        magics.magic_adapter.service.reporter.print()

    # Test with full data
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "cpu_util_avg": [50.0, 60.0],
            "memory": [4.0, 4.5],
            "gpu_util_avg": [30.0, 40.0],
            "gpu_mem_avg": [2.0, 2.5],
        }
    )
    with patch.object(
            magics.magic_adapter.service.monitor.nodes,
            "view",
            return_value=df
    ):
        magics.magic_adapter.service.reporter.print()
        magics.magic_adapter.service.reporter.print(
            (0, 0)
        )  # Custom cell marks (use integer indices)
        magics.perfmonitor_perfreport("--cell 0")  # Via command

    # Test with missing columns
    df_partial = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "cpu_util_avg": [50.0, 60.0],
            "memory": [4.0, 4.5],
        }
    )
    with patch.object(
            magics.magic_adapter.service.monitor.nodes,
            "view",
            return_value=df_partial):
        magics.magic_adapter.service.reporter.print()

    magics.perfmonitor_stop("")


def test_export_and_help(ipython, mock_cpu_only):
    """Test export functions and help"""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())

    # Test exports without monitor
    magics.perfmonitor_export_perfdata("")

    # Test exports with monitor
    magics.perfmonitor_start("")
    df = pd.DataFrame({"time": [1.0]})
    with patch.object(
        magics.magic_adapter.service.monitor.nodes,
        "view",
        return_value=df,
    ):
        magics.perfmonitor_export_perfdata("--name custom_perf")
        assert "custom_perf" in ipython.user_ns
        assert ipython.user_ns["custom_perf"].equals(df)
    ipython.user_ns.pop("custom_perf", None)
    with patch.object(magics.magic_adapter.service.monitor.nodes, "export"):
        magics.perfmonitor_export_perfdata("")
        magics.perfmonitor_export_perfdata("--file custom.csv")
    magics.perfmonitor_stop("")

    # Test cell history export
    ch_df = pd.DataFrame({"cell_index": [0]})
    with patch.object(
        magics.magic_adapter.service.cell_history,
        "view",
        return_value=ch_df,
    ):
        magics.perfmonitor_export_cell_history("--name custom_history")
        assert "custom_history" in ipython.user_ns
        assert ipython.user_ns["custom_history"].equals(ch_df)
    ipython.user_ns.pop("custom_history", None)
    with patch.object(magics.magic_adapter.service.cell_history, "export"):
        magics.perfmonitor_export_cell_history("")
        magics.perfmonitor_export_cell_history("--file custom.json")

    # Test CSV export
    with patch.object(
            magics.magic_adapter.service.cell_history,
            "export"
    ) as mock_export:
        magics.perfmonitor_export_cell_history("--file test.csv")
        mock_export.assert_called_with("test.csv")

    # Test help
    magics.perfmonitor_help("")


def test_extension_lifecycle(ipython, mock_cpu_only):
    """Test IPython extension load/unload"""
    with patch.object(ipython.events, "register"), patch.object(
        ipython, "register_magics"
    ):
        load_ipython_extension(ipython)

    # Test unload with monitor
    from jumper_extension.ipython.extension import _perfmonitor_magics

    _perfmonitor_magics.perfmonitor_start("")
    with patch.object(ipython.events, "unregister"):
        unload_ipython_extension(ipython)

    # Test unload without magics
    from jumper_extension.ipython import extension

    extension._perfmonitor_magics = None
    unload_ipython_extension(ipython)


def test_start_write_script_magic(ipython):
    """Ensure %start_write_script delegates to magic_adapter.start_write_script"""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())
    with patch.object(magics.magic_adapter, "start_write_script") as mock_start:
        magics.start_write_script("")
        mock_start.assert_called_once_with("")

        magics.start_write_script("output.py")
        mock_start.assert_called_with("output.py")


def test_end_write_script_magic(ipython):
    """Ensure %end_write_script delegates to magic_adapter.end_write_script"""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())
    with patch.object(magics.magic_adapter, "end_write_script") as mock_end:
        magics.end_write_script("")
        mock_end.assert_called_once_with("")


def test_load_perfdata_csv(ipython, tmp_path):
    """Load perfdata from CSV and verify returned DataFrame."""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())

    # Create a CSV with all required base columns
    import os
    df = pd.DataFrame([
        {
            "time": 1.23,
            "memory": 4.56,
            "io_read_count": 100,
            "io_write_count": 200,
            "io_read": 1024,
            "io_write": 2048,
            "cpu_util_avg": 12.0,
            "cpu_util_min": 10.0,
            "cpu_util_max": 15.0,
        },
        {
            "time": 2.34,
            "memory": 5.67,
            "io_read_count": 150,
            "io_write_count": 250,
            "io_read": 1536,
            "io_write": 3072,
            "cpu_util_avg": 34.0,
            "cpu_util_min": 30.0,
            "cpu_util_max": 38.0,
        },
    ])
    csv_path = os.path.join(tmp_path, "perfdata.csv")
    df.to_csv(csv_path, index=False)

    # Load without --file flag (positional argument)
    magics.perfmonitor_load_perfdata(str(csv_path))

    # Verify DataFrame was pushed to IPython namespace
    loaded_var = magics.magic_adapter.service.settings.loaded_vars.perfdata
    assert loaded_var in ipython.user_ns
    loaded_df = ipython.user_ns[loaded_var]
    assert len(loaded_df) == 2
    assert float(loaded_df.loc[0, "time"]) == 1.23
    assert float(loaded_df.loc[0, "memory"]) == 4.56
    assert float(loaded_df.loc[0, "cpu_util_avg"]) == 12.0


def test_load_perfdata_json(ipython, tmp_path):
    """Load perfdata from JSON and verify returned DataFrame."""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())

    import json, os
    rows = [
        {
            "time": 10.0,
            "memory": 1.0,
            "io_read_count": 50,
            "io_write_count": 100,
            "io_read": 512,
            "io_write": 1024,
            "cpu_util_avg": 5.0,
            "cpu_util_min": 3.0,
            "cpu_util_max": 7.0,
        },
        {
            "time": 11.0,
            "memory": 2.0,
            "io_read_count": 75,
            "io_write_count": 125,
            "io_read": 768,
            "io_write": 1536,
            "cpu_util_avg": 15.0,
            "cpu_util_min": 13.0,
            "cpu_util_max": 17.0,
        },
    ]
    json_path = os.path.join(tmp_path, "perfdata.json")
    with open(json_path, "w") as f:
        json.dump(rows, f)

    # Load without --file flag (positional argument)
    magics.perfmonitor_load_perfdata(str(json_path))

    # Verify DataFrame was pushed to IPython namespace
    loaded_var = magics.magic_adapter.service.settings.loaded_vars.perfdata
    assert loaded_var in ipython.user_ns
    loaded_df = ipython.user_ns[loaded_var]
    assert len(loaded_df) == 2
    assert float(loaded_df.loc[1, "time"]) == 11.0
    assert float(loaded_df.loc[1, "cpu_util_avg"]) == 15.0


def test_load_cell_history_csv(ipython, tmp_path):
    """Load cell history from CSV and verify returned DataFrame."""
    magics = PerfmonitorMagics(ipython, build_perfmonitor_magic_adapter())

    import os
    ch_df = pd.DataFrame([
        {
            "cell_index": 0,
            "raw_cell": "print('hi')",
            "start_time": 1.0,
            "end_time": 2.0,
            "duration": 1.0,
            "wallclock_start_time": 1700000000.0,
            "wallclock_end_time": 1700000001.0,
        }
    ])
    csv_path = os.path.join(tmp_path, "cell_history.csv")
    ch_df.to_csv(csv_path, index=False)

    # Load without --file flag (positional argument)
    magics.perfmonitor_load_cell_history(str(csv_path))

    # Verify DataFrame was pushed to IPython namespace
    loaded_var = magics.magic_adapter.service.settings.loaded_vars.cell_history
    assert loaded_var in ipython.user_ns
    loaded_df = ipython.user_ns[loaded_var]
    assert len(loaded_df) == 1
    assert int(loaded_df.loc[0, "cell_index"]) == 0
    assert loaded_df.loc[0, "raw_cell"] == "print('hi')"
    assert float(loaded_df.loc[0, "duration"]) == 1.0
