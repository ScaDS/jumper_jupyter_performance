import builtins
import os
import time
from unittest.mock import Mock, patch

from jumper_extension.monitor.common import PerformanceMonitor


# Save the original isinstance before patching
original_isinstance = builtins.__dict__["isinstance"]


def is_mock_instance(obj, cls):
    # If obj is a Mock, return True
    if original_isinstance(obj, Mock):
        return True
    # Otherwise, call the original isinstance
    return original_isinstance(obj, cls)


def test_comprehensive_monitor_functionality(mock_cpu_gpu, temp_dir):
    """Test monitor initialization, GPU support, lifecycle, and data collection"""
    # Test basic initialization with GPU
    monitor = PerformanceMonitor()
    monitor.start(0.1)
    assert monitor.interval == 0.1
    assert monitor.running
    assert monitor.nodes.hardware["local"].num_gpus == 1
    assert monitor.nodes.hardware["local"].gpu_name == "NVIDIA GeForce RTX 3080"
    assert monitor.wallclock_start_time is not None
    assert isinstance(monitor.wallclock_start_time, float)

    # Test start/stop lifecycle
    # already started above
    assert monitor.running
    monitor.start()  # Test already running case

    # Test data collection with GPU metrics
    time.sleep(0.2)
    monitor.stop()
    assert not monitor.running
    assert monitor.wallclock_stop_time is not None
    assert monitor.wallclock_stop_time >= monitor.wallclock_start_time

    # Verify data collection
    df = monitor.nodes.view("system")
    assert len(df) > 0
    assert "cpu_util_avg" in df.columns
    assert "gpu_util_avg" in df.columns

    # Test data export
    filename = f"{temp_dir}/test.csv"
    monitor.nodes.export(filename, level="system")
    assert os.path.exists(filename)


def test_cpu_only_and_slurm(mock_cpu_only):
    """Test CPU-only system and SLURM memory detection"""
    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", create=True
    ) as mock_file, patch("os.getuid", return_value=1000), patch.dict(
        os.environ, {"SLURM_JOB_ID": "12345"}
    ):
        mock_file.return_value.__enter__.return_value.read.return_value = (
            b"8589934592"
        )
        monitor = PerformanceMonitor()
        assert monitor.nodes.hardware["local"].num_gpus == 0
        assert monitor.nodes.hardware["local"].memory_limits["slurm"] == 8.0
        assert monitor.nodes.hardware["local"].num_gpus == 0


def test_gpu_failures():
    """Test GPU setup failure scenarios"""
    def _failing_nvml_setup(self) -> dict:
        self._handles = []
        return {}

    def _noop_adlx_setup(self) -> dict:
        self._handles = []
        return {}

    with patch(
        "jumper_extension.monitor.metrics.gpu.nvml.NvmlGpuCollector.setup",
        _failing_nvml_setup,
    ), patch(
        "jumper_extension.monitor.metrics.gpu.adlx.AdlxGpuCollector.setup",
        _noop_adlx_setup,
    ), patch("psutil.Process") as mock_proc:
        mock_proc.return_value.cpu_affinity.return_value = [0, 1]
        mock_proc.return_value.io_counters.return_value = Mock(
            read_count=100,
            write_count=50,
            read_bytes=1024,
            write_bytes=512,
        )

        monitor = PerformanceMonitor()
        assert monitor.nodes.hardware["local"].num_gpus == 0
