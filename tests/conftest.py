import tempfile
import os
from unittest.mock import Mock, patch

import pytest
from IPython.testing.globalipapp import get_ipython

# Ensure logging goes to a writable temp directory for the entire test session
if "JUMPER_LOG_DIR" not in os.environ:
    _log_dir = tempfile.mkdtemp(prefix="jumper_test_logs_")
    os.environ["JUMPER_LOG_DIR"] = _log_dir


def _noop_gpu_setup(self) -> None:
    self._handles = []
    self.gpu_memory = 0.0
    self.gpu_name = ""


@pytest.fixture
def ipython():
    # Try to get actual IPython instance, fallback to mock if not available
    ip = get_ipython()
    if ip is not None:
        return ip

    # Create a mock IPython instance with required attributes
    from IPython import InteractiveShell

    # Create a basic InteractiveShell instance for testing
    shell = InteractiveShell.instance()
    shell.events = Mock()
    shell.register_magics = Mock()
    shell.events.register = Mock()
    shell.events.unregister = Mock()

    return shell


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_cpu_base():
    """Mock system with 1 CPU (4 cores)."""
    with patch("psutil.cpu_count", return_value=4), patch(
        "psutil.cpu_percent", return_value=[25.0, 30.0, 20.0, 35.0]
    ), patch("psutil.virtual_memory") as mock_mem, patch(
        "psutil.Process"
    ) as mock_proc, patch(
        "psutil.disk_io_counters"
    ) as mock_disk:
        mock_mem.return_value.total = 8 * 1024**3
        mock_mem.return_value.available = 4 * 1024**3
        mock_proc.return_value.cpu_affinity.return_value = [0, 1, 2, 3]
        # psutil.Process.cpu_percent() returns a single float value
        mock_proc.return_value.cpu_percent.return_value = 25.0
        mock_proc.return_value.memory_full_info.return_value.uss = 2 * 1024**3
        mock_proc.return_value.io_counters.return_value = Mock(
            read_count=100,
            write_count=50,
            read_bytes=1024,
            write_bytes=512,
        )
        mock_disk.return_value = Mock(
            read_count=1000,
            write_count=500,
            read_bytes=10240,
            write_bytes=5120,
        )
        yield


@pytest.fixture
def mock_cpu_only(mock_cpu_base):
    """Mock system with 1 CPU (4 cores) and no GPU."""
    with patch(
        "jumper_extension.monitor.metrics.gpu.nvml.NvmlGpuCollector.setup",
        _noop_gpu_setup,
    ), patch(
        "jumper_extension.monitor.metrics.gpu.adlx.AdlxGpuCollector.setup",
        _noop_gpu_setup,
    ):
        yield


@pytest.fixture
def mock_cpu_gpu(mock_cpu_base):
    """Mock system with 1 CPU (4 cores) and 1 GPU."""
    gpu_handle = Mock()
    mock_mem_info = Mock(
        total=10 * 1024**3,
        used=2 * 1024**3,
        free=8 * 1024**3,
    )
    mock_util_rates = Mock(gpu=75, memory=20)

    def _nvml_gpu_setup(self) -> dict:
        self._handles = [gpu_handle]
        pynvml_mock = Mock()
        pynvml_mock.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        pynvml_mock.nvmlDeviceGetUtilizationRates.return_value = mock_util_rates
        pynvml_mock.nvmlDeviceGetComputeRunningProcesses.return_value = []
        pynvml_mock.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 3080"
        pynvml_mock.NVMLError = Exception
        self._pynvml = pynvml_mock
        return {
            "gpu_memory": round(mock_mem_info.total / (1024**3), 2),
            "gpu_name": "NVIDIA GeForce RTX 3080",
        }

    with patch(
        "jumper_extension.monitor.metrics.gpu.nvml.NvmlGpuCollector.setup",
        _nvml_gpu_setup,
    ), patch(
        "jumper_extension.monitor.metrics.gpu.adlx.AdlxGpuCollector.setup",
        _noop_gpu_setup,
    ):
        yield
