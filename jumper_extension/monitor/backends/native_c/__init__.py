"""Subprocess-based performance monitor using a compiled C collector."""

from jumper_extension.monitor.backends.native_c.monitor import (
    CSubprocessPerformanceMonitor,
)

__all__ = ["CSubprocessPerformanceMonitor"]
