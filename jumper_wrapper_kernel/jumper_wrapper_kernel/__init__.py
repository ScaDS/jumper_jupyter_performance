"""
Jumper Wrapper Kernel - A Jupyter kernel that wraps other kernels.

This kernel provides magic commands to wrap existing Jupyter kernels while
preserving the jumper-extension magic commands locally.
"""

__version__ = "0.1.0"

from .kernel import JumperWrapperKernel

__all__ = ["JumperWrapperKernel", "__version__"]
