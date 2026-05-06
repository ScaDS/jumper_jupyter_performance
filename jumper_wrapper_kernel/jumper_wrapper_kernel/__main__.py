"""
Entry point for running the Jumper Wrapper Kernel.
"""

from .kernel import JumperWrapperKernel
from ipykernel.kernelapp import IPKernelApp

if __name__ == '__main__':
    IPKernelApp.launch_instance(kernel_class=JumperWrapperKernel)
