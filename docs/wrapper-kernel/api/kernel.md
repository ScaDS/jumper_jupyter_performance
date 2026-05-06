---
title: Kernel
---

# Kernel Module

The kernel module contains the main `JumperWrapperKernel` class and the `JumperWrapperMagics` class for magic command handling.

## JumperWrapperMagics

::: jumper_wrapper_kernel.kernel.JumperWrapperMagics
    options:
      show_root_heading: true
      members:
        - list_kernels
        - wrap_kernel

## JumperWrapperKernel

::: jumper_wrapper_kernel.kernel.JumperWrapperKernel
    options:
      show_root_heading: true
      members:
        - do_execute
        - do_shutdown
        - do_complete
        - do_inspect
        - kernel_info
