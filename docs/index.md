---
title: JUmPER Documentation
---

# JUmPER – Jupyter meets Performance

<p align="left">
  <img src="img/JUmPER01.png" alt="JUmPER logo" width="450">
</p>

Welcome to the JUmPER documentation. JUmPER brings performance engineering to Jupyter and consists of two packages:

## JUmPER IPython Extension

A tool for monitoring the performance of code cells in IPython and Jupyter notebooks.

- Collect CPU, memory, GPU, and I/O metrics for individual cells.
- View textual performance reports for entire sessions or selected cell ranges.
- Explore interactive plots of collected metrics over time.
- Export performance data and cell history for offline analysis.

Get started with the [Installation](getting-started/installation.md) and [Quickstart](getting-started/quickstart.md) guides. For detailed command descriptions, refer to the [Public API](api/index.md) section.

## JUmPER Wrapper Kernel

A Jupyter kernel that wraps other kernels (Python, R, Julia, etc.) while providing jumper-extension performance monitoring.

- Wrap any Jupyter kernel with a single magic command.
- Monitor performance of any wrapped kernel using jumper-extension magic commands.
- Forward code transparently to the wrapped kernel while keeping monitoring local.

Get started with the [Wrapper Kernel Installation](wrapper-kernel/getting-started/installation.md) and [Quickstart](wrapper-kernel/getting-started/quickstart.md) guides. For API details, see the [Wrapper Kernel API](wrapper-kernel/api/index.md).
