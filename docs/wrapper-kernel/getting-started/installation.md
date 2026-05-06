---
title: Installation
---

# Installation

## Requirements

- Python >= 3.8
- ipykernel >= 6.0
- jupyter_client >= 7.0
- jumper-extension >= 0.3.0

## Standard Installation

Install the package from PyPI:

```bash
pip install jumper_wrapper_kernel
```

Then install the kernel specification:

```bash
python -m jumper_wrapper_kernel.install install
```

## Installation for Virtual Environments

If you're using a virtual environment or conda, install to `sys.prefix`:

```bash
python -m jumper_wrapper_kernel.install install --sys-prefix
```

## Development Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ScaDS/jumper_jupyter_performance.git
cd jumper_jupyter_performance/jumper_wrapper_kernel
pip install -e .
python -m jumper_wrapper_kernel.install install
```

## Verification

After installation, verify the kernel is available:

```bash
jupyter kernelspec list
```

You should see `jumper_wrapper` in the list of available kernels.

## Uninstallation

To remove the kernel:

```bash
python -m jumper_wrapper_kernel.install uninstall
pip uninstall jumper_wrapper_kernel
```
