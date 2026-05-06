---
title: Quickstart
---

# Quickstart

This guide shows you how to use the Jumper Wrapper Kernel in a few simple steps.

## Step 1: Select the Kernel

1. Start Jupyter Notebook or JupyterLab
2. Create a new notebook
3. Select **Jumper Wrapper Kernel** as your kernel

## Step 2: List Available Kernels

See which kernels can be wrapped:

```python
%list_kernels
```

Output:
```
Available Jupyter Kernels:
--------------------------------------------------
  python3: Python 3 (ipykernel) (python)
  ir: R (r)
  julia-1.9: Julia 1.9 (julia)
--------------------------------------------------
```

## Step 3: Wrap a Kernel

Wrap your desired kernel:

```python
%wrap_kernel python3
```

Output:
```
Successfully wrapped kernel: python3
Hint: Refresh the page (without restarting the kernel) to enable syntax highlighting for the wrapped language.
```

## Step 4: Use Performance Monitoring

Now you can use jumper-extension commands while running code on the wrapped kernel:

```python
# Start monitoring (handled locally)
%perfmonitor_start

# Run code on the wrapped kernel
import numpy as np
x = np.random.rand(1000, 1000)
y = np.dot(x, x.T)

# View performance report (handled locally)
%perfmonitor_perfreport
```

## How It Works

The Jumper Wrapper Kernel acts as a proxy:

1. **Magic commands** from jumper-extension are intercepted and executed locally
2. **All other code** is forwarded to the wrapped kernel
3. **Output** is streamed back to the notebook

This allows you to monitor performance of any Jupyter kernel, regardless of its language.
