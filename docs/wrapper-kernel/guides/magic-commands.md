---
title: Magic Commands
---

# Magic Commands

The Jumper Wrapper Kernel provides its own magic commands for kernel management, plus full access to jumper-extension magic commands for performance monitoring.

## Wrapper Magic Commands

### `%list_kernels`

Lists all available Jupyter kernels that can be wrapped.

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
Currently wrapped kernel: python3
```

### `%wrap_kernel`

Wraps an existing Jupyter kernel. All subsequent code (except local magic commands) will be forwarded to this kernel.

**Basic usage:**

```python
%wrap_kernel python3
```

**With permanent kernel spec:**

```python
%wrap_kernel ir --save jumper-r
```

This creates a new kernel spec `jumper-r` that automatically wraps the R kernel on startup.

## Jumper Extension Commands

All jumper-extension magic commands are available and executed locally:

| Command | Description |
|---------|-------------|
| `%perfmonitor_start [interval]` | Start performance monitoring |
| `%perfmonitor_stop` | Stop performance monitoring |
| `%perfmonitor_perfreport` | View performance report |
| `%perfmonitor_plot` | Plot performance data |
| `%cell_history` | View cell execution history |

For complete documentation on jumper-extension commands, see the [jumper-extension documentation](https://scads.github.io/jumper_jupyter_performance/).

## Command Routing

The kernel automatically routes commands:

- **Local execution**: Wrapper magics and jumper-extension magics
- **Forwarded to wrapped kernel**: All other code

This routing is determined by analyzing the cell content before execution.
