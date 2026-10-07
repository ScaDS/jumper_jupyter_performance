# noJUmP — Single-Node Performance Monitoring Demo (Pure Python)

This demo shows how to use JUmPER's performance monitoring from a **plain
Python script** — no Jupyter notebook required.  It replicates the workload
from `demos/metric_demo.ipynb` but uses the context-manager API
(`service.monitored()`) to delimit "cells" (code segments) so that each
phase of the workload is tracked separately.

## Prerequisites

- Python ≥ 3.8
- A virtual environment (recommended)
- The `jumper-extension` package installed (see below)

## 1. Create a virtual environment and install JUmPER

```bash
# From the project root (the folder containing pyproject.toml)
python -m venv venv
source venv/bin/activate

# Install the extension in editable mode
pip install -e .

# For GPU monitoring (optional, NVIDIA only):
pip install nvidia-ml-py
```

## 2. Run the demo interactively

From this directory (`demos/noJUmP`):

```bash
python run_demo.py
```

The script runs several workload phases (CPU-bound Pi computation, memory
allocation, disk I/O, and optional CUDA operations) while monitoring
performance.  Each phase is wrapped in a `service.monitored()` context
manager so it appears as a separate "cell" in the performance data.

### Export level

The export level is chosen automatically based on the environment:

- **Running directly** (no SLURM): exports use the `user` level.
- **Running inside a SLURM allocation** (`SLURM_JOB_ID` set): exports use the `slurm` level.

All export formats (JPEG, pickle, CSV) use the same level, and the level
is included in the filenames.

### Output files

After the script finishes, the following files are created in this
directory (shown with `user` level; replace with `slurm` when running
under SLURM):

| File | Description |
|---|---|
| `perf_plot_user.jpeg` | Static performance plot (JPEG) |
| `perf_data_user.pkl` | Pickle file with plot data |
| `perf_data_user.csv` | Performance data as CSV (user level) |
| `session/` | Full JUmPER session directory (CSVs, cell history, manifest) |

## 3. Run via SLURM (HPC)

An sbatch script is provided for submitting the demo to a SLURM cluster.

```bash
# Edit run_demo.sbatch to set your venv path or module loads, then:
sbatch run_demo.sbatch
```

Before submitting, adjust the environment activation in
`run_demo.sbatch`:

- Set `VENV_PATH` to your virtual environment, or
- Adjust the `module load` line to match your HPC environment.

The sbatch script requests 1 node, 8 CPUs, 8 GB RAM, and 15 minutes wall
time.  Adjust these in the `#SBATCH` directives as needed.  Uncomment the
`--gres=gpu:1` line if you want GPU monitoring.

SLURM output (stdout/stderr) is written to `nojump_demo_<jobid>.out` /
`.err`.

## 4. Analyzing the results

An analysis notebook is provided in `analyze_results.ipynb`.  It shows
three ways to load and inspect the exported data:

1. **Session import** — `%import_session session/` via the JUmPER
   extension, giving full interactive plots and reports.
2. **Pickle file** — load `perf_data.pkl` directly with `pickle` or
   `pandas`.
3. **CSV files** — read the per-level CSVs in the `session/` directory
   directly.

Open the notebook in JupyterLab (or VS Code) on your local machine or
any environment with JUmPER installed:

```bash
jupyter lab analyze_results.ipynb
```

No execution of the notebook is required — it is provided as a reference
for how to load and explore the data.
