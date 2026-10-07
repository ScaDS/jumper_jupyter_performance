#!/bin/bash
#SBATCH --job-name=nojump_demo
#SBATCH --output=nojump_demo_%j.out
#SBATCH --error=nojump_demo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --mem=8G
##SBATCH --gres=gpu:1

# --- Activate environment -------------------------------------------------
# Adjust the module load and venv path to match your HPC environment.
module load release/24.04 GCCcore/11.3.0 Python 2>/dev/null || true

# Either set VENV_PATH or edit the line below:
# VENV_PATH="/path/to/your/venv"
if [ -n "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

# Ensure the project root is on PYTHONPATH so jumper_extension is importable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "=============================================="
echo "noJUmP Single-Node Demo (SLURM)"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="

cd "$SCRIPT_DIR"
python run_demo.py

echo ""
echo "Demo finished at $(date)"
echo "Outputs saved to run_<timestamp>/ subdirectory:"
echo "  - perf_plot_slurm.jpeg"
echo "  - perf_data_slurm.pkl"
echo "  - perf_data_slurm.csv"
echo "  - session/"
