#!/bin/bash
#SBATCH --job-name=nojump_simple
#SBATCH --output=nojump_simple_%j.out
#SBATCH --error=nojump_simple_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem=2G

# Optional: path to a virtual environment with JUmPER installed.
# Defaults to the currently active venv (sbatch inherits the environment).
if [ -n "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi

echo "=============================================="
echo "noJUmP Simple Demo (SLURM)"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Python:     $(which python) ($(python --version 2>&1))"
echo "Start time: $(date)"
echo "=============================================="

cd "$SLURM_SUBMIT_DIR"
python run_demo_simple.py

echo ""
echo "Demo finished at $(date)"
echo "Outputs saved to run_simple_<timestamp>/ subdirectory"
