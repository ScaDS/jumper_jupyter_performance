#!/bin/bash
#SBATCH --job-name=nojump_simple
#SBATCH --output=nojump_simple_%j.out
#SBATCH --error=nojump_simple_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --gres=gpu:1

# Virtual environment with JUmPER installed. Resolution order:
#   1. $VENV_PATH (e.g. sbatch --export=ALL,VENV_PATH=/path/to/venv ...)
#   2. $VIRTUAL_ENV of the shell that submitted the job
#   3. ../../../venv relative to the submit dir (layout from the README)
if [ -z "$VENV_PATH" ]; then
    if [ -n "$VIRTUAL_ENV" ]; then
        VENV_PATH="$VIRTUAL_ENV"
    else
        VENV_PATH="$SLURM_SUBMIT_DIR/../../../venv"
    fi
fi
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "WARNING: no venv found at $VENV_PATH, using $(which python)" >&2
fi

if ! python -c "import jumper_extension" 2>/dev/null; then
    echo "ERROR: jumper_extension not importable with $(which python)." >&2
    echo "Submit with: sbatch --export=ALL,VENV_PATH=/path/to/venv run_demo_simple.sbatch" >&2
    exit 1
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
