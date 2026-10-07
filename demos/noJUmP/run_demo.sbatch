#!/bin/bash
#SBATCH --job-name=nojump_demo
#SBATCH --output=nojump_demo_%j.out
#SBATCH --error=nojump_demo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

module load release/2026 GCCcore/14.2.0 GCC/14.2.0 OpenMPI/5.0.7 PyTorch/2.9.1-CUDA-12.8.0

PROJECT_ROOT="/data/horse/ws/s4122485-noJUmP"
source "$PROJECT_ROOT/venv/bin/activate"

echo "=============================================="
echo "noJUmP Single-Node Demo (SLURM)"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="

cd "$PROJECT_ROOT/jumper_jupyter_performance/demos/noJUmP"
python run_demo.py

echo ""
echo "Demo finished at $(date)"
echo "Outputs saved to run_<timestamp>/ subdirectory:"
echo "  - perf_plot_slurm.jpeg"
echo "  - perf_data_slurm.pkl"
echo "  - perf_data_slurm.csv"
echo "  - session/"
