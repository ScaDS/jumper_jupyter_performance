#!/bin/bash
#SBATCH --job-name=jumper_benchmark
#SBATCH --output=jumper_benchmark_%j.out
#SBATCH --error=jumper_benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=4G
##SBATCH --hint=nomultithread

# --- Adjust these as needed -------------------------------------------
# Duration per single run (seconds) and number of repeats per config
DURATION=20
REPEATS=20
# Backends: thread,subprocess_python,native_c  (or a subset)
BACKENDS="native_c,subprocess_python,native_c"
# Frequencies in Hz
FREQUENCIES="0.5,1,2,4"
# ----------------------------------------------------------------------

echo "=============================================="
echo "JUmPER Monitor Benchmark"
echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Start time:    $(date)"
echo "Duration/run:  ${DURATION}s"
echo "Repeats:       ${REPEATS}"
echo "Backends:      ${BACKENDS}"
echo "Frequencies:   ${FREQUENCIES}"
echo "=============================================="

# Ensure the project root is on PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "Project root:  ${PROJECT_ROOT}"
echo ""

module load release/24.04  GCCcore/11.3.0 Python
source /data/horse/ws/s4122485-jumper_test/venv/bin/activate

# Run benchmark
python -m jumper_extension.monitor.benchmark.run_benchmark \
    --duration "$DURATION" \
    --repeats "$REPEATS" \
    --backends "$BACKENDS" \
    --frequencies "$FREQUENCIES" \
    --workers 64

echo ""
echo "Benchmark finished at $(date)"

# Generate plots
echo "Generating plots..."
python -m jumper_extension.monitor.benchmark.plot_results

echo "Done. Results in: ${SCRIPT_DIR}/results/"
