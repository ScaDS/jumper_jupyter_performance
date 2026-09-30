#!/bin/bash
#SBATCH --job-name=ablations
#SBATCH --partition=romeo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# One sweep of the ablation harness on the romeo partition.
#
#   sbatch scripts/romeo.sbatch suite=minian_cell_77 protocol=pilot
#
# Everything after the script name is passed to `cli.run` verbatim, so the
# hydra tokens are the same ones that work interactively. Two things this
# script exists for, and both are site facts rather than experiment ones:
# the venv is built against the rome Python module and will not start without
# it, and the api key must reach the kernel without being written down.
set -euo pipefail

VENV=/data/horse/ws/albu670g-jumper/kernels/jumper-romeo

# Slurm copies the batch script to its own spool directory before running it,
# so the script's own location says nothing about where the experiment is.
# The directory sbatch was called from does, and ABLATIONS_ROOT overrides it.
EXPERIMENT="${ABLATIONS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${EXPERIMENT}/jumper_ablations" ]]; then
    echo "no experiment at '${EXPERIMENT}'; submit from its directory or" \
         "set ABLATIONS_ROOT" >&2
    exit 1
fi

module load release/25.06 GCCcore/13.3.0 Python/3.12.3
source "${VENV}/bin/activate"
cd "${EXPERIMENT}"

if [[ -z "${JUMPER_AI_API_KEY:-}" ]]; then
    echo "JUMPER_AI_API_KEY is unset; export it before sbatch" >&2
    exit 1
fi

# The reviewer is useless without the endpoint, and a compute node's route to
# it is not the login node's. Failing here costs seconds; failing after the
# prefix has run costs the whole allocation.
if ! curl -sf -o /dev/null --max-time 20 \
        -H "Authorization: Bearer ${JUMPER_AI_API_KEY}" \
        https://llm.scads.ai/v1/models; then
    echo "llm.scads.ai is not reachable from $(hostname)" >&2
    exit 1
fi

# Results go to the project filesystem, never to the home directory. A full
# replay exports a benchmark session per measurement, so one sweep over the
# minian usecases runs to hundreds of gigabytes; home has a 50 GB quota. The
# harness puts each pass's temporary files inside the run directory, so this
# one variable moves the whole lot.
DEFAULT_STORAGE=/data/horse/ws/albu670g-jumper/ablations
export JUMPER_ABLATION_STORAGE="${JUMPER_ABLATION_STORAGE:-$DEFAULT_STORAGE}"
mkdir -p "${JUMPER_ABLATION_STORAGE}"
echo "storage: ${JUMPER_ABLATION_STORAGE}"
df -h "${JUMPER_ABLATION_STORAGE}" | tail -1

# minian's dask arrays are chunked for one machine; the default thread pool
# would oversubscribe the allocation and add timing noise to every benchmark.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# An array task takes its share of the pass grid. Filled in here rather than
# on the command line so that the hydra tokens do not have to quote shell
# variables: one missed backslash and every task runs the whole grid.
SHARD=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    COUNT="${SLURM_ARRAY_TASK_COUNT:-}"
    if [[ -z "${COUNT}" ]]; then
        COUNT=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
    fi
    SHARD=(
        "shard.index=$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN))"
        "shard.count=${COUNT}"
    )
    echo "shard: ${SHARD[*]}"
fi

python -m jumper_ablations.cli.run "${SHARD[@]}" "$@"
