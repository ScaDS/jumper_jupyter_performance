#!/bin/bash
#SBATCH --job-name=ablations
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# One sweep of the ablation harness on the capella partition, one GPU each.
#
#   sbatch scripts/capella.sbatch suite=robotics protocol=pilot
#
# The GPU twin of romeo.sbatch, for the usecases that need CUDA or an EGL
# context. A task takes a quarter of a node, so four shards of an array share
# one; ask for --exclusive on the command line when timing noise from the
# neighbours matters more than the queue.
set -euo pipefail

VENV=/data/horse/ws/albu670g-jumper/kernels/jumper-capella

# Slurm copies the batch script to its own spool directory before running it,
# so the script's own location says nothing about where the experiment is.
# The directory sbatch was called from does, and ABLATIONS_ROOT overrides it.
EXPERIMENT="${ABLATIONS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${EXPERIMENT}/jumper_ablations" ]]; then
    echo "no experiment at '${EXPERIMENT}'; submit from its directory or" \
         "set ABLATIONS_ROOT" >&2
    exit 1
fi

# torch comes from the module, not from the venv: the venv is built with
# --system-site-packages on top of it and will not import torch without it.
module load release/2026 GCC/14.2.0 OpenMPI/5.0.7 \
    PyTorch/2.9.1-CUDA-12.8.0 FFmpeg/7.1.1
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

# A GPU usecase on a node whose GPU the job cannot see would silently fall
# back to CPU in half of the payloads and crash in the other half.
if ! nvidia-smi -L; then
    echo "no GPU visible on $(hostname)" >&2
    exit 1
fi

# Results go to the project filesystem, never to the home directory. The
# harness puts each pass's temporary files inside the run directory, so this
# one variable moves the whole lot.
DEFAULT_STORAGE=/data/horse/ws/albu670g-jumper/ablations
export JUMPER_ABLATION_STORAGE="${JUMPER_ABLATION_STORAGE:-$DEFAULT_STORAGE}"
mkdir -p "${JUMPER_ABLATION_STORAGE}"
echo "storage: ${JUMPER_ABLATION_STORAGE}"
df -h "${JUMPER_ABLATION_STORAGE}" | tail -1

# The default thread pools would size themselves to the node, not to the
# quarter of it this task holds, and add timing noise to every benchmark.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# MuJoCo renders offscreen; a compute node has no display to fall back to.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

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
