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
EXPERIMENT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

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

# Benchmark replays mkdtemp per measurement and do not clean up, so they are
# pointed at the workspace filesystem rather than at the node's small /tmp.
export TMPDIR="${EXPERIMENT}/workspace/tmp"
mkdir -p "${TMPDIR}"

# minian's dask arrays are chunked for one machine; the default thread pool
# would oversubscribe the allocation and add timing noise to every benchmark.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -m jumper_ablations.cli.run "$@"
