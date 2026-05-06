#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --job-name=jumper_multinode_srun_test
#SBATCH --output=jumper_multinode_srun_test_%j.out
#SBATCH --error=jumper_multinode_srun_test_%j.err

# Load any required modules
# module load python/3.9

# Set environment variables for proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the multinode srun example
echo "Starting multinode JUmPER monitor example (srun-based)..."
python multinode_srun_example.py

echo "Job completed"
