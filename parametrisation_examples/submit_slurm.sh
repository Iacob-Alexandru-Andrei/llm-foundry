# Take an input file and submit it to SLURM with the specified parameters.
#!/bin/bash
#SBATCH --job-name=parametrisation_example
#SBATCH --output=parametrisation_example.out
#SBATCH --error=parametrisation_example.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta


# Launch script passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <script_to_run>"
    exit 1
fi
SCRIPT_TO_RUN=$1
if [ ! -f "$SCRIPT_TO_RUN" ]; then
    echo "Script $SCRIPT_TO_RUN does not exist."
    exit 1
fi
# Launch script with the specified parameters
srun "$SCRIPT_TO_RUN" \