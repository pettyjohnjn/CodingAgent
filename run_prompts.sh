#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/baseline_paper_generation.err
#SBATCH --output=logs/baseline_paper_generation.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=baseline_paper_generation
#SBATCH --mem-per-gpu=80000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=general
#SBATCH --time=700

# activate env
cd /home/rachelgordon/assignmnt-3-battleship-rachelngordon || exit 1
source venv/bin/activate

# Move to CodingAgent directory (sibling of assignmnt-3-battleship-rachelngordon)
cd ../CodingAgent || exit 1

export GITHUB_TOKEN="${GITHUB_TOKEN:?GITHUB_TOKEN not set}"

srun python run_prompts.py
