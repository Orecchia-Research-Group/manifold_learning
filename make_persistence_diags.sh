#!/bin/bash
#SBATCH --job-name=persistence_gen          # Job name
#SBATCH --mail-type=ALL                     # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=robinett@uchicago.edu   # Where to send mail
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --mem=8gb                           # Job Memory
#SBATCH --output=array_%A-%a.log            # Standard output and error log
#SBATCH --array=5-30                        # Array range

python3 generate_persistence_diags.py $SLURM_ARRAY_TASK_ID
