#!/bin/bash

#SBATCH --gres=gpu:1 
#SBATCH --mem=16gb
#SBATCH --output out/main_output.log
#SBATCH -c 16 
#SBATCH --time=6:00:00
#SBATCH --partition=mia
#SBATCH --nodelist=ctit093
#SBATCH --error=out/test.err

python src/test.py