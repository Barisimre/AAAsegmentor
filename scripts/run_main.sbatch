#!/bin/bash

#SBATCH --gres=gpu:1 
#SBATCH --mem=32gb
#SBATCH --output out/main_output.log
#SBATCH -c 32 
#SBATCH --time=10:00:00
#SBATCH --partition=mia
#SBATCH --nodelist=ctit094
#SBATCH --error=out/err.err

module load anaconda3/2022.05
source activate my_env
module load nvidia/cuda-11.5 

export PYTHONPATH="${PYTHONPATH}:/home/s1797743/thesis/final/AAAsegmentor/"

cd /local
mkdir ${SLURM_JOBID}
cd ${SLURM_JOBID}
cp -r ${SLURM_SUBMIT_DIR}/src .



python3 src/main.py

rm -r /local/${SLURM_JOBID}
