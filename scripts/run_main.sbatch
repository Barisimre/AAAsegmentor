#!/bin/bash

#SBATCH --gres=gpu:1 
#SBATCH --mem=32gb
#SBATCH --output main_output.log
#SBATCH -c 32 
#SBATCH --time=10:00:00
#SBATCH --partition=mia
#SBATCH --nodelist=ctit094
#SBATCH --error=err.err

module load anaconda3/2022.05
source activate my_env
module load nvidia/cuda-11.5 



cd /local
mkdir ${SLURM_JOBID}
cd ${SLURM_JOBID}
cp ${SLURM_SUBMIT_DIR}/src .

python3 unet3.py

rm -r /local/${SLURM_JOBID}