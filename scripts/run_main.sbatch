#!/bin/bash

#SBATCH --gres=gpu:1 
#SBATCH --mem=16gb
#SBATCH --output out/main_output.log
#SBATCH -c 16 
#SBATCH --time=24:00:00
#SBATCH --partition=mia
#SBATCH --nodelist=ctit093
#SBATCH --error=out/skip_16.err

module load anaconda3/2022.05
source activate aaasegmentor
module load nvidia/cuda-11.7 

export PYTHONPATH="${PYTHONPATH}:/home/imreb/AAAsegmentor/"

cd /local
mkdir ${SLURM_JOBID}
cd ${SLURM_JOBID}
cp -r ${SLURM_SUBMIT_DIR}/src .



python3 src/main.py

rm -r /local/${SLURM_JOBID}
