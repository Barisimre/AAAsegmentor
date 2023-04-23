#!/bin/bash

#SBATCH --gres=gpu:1 
#SBATCH --mem=16gb
#SBATCH --output out/main_output.log
#SBATCH -c 16 
#SBATCH --time=00:10:00
#SBATCH --partition=mia
#SBATCH --nodelist=ctit090
#SBATCH --error=out/benchmark.err

module load anaconda3/2022.05
source activate my_env
module load nvidia/cuda-11.7 

export PYTHONPATH="${PYTHONPATH}:/home/s1797743/thesis/final/AAAsegmentor/"

cd /local
mkdir ${SLURM_JOBID}
cd ${SLURM_JOBID}
cp -r ${SLURM_SUBMIT_DIR}/src .



python3 src/utils/benchmark.py > 90.txt

rm -r /local/${SLURM_JOBID}
