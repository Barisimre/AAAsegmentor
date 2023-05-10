#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output out/jupyter-notebook.log
#SBATCH -c 16
#SBATCH --partition=mia
#SBATCH --nodelist=ctit093


module load anaconda3/2022.05
source activate final
module load nvidia/cuda-11.7 

export PYTHONPATH="${PYTHONPATH}:/home/s1797743/thesis/final/AAAsegmentor/"

# Warning !
# Do not modify the sbatch script bolow this line !

#clean up XDG_RUNTIME_DIR variable
export XDG_RUNTIME_DIR=""

#Log Node Name
NODE=$(hostname)

# generate random port 8800-8809
PORT=$(((RANDOM % 10)+8800))

# start the notebook
jupyter lab --collaborative --no-browser --ip=$NODE.ewi.utwente.nl --port=$PORT
# ~/.local/bin/jupyter-lab --collaborative --no-browser --ip=$NODE.ewi.utwente.nl --port=$PORT
