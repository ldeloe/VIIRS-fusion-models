#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/compute_canada_output/%j.out
#SBATCH --account=def-ka3scott
#SBATCH --mail-user=lcbdeloe@uwaterloo.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module --force purge
module load StdEnv/2020  gcc/9.3.0 opencv/4.8.0
module load python/3.10.2

echo "Loading module done"
echo "test to see if change occurred"

source ~/wandbEnv/bin/activate

echo "Activating virtual environment done"

cd /home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/

echo "starting training..."

export WANDB_MODE=offline

python launch.py $1 --wandb-project=$2




