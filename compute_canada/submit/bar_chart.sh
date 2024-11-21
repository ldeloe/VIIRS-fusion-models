#!/bin/bash
#SBATCH --time=0:20:00
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

source ~/wandbEnv/bin/activate

echo "Activating virtual environment done"

cd /home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/

echo "starting..."

python train_class_percentage.py



