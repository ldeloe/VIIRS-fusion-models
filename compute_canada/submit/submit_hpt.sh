#!/bin/bash
#SBATCH --account=def-ka3scott

set -e

#declare -a scripts=("/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/bs_4_300.py" 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/bs_8_300.py" 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/bs_16_300.py" 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/bs_32_300.py")
declare -a scripts=("/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/lr_1.py" 
                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/lr_01.py" 
                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/lr_001.py" 
                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/lr_0001.py")
for i in "${scripts[@]}"
do
    #sbatch batch_size.sh $i
    sbatch learning_rate.sh $i
    echo "task successfully submitted"
    sleep 10

done