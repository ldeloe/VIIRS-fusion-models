#!/bin/bash
#SBATCH --account=def-ka3scott

set -e

#declare -a scripts=("/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/opt_adamw_base.py" 
declare -a scripts=("/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/opt_adamw_b1.py" 
                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/opt_adamw_b2.py") 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/opt_sgd_base.py")
#declare -a scripts=("/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/wr_5_x2.py"  
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/wr_10_x2.py")
#declare -a scripts=("/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/bs_4_300_2.py" 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/bs_8_300_2.py" 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/bs_16_300_2.py" 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/bs_32_300_2.py")
#declare -a scripts=("/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/lr_0005_2.py" 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/lr_01_2.py" 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/lr_001_2.py" 
#                    "/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/lr_0001_2.py")
for i in "${scripts[@]}"
do
    sbatch optims.sh $i
    #sbatch warm_restart.sh $i
    #sbatch batch_size.sh $i
    #sbatch learning_rate.sh $i
    echo "task successfully submitted"
    sleep 10

done