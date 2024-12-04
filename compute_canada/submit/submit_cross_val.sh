#!/bin/bash 
#SBATCH --account=def-ka3scott

set -e

item="/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/unet_ai4arctic.py" 
wandb_project=ai4arctic-unet-cross-validation
for i in {1..6}; do
   #sbatch w_net_cross_val.sh $item 
   sbatch cross_validation.sh $item $wandb_project
   echo "task successfully submitted" 
   sleep 10

done

#item="/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/opt_cross_val.py" 
#wandb_project=dual-encoder-unet-cross-validation
#for i in {1..6}; do
   #sbatch w_net_cross_val.sh $item 
#   sbatch cross_validation.sh $item $wandb_project
#   echo "task successfully submitted" 
#   sleep 10

#done
