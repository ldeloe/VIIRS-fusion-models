#!/bin/bash 
#SBATCH --account=def-ka3scott

set -e

#item="/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/unet_ai4arctic.py" 
#wandb_project=ai4arctic-unet-cross-validation

#item1="/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/dual_encoder_unet_0.py" 
#item2="/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/dual_encoder_unet_60_actual.py"
#wandb_project=dual-encoder-unet-cross-validation

item1="/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/viirs_sar_dual_encoder_bs_16.py"
wandb_project=dual-encoder-separate-viirs-cross-validation
for i in {1..5}; do
   #sbatch w_net_cross_val.sh $item 
   sbatch cross_validation.sh $item1 $wandb_project
   echo "task1 successfully submitted" 
   sleep 10

   #sbatch cross_validation.sh $item2 $wandb_project
   #echo "task2 successfully submitted" 
   #sleep 10
done

#item="/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/opt_cross_val.py" 
#wandb_project=dual-encoder-unet-cross-validation
#for i in {1..6}; do
   #sbatch w_net_cross_val.sh $item 
#   sbatch cross_validation.sh $item $wandb_project
#   echo "task successfully submitted" 
#   sleep 10

#done
