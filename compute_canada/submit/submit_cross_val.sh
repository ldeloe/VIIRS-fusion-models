#!/bin/bash 
#SBATCH --account=def-ka3scott

set -e

item="/home/lcbdeloe/projects/def-ka3scott/lcbdeloe/VIIRS-fusion-models/configs/opt_cross_val.py" 

for i in {1..6}; do
   sbatch w_net_cross_val.sh $item 
   echo "task successfully submitted" 
   sleep 10

done
