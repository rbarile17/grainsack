#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_MINA
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=64G

explanations_path=$1
kg_name=$2
output_path=$3
log_path=$4

python -m grainsack.operations evaluate \
    --explanations_path $explanations_path \
    --kg_name $kg_name \
    --output_path $output_path >> $log_path 2>&1