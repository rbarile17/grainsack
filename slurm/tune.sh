#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_MINA
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=64G

kg_name=$1
kge_model_name=$2
output_path=$3
log_path=$4

python -m grainsack.operations tune \
    --kg_name $kg_name \
    --kge_model_name $kge_model_name \
    --output_path $output_path >> $log_path 2>&1