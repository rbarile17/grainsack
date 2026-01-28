#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_MINA
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --mem=64G

explanations_path=$1
kg_name=$2
kge_model_path=$3
kge_config_path=$4
output_path=$5
log_path=$6


python -u -m grainsack.operations evaluate \
    --explanations_path $explanations_path \
    --kg_name $kg_name \
    --kge_model_path $kge_model_path \
    --kge_config_path $kge_config_path \
    --output_path $output_path >> $log_path 2>&1