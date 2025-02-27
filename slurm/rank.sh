#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_MINA
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

kg_name=$1
kge_model_path=$2
kge_config_path=$3
output_path=$4

python -m grainsack.operations rank \
    --kg_name $kg_name \
    --kge_model_path $kge_model_path \
    --kge_config_path $kge_config_path \
    --output_path $output_path >> mylog.log 2>&1