#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_MINA
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

predictions_path=$1
kg_name=$2
kge_model_path=$3
kge_config_path=$4
lpx_config=$5
output_path=$6

python -m grainsack.operations explain \
    --predictions_path $predictions_path \
    --kg_name $kg_name \
    --kge_model_path $kge_model_path \
    --kge_config_path $kge_config_path \
    --lpx_config "$lpx_config" \
    --output_path $output_path >> mylog.log 2>&1