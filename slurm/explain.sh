#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

method=$1
mode=$2
kg=$3
model=$4
summarization=$5

python -m src.explain --method $method --mode $mode --dataset $dataset --model $model --summarization $summarization
