#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=1-576

config=./scripts/fs/fs_config_mixtral.txt

method=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $2}' $config)
mode=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $3}' $config)
dataset=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $4}' $config)
model=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $5}' $config)
summarization=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $6}' $config)
include_ranking=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $7}' $config)
include_examples=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $8}' $config)
parameters=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $9}' $config)
llm=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $10}' $config)

python -m src.evaluation.llm_forward_simulatability \
  --method $method \
  --mode $mode \
  --dataset $dataset \
  --model $model \
  --summarization $summarization \
  --include-ranking $include_ranking \
  --include-examples $include_examples \
  --parameters $parameters \
  --llm $llm \
  > "./logs/fs/${method}_${mode}_${dataset}_${model}_${summarization}_${include_ranking}_${include_examples}_${llm}_${parameters}B.log" 2>&1