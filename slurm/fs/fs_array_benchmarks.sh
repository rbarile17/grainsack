#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_DIXTI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --array=1-3

config=./scripts/fs/fs_config_benchmarks_mixtral.txt

dataset=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $2}' $config)
model=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $3}' $config)
include_ranking=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $4}' $config)
include_examples=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $5}' $config)
parameters=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $6}' $config)
llm=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $7}' $config)

python -m src.evaluation.llm_forward_simulatability \
  --method benchmark \
  --dataset $dataset \
  --model $model \
  --include-ranking $include_ranking \
  --include-examples $include_examples \
  --parameters $parameters \
  --llm $llm \
  > "./logs/fs_benchmarks/benchmark_None_${dataset}_${model}_None_${include_ranking}_${include_examples}_${llm}_${parameters}B.log" 2>&1
