#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --account=IscrC_MINA
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

python -m grainsack.operations comparison