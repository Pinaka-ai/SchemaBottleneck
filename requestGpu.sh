#!/bin/bash

#SBATCH --job-name=ppo|n_envs=2|n_steps=512|num_scenarios=5|n_iters=18
#SBATCH --gres=gpu:1
#SBATCH --partition=gypsum-rtx8000
#SBATCH --mem=40G
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH --mail-user=pvashisht@umass.edu
#SBATCH --output=/work/pi_dhruveshpate_umass_edu/project_3/SchemaBottleneckPPO/sbatch_logs/out/%x_%J.out
#SBATCH --error=/work/pi_dhruveshpate_umass_edu/project_3/SchemaBottleneckPPO/sbatch_logs/error/%x_%J.err

BASE_PATH="./morality_checkpoints-debanjan"

PROJECT_NAME="ppo_checkpoints"
EXPERIMENT_NAME="ppo|n_envs=2|n_steps=512|num_scenarios=5|n_iters=18"

mkdir -p $BASE_PATH/$PROJECT_NAME/$EXPERIMENT_NAME

module load miniconda/22.11.1-1
conda activate parht_env

rm data/ppo_generation.csv
module load cuda/12.2.1
module load cudnn/cuda11-8.4.1.50

cd /work/pi_dhruveshpate_umass_edu/project_3/SchemaBottleneckPPO/rl4f

python scripts/training/train_text_generation.py \
--base_path_to_store_results $BASE_PATH \
--config_path scripts/training/task_configs/openai_summ/t5_ppo_schema_bottleneck.yml \
--project_name $PROJECT_NAME \
--experiment_name $EXPERIMENT_NAME \
--entity_name pvashisht \
--log_to_wandb 