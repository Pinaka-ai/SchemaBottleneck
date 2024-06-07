#!/bin/bash -l

# Make sure to edit:
# - model_name in config yml file in config_path to point to the ../model folder.
#   or if using checkpoint to t5large_openai_summ_QMQ3V
# - data file paths in rl4lms/data_pools/custom_text_generation_pools.py
# - prompt_path (x2) in config file to point to the ../prompts_edit_instructional_novo.txt
# - cache_path (x2) for storing generations in config file.

# Output path.
BASE_PATH="./morality_checkpoints-debanjan/"
MODEL_NAME="llama3-70B"
DIVERSITY_PENALTY="without_diversity_penalty"
# Project name and experiment name - used for wandb logging and output folder.
PROJECT_NAME="ppo_checkpoints_test"
CURRENT_TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
EXPERIMENT_NAME="t5large|n_envs=1|n_steps=50|n_iters=50|batch_size=8|priority_sampling=True|caching=False|$CURRENT_TIMESTAMP"

mkdir -p $BASE_PATH/$PROJECT_NAME/$MODEL_NAME/$DIVERSITY_PENALTY/$EXPERIMENT_NAME

rm data/ppo_generation.csv
module load cuda/12.2.1
module load cudnn/cuda11-8.4.1.50

python scripts/training/train_text_generation.py \
--base_path_to_store_results $BASE_PATH \
--config_path scripts/training/task_configs/openai_summ/t5_ppo_schema_bottleneck.yml \
--project_name $PROJECT_NAME \
--experiment_name $EXPERIMENT_NAME \
--entity_name pvashisht \
# --log_to_wandb
