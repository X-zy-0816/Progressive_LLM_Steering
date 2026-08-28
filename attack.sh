#!/bin/bash
#SBATCH --job-name=hf_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:2


# Environment setup
source ~/.bashrc
conda activate torch

export HF_HOME=/path/to/huggingface/cache
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub

echo "Running on $(hostname)"
echo "Job started at $(date)"

# Run steering attack evaluation
python utility/run_eval_transition.py \
  --model_id unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit \
  --vectors_dir featureDirections/deepseek_vectors \
  --layer_idx 15 \
  --max_new_tokens 2048 \
  --temperature 0 \
  --system "" \
  --batch_size 32 \
  --max_workers 4 \


echo "Job finished at $(date)"
