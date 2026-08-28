#!/bin/bash
#SBATCH --job-name=hf_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL

# Environment setup
source ~/.bashrc
conda activate torch

export HF_HOME=/path/to/huggingface/cache
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub

echo "Running on $(hostname)"
echo "Job started at $(date)"

# Run evaluation script (example: benign generation)
python3 ./utility/run_llmjudge_coherence.py \
  --model_id unsloth/Llama-3.3-70B-Instruct-bnb-4bit \
  --batch_size 64 \
  --max_new_tokens 8 \
  --temperature 0.0

echo "Job finished at $(date)"
