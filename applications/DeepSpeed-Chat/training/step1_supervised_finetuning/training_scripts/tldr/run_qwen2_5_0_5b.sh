#!/bin/bash
# Single-GPU SFT for Qwen2.5-0.5B on the TL;DR dataset.
# Target: 32GB VRAM box (e.g. RTX 5090, V100 32GB, L4 24GB also fits).
# Effective batch size = per_device * num_gpus * grad_accum = 8 * 1 * 32 = 256
# (matches the paper's effective batch on the 8-GPU Llama-1B SFT script).

OUTPUT=./models/qwen2.5-0.5b-tldr-sft
mkdir -p $OUTPUT

export CUDA_VISIBLE_DEVICES=0

deepspeed --num_gpus 1 main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --model_name_or_path Qwen/Qwen2.5-0.5B \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
   --learning_rate 1e-4 \
   --weight_decay 0.01 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 64 \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.1 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 2 \
   --deepspeed \
   --dtype fp16 \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
   --print_loss \
   &> $OUTPUT/training.log
