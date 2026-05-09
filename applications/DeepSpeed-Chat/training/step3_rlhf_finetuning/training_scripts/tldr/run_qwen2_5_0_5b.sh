#!/bin/bash
# Single-GPU vanilla RLHF (PPO) for Qwen2.5-0.5B on TL;DR.
# eta=0.0 disables the curiosity bonus -- this is the baseline against which
# CD-RLHF is compared. Use run_qwen2_5_0_5b_cdrlhf.sh for the CD-RLHF variant.
#
# Memory footprint on 32GB VRAM (fp16, ZeRO-2, 0.5B model x4 copies):
#   actor (train) + critic (train) + ref (frozen, CPU-offloaded) + reward (frozen)
#   Optimizer state for actor+critic offloaded to CPU via --offload.
#   Sequence lengths reduced to 256+256 (from 512+512) to cut KV-cache 4×.
#
# Effective batch size = per_device_train * num_gpus * grad_accum = 1 * 1 * 64 = 64
# (smaller than the paper's 256 to keep wall-clock manageable on a single GPU;
# raise grad_accum to 256 if you want to match the paper exactly).

OUTPUT=./models/qwen2.5-0.5b-tldr-rlhf
mkdir -p $OUTPUT

echo $(basename $OUTPUT)
branch_info=$(git branch | grep '*')
commit_info=$(git rev-parse --short HEAD)
echo "branch: $branch_info commit id: $commit_info" > $OUTPUT/training.log

Actor_Lr=1e-5
Critic_Lr=1.5e-5

deepspeed --num_gpus 1 main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --actor_model_name_or_path ../step1_supervised_finetuning/models/qwen2.5-0.5b-tldr-sft \
   --critic_model_name_or_path ../step2_reward_model_finetuning/models/qwen2.5-0.5b-tldr-rm \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type linear \
   --gradient_accumulation_steps 64 \
   --actor_dropout 0.0 \
   --warmup_ratio 0.05 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 2 \
   --critic_zero_stage 2 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload \
   --offload_reference_model \
   --dtype fp16 \
   --output_dir $OUTPUT \
   --icm_learning_rate 1e-5 \
   --eta 0.0 \
   --cdrlhf_topk 1 \
   --sample_size 1000 \
   --kl_ctl 0.05 \
   --print_answers \
   --print_answers_interval 100 \
   --save_steps 1000 \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
    &>> $OUTPUT/training.log
