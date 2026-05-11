#!/bin/bash
# Single-GPU CD-RLHF (curiosity-driven PPO) for Qwen2.5-0.5B on TL;DR.
# Differs from the vanilla RLHF script only in --eta (the curiosity bonus weight)
# and --offload_icm_model (ICM module added to the 4 base copies; offload to keep
# VRAM in budget).
#
# eta=0.04 follows the gemma-2b CD-RLHF setting from the paper. Tune in [0.01, 0.1].
# cdrlhf_topk=1 means curiosity is added to the top-1 (greedy) sampled token.

OUTPUT=./models/qwen2.5-0.5b-tldr-cdrlhf
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
   --offload_reference_model \
   --offload_icm_model \
   --dtype bf16 \
   --actor_lora_dim 16 \
   --actor_lora_module_name "model.layers." \
   --critic_lora_dim 16 \
   --critic_lora_module_name "rwtransformer.layers." \
   --only_optimize_lora \
   --output_dir $OUTPUT \
   --icm_learning_rate 1e-5 \
   --eta 0.04 \
   --cdrlhf_topk 1 \
   --sample_size 1000 \
   --kl_ctl 0.05 \
   --print_answers \
   --print_answers_interval 100 \
   --save_steps 1000 \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
    &>> $OUTPUT/training.log
