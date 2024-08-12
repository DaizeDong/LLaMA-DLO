#!/usr/bin/bash

num_nodes=1
num_processes=8

config_file="./configs/accelerate/deepspeed_llama.yaml"

# CHANGE THIS CORRESPONDINGLY
data_file="PATH_TO_THE_JSONL_FILE"              # 🔍
model_name_or_path="PATH_TO_NORMAL_LLAMA_MODEL" # 🔍
output_dir="PATH_TO_SAVE_THE_RESULTS"           # 🔍

BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / ${num_processes} / $BATCH_SIZE_PER_GPU))
echo "Training llama model using ${num_processes} GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
  --config_file ${config_file} \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  ./entrypoints/finetune/finetune_trainer.py \
  --model_name_or_path ${model_name_or_path} \
  --tokenizer_name ${model_name_or_path} \
  --use_fast_tokenizer False \
  --train_file ${data_file} \
  --max_seq_length 4096 \
  --do_train \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --learning_rate 2e-5 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --weight_decay 0. \
  --evaluation_strategy "no" \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --num_train_epochs 1 \
  --preprocessing_num_workers 64 \
  --use_flash_attn \
  --use_checkpointing \
  --output_dir ${output_dir} \
  --bf16 \
  --tf32 True \
  --report_to "tensorboard"
