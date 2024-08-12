#!/usr/bin/bash

num_nodes=1
num_processes=8

# DLO configurations
# This will override the config.json file in the model path
dlo_capacity=0.9
dlo_loss_coefficient=1.0
dlo_loss_type="cos-global" # self cos cos-global
rescale_hidden_states=True # whether to scale the layer outputs using the gating scores
scale_factor=2.0
scale_gap=0.05
dlo_capacity_annealing_steps=1000 # -1 denotes no annealing
dynamic_lr="True"                 # whether to use the dynamic learning rate strategy for different layers

config_file="./configs/accelerate/deepspeed_llama.yaml"

# CHANGE THIS CORRESPONDINGLY
data_file="PATH_TO_THE_JSONL_FILE"           # 🔍
model_name_or_path="PATH_TO_LLAMA_DLO_MODEL" # 🔍
output_dir="PATH_TO_SAVE_THE_RESULTS"        # 🔍

folder_name="cap${dlo_capacity}-${dlo_loss_type}${dlo_loss_coefficient}"
if [ ${rescale_hidden_states} = "True" ]; then
  folder_name="${folder_name}-Scale${scale_factor}+${scale_gap}"
else
  folder_name="${folder_name}-NoScale"
fi
if [ ${dlo_capacity_annealing_steps} != "-1" ]; then
  folder_name="${folder_name}-Anneal${dlo_capacity_annealing_steps}"
fi
if [ ${dynamic_lr} = "True" ]; then
  folder_name="${folder_name}-DyLr"
fi
output_dir="${output_dir}/${folder_name}"

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
  --report_to "tensorboard" \
  --dlo_capacity ${dlo_capacity} \
  --dlo_loss_coefficient ${dlo_loss_coefficient} \
  --dlo_loss_type ${dlo_loss_type} \
  --rescale_hidden_states ${rescale_hidden_states} \
  --scale_factor ${scale_factor} \
  --scale_gap ${scale_gap} \
  --dlo_capacity_annealing_steps ${dlo_capacity_annealing_steps} \
  --dynamic_lr ${dynamic_lr}
