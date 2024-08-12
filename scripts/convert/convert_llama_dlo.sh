#!/usr/bin/bash

# The default expansion setting in our paper
model_type="LlamaDLOExAttnForCausalLM" # LlamaDLOForCausalLM LlamaDLOExAttnForCausalLM
extend_method="interleave"             # interleave stack-side stack-manual
extend_init_method="zeros"             # random zeros normal merge-linear merge-slerp
original_layers=32
converted_layers=40
extended_use_dlo="True"   # whether to add DLO to the extended layers
original_dlo_frequency=1  # the frequency of adding DLO to the original layers
original_dlo_shrink_num=0 # the first & last `N` layers to exclude when adding DLO

# The default DLO hyperparameter in our paper
dlo_capacity=0.9
dlo_loss_coefficient=1.0
dlo_loss_type="cos-global" # self cos cos-global
rescale_hidden_states="True"
scale_factor=2.0
scale_gap=0.05
gate_init_method="zero" # zero random

# CHANGE THIS CORRESPONDINGLY
model_path="PATH_TO_LLAMA2_MODEL"    # 🔍
output_path="PATH_TO_SAVE_THE_MODEL" # 🔍

folder_name="llama-dlo-${extend_method}-${extend_init_method}${converted_layers}-${gate_init_method}G"
if [ ${original_dlo_frequency} -gt 0 ]; then
  folder_name="${folder_name}-freq${original_dlo_frequency}"
fi
if [ ${original_dlo_shrink_num} -gt 0 ]; then
  folder_name="${folder_name}-shrink${original_dlo_shrink_num}"
fi
if [ ${extended_use_dlo} = "False" ]; then
  folder_name="${folder_name}-NoExtended"
fi
output_path="${output_path}/${folder_name}" # update the output path by config

if [ ${extend_method} = "stack-manual" ]; then
  extend_method_args="[0,8];[6,12];[10,22];[20,26];[24,32]" # This only supports 32 -> 40 layers extension
  python ./entrypoints/convert/convert_llama_dlo.py \
    --model_path ${model_path} \
    --output_path ${output_path} \
    --model_type ${model_type} \
    --extend_method ${extend_method} \
    --extend_method_args ${extend_method_args} \
    --extend_init_method ${extend_init_method} \
    --original_layers ${original_layers} \
    --converted_layers ${converted_layers} \
    --extended_use_dlo ${extended_use_dlo} \
    --original_dlo_frequency ${original_dlo_frequency} \
    --original_dlo_shrink_num ${original_dlo_shrink_num} \
    --dlo_capacity ${dlo_capacity} \
    --dlo_loss_coefficient ${dlo_loss_coefficient} \
    --dlo_loss_type ${dlo_loss_type} \
    --rescale_hidden_states ${rescale_hidden_states} \
    --scale_factor ${scale_factor} \
    --scale_gap ${scale_gap} \
    --gate_init_method ${gate_init_method}
else
  python ./entrypoints/convert/convert_llama_dlo.py \
    --model_path ${model_path} \
    --output_path ${output_path} \
    --model_type ${model_type} \
    --extend_method ${extend_method} \
    --extend_init_method ${extend_init_method} \
    --original_layers ${original_layers} \
    --converted_layers ${converted_layers} \
    --extended_use_dlo ${extended_use_dlo} \
    --original_dlo_frequency ${original_dlo_frequency} \
    --original_dlo_shrink_num ${original_dlo_shrink_num} \
    --dlo_capacity ${dlo_capacity} \
    --dlo_loss_coefficient ${dlo_loss_coefficient} \
    --dlo_loss_type ${dlo_loss_type} \
    --rescale_hidden_states ${rescale_hidden_states} \
    --scale_factor ${scale_factor} \
    --scale_gap ${scale_gap} \
    --gate_init_method ${gate_init_method}
fi
