#!/usr/bin/bash

# The default expansion setting in our paper
original_layers=32
converted_layers=40

# CHANGE THIS CORRESPONDINGLY
model_path="PATH_TO_LLAMA2_MODEL"    # 🔍
output_path="PATH_TO_SAVE_THE_MODEL" # 🔍
output_path="${output_path}llama-pro"

python ./entrypoints/convert/convert_llama_pro.py \
  --model_path ${model_path} \
  --output_path ${output_path} \
  --original_layers ${original_layers} \
  --converted_layers ${converted_layers}
