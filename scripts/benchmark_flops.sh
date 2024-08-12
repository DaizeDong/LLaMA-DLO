#!/usr/bin/bash

model_name_or_path="PATH_TO_THE_MODEL"         # 🔍
save_file="PATH_TO_SAVE_THE_RESULTS/flops.txt" # 🔍

python ./entrypoints/benchmark_flops.py \
  --model_name_or_path ${model_name_or_path} \
  --save_file ${save_file}
