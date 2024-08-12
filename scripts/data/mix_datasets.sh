#!/usr/bin/bash

save_path="./results/data/mixed"
seed="233"

python ./entrypoints/data/mix_datasets.py \
  --save_path ${save_path} \
  --seed ${seed}
