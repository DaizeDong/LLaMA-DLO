#!/usr/bin/bash

save_path="./results/data/reformatted"

python ./entrypoints/data/reformat_datasets.py \
  --save_path ${save_path}
