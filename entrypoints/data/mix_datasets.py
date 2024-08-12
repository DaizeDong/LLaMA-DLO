"""Mix the instruction tuning data by given portions"""

import argparse
import os
import random

from utils.io import find_files, load_jsonl, save_jsonl, create_dir
from utils.operations.operation_list import replicate_elements

DATASET_PATH = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 🔍

REFORMATTED_DATA_DIR = {
    "vicuna_sharegpt": os.path.join(DATASET_PATH, "vicuna_sharegpt"),
    "evol_instruct": os.path.join(DATASET_PATH, "evol_instruct"),
    "slim_orca": os.path.join(DATASET_PATH, "slim_orca"),
    "meta_math_qa": os.path.join(DATASET_PATH, "meta_math_qa"),
    "evol_code_alpaca": os.path.join(DATASET_PATH, "evol_code_alpaca"),
}

MIX_PORTION = {
    "vicuna_sharegpt": 2.9,
    "evol_instruct": 1.0,
    "slim_orca": 1.0,
    "meta_math_qa": 1.0,
    "evol_code_alpaca": 1.0,
}

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--save_path", type=str, default="./results/data/mixed")
    arg_parser.add_argument("--seed", type=int, default=233)
    args = arg_parser.parse_args()
    random.seed(args.seed)

    final_data_list = []

    for dataset_name in REFORMATTED_DATA_DIR.keys():
        dataset_path = REFORMATTED_DATA_DIR[dataset_name]
        dataset_portion = MIX_PORTION[dataset_name]

        for data_file in find_files(dataset_path, "*.jsonl"):
            data_list = load_jsonl(data_file)
            print(f"{dataset_name} {data_file}: original length {len(data_list)}, portion {dataset_portion}")
            data_list = replicate_elements(data_list, dataset_portion)
            print(f"{dataset_name} {data_file}: replicated length {len(data_list)}")
            final_data_list.extend(data_list)

    print(f"Shuffling final data list...")
    random.shuffle(final_data_list)
    print(f"final mixed data list length: {len(final_data_list)}")

    create_dir(args.save_path)
    save_file = os.path.join(args.save_path, "data.jsonl")
    save_jsonl(final_data_list, save_file)
    print("Done.")
