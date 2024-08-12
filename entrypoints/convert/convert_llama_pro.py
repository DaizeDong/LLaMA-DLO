import argparse
from copy import deepcopy

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.io import create_dir


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_path", default='meta-llama/Llama-2-7b-hf', type=str, help="original model path")
    parser.add_argument("--output_path", default='./results/ckpt/llama-pro', type=str, help="deepened model ckpt save path")
    parser.add_argument("--original_layers", default=32, type=int, help="original model num layers")
    parser.add_argument("--converted_layers", default=40, type=int, help="deepen model num layers")
    args = parser.parse_args()

    # Load model
    model = LlamaForCausalLM.from_pretrained(args.model_path)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)

    # Convert state dict
    original_state_dict = model.state_dict()
    converted_state_dict = {}
    layer_cnt = 0
    split = int(args.original_layers / (args.converted_layers - args.original_layers))
    print("split:", split)

    for i in range(args.original_layers):
        for k in original_state_dict:
            if ('layers.' + str(i) + '.') in k:
                converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
        layer_cnt += 1
        if (i + 1) % split == 0:
            print(f"layer {layer_cnt} is a copy of layer {i}")
            for k in original_state_dict:
                if ('layers.' + str(i) + '.') in k:
                    if 'down_proj' in k or 'o_proj' in k:
                        converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = torch.zeros_like(original_state_dict[k]).bfloat16()
                    else:
                        converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
            layer_cnt += 1

    assert (layer_cnt == args.converted_layers)
    for k in original_state_dict:
        if not 'layers' in k:
            converted_state_dict[k] = original_state_dict[k].bfloat16()

    # Save model
    print("Saving...")
    converted_config = deepcopy(model.config)
    converted_config.num_hidden_layers = args.converted_layers
    print(converted_config)

    create_dir(args.output_path)
    convert_model = LlamaForCausalLM(converted_config)
    convert_model.bfloat16()
    convert_model.load_state_dict(converted_state_dict)
    convert_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(convert_model)
    print("Done.")


if __name__ == "__main__":
    main()
