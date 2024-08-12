import argparse
import warnings
from typing import Union

import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from models.llama_dlo.configuration_llama_dlo import LlamaDLOConfig
from models.llama_dlo.configuration_llama_dlo_exattn import LlamaDLOExAttnConfig
from models.llama_dlo.modeling_llama_dlo import LlamaDLOForCausalLM
from models.llama_dlo.modeling_llama_dlo_exattn import LlamaDLOExAttnForCausalLM
from utils.io import create_dir
from utils.operations.operation_others import auto_convert_args_to_bool
from utils.operations.operation_string import extract_numbers

ALL_MODEL_TYPES = ("LlamaDLOForCausalLM", "LlamaDLOExAttnForCausalLM", "MistralDLOExAttnForCausalLM",)
ALL_EXTEND_METHODS = ("interleave", "stack-side", "stack-manual")
ALL_INIT_METHODS = ("random", "zeros", "normal", "merge-linear", "merge-slerp")


def normalize(v: np.ndarray, eps: float):
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v


def lerp(
        t: float, v0: Union[np.ndarray, torch.Tensor], v1: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    return (1 - t) * v0 + t * v1


def slerp(
        t: Union[float, np.ndarray],
        v0: Union[np.ndarray, torch.Tensor],
        v1: Union[np.ndarray, torch.Tensor],
        DOT_THRESHOLD: float = 0.9995,
        eps: float = 1e-8,
):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """
    if not isinstance(v0, np.ndarray):
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        v1 = v1.detach().cpu().float().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles
    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)

    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return torch.from_numpy(res)

    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return torch.from_numpy(res)


def main(args):
    # Load model
    model = LlamaForCausalLM.from_pretrained(args.model_path)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)

    # Convert state dict
    original_state_dict = model.state_dict()
    converted_state_dict = {}
    is_dlo = []

    if args.extend_method == "interleave":
        # Extend layers sequentially, following LLaMA-Pro
        # https://arxiv.org/abs/2401.02415
        if args.converted_layers > args.original_layers:
            split = int(args.original_layers / (args.converted_layers - args.original_layers))
        else:
            split = 99999999999  # set to a large number so that no expansion is made
        print("split:", split)

        layer_cnt = 0
        for i in range(args.original_layers):
            for k in original_state_dict:
                if ('layers.' + str(i) + '.') in k:
                    converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
            if args.original_dlo_frequency > 0 and (i + 1) % args.original_dlo_frequency == 0:
                if i >= args.original_dlo_shrink_num and i < args.original_layers - args.original_dlo_shrink_num:
                    is_dlo.append(True)
                else:
                    is_dlo.append(False)
            else:
                is_dlo.append(False)
            layer_cnt += 1

            if (i + 1) % split == 0:
                print(f"new layer {layer_cnt} is a copy of original layer {i}")
                for k in original_state_dict:
                    if ('layers.' + str(i) + '.') in k:
                        if args.extend_init_method == "random":
                            pass
                        elif args.extend_init_method == "zeros":
                            if 'down_proj' in k or 'o_proj' in k:
                                converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = torch.zeros_like(original_state_dict[k]).bfloat16()
                            else:
                                converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
                        elif args.extend_init_method == "normal":
                            converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
                        elif args.extend_init_method == "merge-linear":
                            if i >= 1:
                                converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = torch.mean(torch.stack((
                                    original_state_dict[k].bfloat16(),
                                    original_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(i - 1) + '.'))].bfloat16(),
                                ), dim=0), dim=0)
                            else:  # use normal
                                converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
                        elif args.extend_init_method == "merge-slerp":
                            if i >= 1:
                                converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = slerp(
                                    0.5,
                                    original_state_dict[k].bfloat16(),
                                    original_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(i - 1) + '.'))].bfloat16(),
                                )
                            else:  # use normal
                                converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
                        else:
                            raise NotImplementedError
                if args.extended_use_dlo:
                    is_dlo.append(True)
                else:
                    is_dlo.append(False)
                layer_cnt += 1

        assert (layer_cnt == args.converted_layers)
        for k in original_state_dict:
            if not 'layers' in k:
                converted_state_dict[k] = original_state_dict[k].bfloat16()

    elif args.extend_method == "stack-side":
        # Extend layers by concatenating the former & latter half layers, following SOLAR
        # https://arxiv.org/abs/2312.15166
        overlapped_layer_cnt = args.converted_layers - args.original_layers
        assert overlapped_layer_cnt <= args.original_layers
        assert overlapped_layer_cnt % 2 == 0
        each_side_layer_cnt = (args.converted_layers - overlapped_layer_cnt * 2) // 2

        copied_layer_ids = []
        init_methods = []
        is_dlo = []
        for i in range(0, each_side_layer_cnt):
            copied_layer_ids.append(i)
            init_methods.append("normal")
            is_dlo.append((args.original_dlo_frequency > 0 and (i + 1) % args.original_dlo_frequency == 0) and (i >= args.original_dlo_shrink_num and i < args.original_layers - args.original_dlo_shrink_num))
        for i in range(each_side_layer_cnt, each_side_layer_cnt + overlapped_layer_cnt):
            copied_layer_ids.append(i)
            init_methods.append("normal")
            is_dlo.append((args.original_dlo_frequency > 0 and (i + 1) % args.original_dlo_frequency == 0) and (i >= args.original_dlo_shrink_num and i < args.original_layers - args.original_dlo_shrink_num))
        for i in range(each_side_layer_cnt, each_side_layer_cnt + overlapped_layer_cnt):  # Here the extended part
            copied_layer_ids.append(i)
            init_methods.append(args.extend_init_method)
            is_dlo.append(args.extended_use_dlo)
        for i in range(each_side_layer_cnt + overlapped_layer_cnt, args.original_layers):
            copied_layer_ids.append(i)
            init_methods.append("normal")
            is_dlo.append((args.original_dlo_frequency > 0 and (i + 1) % args.original_dlo_frequency == 0) and (i >= args.original_dlo_shrink_num and i < args.original_layers - args.original_dlo_shrink_num))
        print("copied_layer_ids", copied_layer_ids)

        layer_cnt = 0
        for i in range(len(copied_layer_ids)):
            layer_id = copied_layer_ids[i]
            init_method = init_methods[i]
            for k in original_state_dict:
                if ('layers.' + str(layer_id) + '.') in k:
                    if init_method == "zeros":
                        if 'down_proj' in k or 'o_proj' in k:
                            converted_state_dict[k.replace(('layers.' + str(layer_id) + '.'), ('layers.' + str(layer_cnt) + '.'))] = torch.zeros_like(original_state_dict[k]).bfloat16()
                        else:
                            converted_state_dict[k.replace(('layers.' + str(layer_id) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
                    elif init_method == "normal":
                        converted_state_dict[k.replace(('layers.' + str(layer_id) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
                    elif args.extend_init_method == "merge-linear":
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
                    layer_cnt += 1
                    break

        assert (layer_cnt == args.converted_layers)
        for k in original_state_dict:
            if not 'layers' in k:
                converted_state_dict[k] = original_state_dict[k].bfloat16()

    elif args.extend_method == "stack-manual":
        # Specify layers ids by hand to form the final layers for the extended model
        if args.extend_init_method != "normal":
            warnings.warn(f"Changing \"extend_init_method\" {args.extend_init_method} to \"normal\", "
                          f"which is the only supported method for stack-manual.")
            args.extend_init_method = "normal"

        if args.extend_method_args is None:
            raise ValueError("extend_method_args is None")
        stack_layer_ranges = [extract_numbers(string) for string in args.extend_method_args.split(';')]
        assert sum([this_range[1] - this_range[0] for this_range in stack_layer_ranges]) == args.converted_layers
        print("stack_layer_ranges", stack_layer_ranges)

        is_dlo = []

        layer_cnt = 0
        for stack_range in stack_layer_ranges:
            for i in range(stack_range[0], stack_range[1]):
                for k in original_state_dict:
                    if ('layers.' + str(i) + '.') in k:
                        converted_state_dict[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = original_state_dict[k].bfloat16()
                        if args.original_dlo_frequency > 0 and (layer_cnt + 1) % args.original_dlo_frequency == 0:
                            if layer_cnt >= args.original_dlo_shrink_num and layer_cnt < args.converted_layers - args.original_dlo_shrink_num:  # Note that here is the "converted_layers"
                                is_dlo.append(True)
                            else:
                                is_dlo.append(False)
                        else:
                            is_dlo.append(False)
                        layer_cnt += 1
                        break

        assert (layer_cnt == args.converted_layers)
        for k in original_state_dict:
            if not 'layers' in k:
                converted_state_dict[k] = original_state_dict[k].bfloat16()
    else:
        raise NotImplementedError

    # Save model
    print("Saving...")
    create_dir(args.output_path)

    if args.model_type == "LlamaDLOForCausalLM":
        converted_config = LlamaDLOConfig.from_llama_config(
            model.config,
            is_dlo=is_dlo,
            dlo_capacity=args.dlo_capacity,
            dlo_loss_coefficient=args.dlo_loss_coefficient,
            dlo_loss_type=args.dlo_loss_type,
            rescale_hidden_states=args.rescale_hidden_states,
            scale_factor=args.scale_factor,
            scale_gap=args.scale_gap,
            gate_init_method=args.gate_init_method,
        )
        converted_config.num_hidden_layers = args.converted_layers
        print(converted_config)
        convert_model = LlamaDLOForCausalLM(converted_config)

    elif args.model_type == "LlamaDLOExAttnForCausalLM":
        converted_config = LlamaDLOExAttnConfig.from_llama_config(
            model.config,
            is_dlo=is_dlo,
            dlo_capacity=args.dlo_capacity,
            dlo_loss_coefficient=args.dlo_loss_coefficient,
            dlo_loss_type=args.dlo_loss_type,
            rescale_hidden_states=args.rescale_hidden_states,
            scale_factor=args.scale_factor,
            scale_gap=args.scale_gap,
            gate_init_method=args.gate_init_method,
        )
        converted_config.num_hidden_layers = args.converted_layers
        print(converted_config)
        convert_model = LlamaDLOExAttnForCausalLM(converted_config)

    else:
        raise NotImplementedError

    convert_model.bfloat16()
    convert_model.load_state_dict(converted_state_dict, strict=False)
    convert_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(convert_model)
    print(f"Model saved to {args.output_path}")
    print("Done.")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_path", default='meta-llama/Llama-2-7b-hf', type=str, help="original model path")
    parser.add_argument("--output_path", default='./results/ckpt/llama-dlo', type=str, help="deepened model ckpt save path")
    parser.add_argument("--model_type", default='LlamaDLOForCausalLM', type=str, choices=ALL_MODEL_TYPES, help="converted model type")
    parser.add_argument("--extend_method", default='interleave', type=str, choices=ALL_EXTEND_METHODS, help="method to upcycle the layers")
    parser.add_argument("--extend_method_args", default=None, type=str, help="used for \"extend_method\"")
    parser.add_argument("--extend_init_method", default='zeros', type=str, choices=ALL_INIT_METHODS, help="method to initialize the upcycled the layers")
    parser.add_argument("--original_layers", default=32, type=int, help="original model num layers")
    parser.add_argument("--converted_layers", default=40, type=int, help="deepen model num layers")
    parser.add_argument("--extended_use_dlo", default="True", type=str, help="whether to add DLO to extended layers")
    parser.add_argument("--original_dlo_frequency", default=0, type=int, help="the frequency of DLO on original layers")
    parser.add_argument("--original_dlo_shrink_num", default=0, type=int, help="the number of layers on both sides to avoid when applying DLO")
    # For DLO Configuration
    parser.add_argument("--dlo_capacity", default=0.5, type=float)
    parser.add_argument("--dlo_loss_coefficient", default=0.01, type=float)
    parser.add_argument("--dlo_loss_type", default="self", type=str)  # self cos cos-global
    parser.add_argument("--rescale_hidden_states", default="True", type=str)
    parser.add_argument("--scale_factor", default=1.0, type=float)
    parser.add_argument("--scale_gap", default=1.0, type=float)
    parser.add_argument("--gate_init_method", default="zero", type=str)  # zero random
    args = parser.parse_args()
    args = auto_convert_args_to_bool(args)
    main(args)
