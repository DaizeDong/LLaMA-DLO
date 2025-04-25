# DLO: Dynamic Layer Operation for Efficient Vertical Scaling of LLMs

**Zhen Tan\*, Daize Dong\*, Xinyu Zhao, Jianing Cai, Jie Peng, Yu Cheng, Tianlong Chen**

Published on the *First Workshop on Scalable Optimization for Efficient and Adaptive Foundation Models (SCOPE - ICLR 2025 Workshop)*.

[![OpenReview](https://img.shields.io/badge/arXiv-2402.02464-b31b1b.svg?style=plastic)](https://openreview.net/forum?id=E9Jw3IHuDH)

## Introduction

In this paper, we introduce Dynamic Layer Operations (DLO), a novel approach for vertically scaling transformer-based Large Language Models (LLMs) by dynamically expanding, activating, or skipping layers using a sophisticated routing policy based on layerwise feature similarity. Unlike traditional Mixture-of-Experts (MoE) methods that focus on extending the model width, our approach targets model depth, addressing the redundancy observed across layer representations for various input samples. Our framework is integrated with the Supervised Fine-Tuning (SFT) stage, eliminating the need for resource-intensive Continual Pre-Training (CPT). Experimental results demonstrate that DLO not only outperforms the original unscaled models but also achieves comparable results to densely expanded models with significantly improved efficiency. Our work offers a promising direction for building efficient yet powerful LLMs.

## Installation

```bash
conda create --name llama-dlo python=3.11
conda activate llama-dlo
pip install -e .
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Prepare Datasets

First change the `SAVE_PATH` in `entrypoints/data/download_dataset.py` and run the following command to download the dataset:

```bash
python entrypoints/data/download_dataset.py
```

Then you should change the `DATASET_PATH` in `entrypoints/data/reformat_datasets.py` and `entrypoints/data/mix_datasets.py`, and run the following command to reformat and mix the dataset:

```bash
bash scripts/data/reformat_datasets.sh
bash scripts/data/mix_datasets.sh
```

## Prepare Models

To convert a LLaMA model into LLaMA-DLO / LLaMA-Pro, please change the `model_path` and `output_path` first, and run:

```bash
bash scripts/convert/convert_llama_dlo.sh
bash scripts/convert/convert_llama_pro.sh
```

## Run Finetuning

To finetune a model w/o DLO implementation, please run:

```bash
bash scripts/finetune/finetune_normal.sh
```

To finetune a DLO model, please run:

```bash
bash scripts/finetune/finetune_dlo.sh
```
