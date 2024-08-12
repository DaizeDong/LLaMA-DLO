# DLO: Dynamic Layer Operation for Efficient Vertical Scaling of LLMs

## Environments

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
