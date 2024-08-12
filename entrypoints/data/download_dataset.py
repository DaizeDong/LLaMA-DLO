import multiprocessing
import os

# huggingface-cli login

SAVE_PATH = "xxxxxxxxxxxxxxxxxxxxxxxxx"  # 🔍

datasets = {
    "anon8231489123/ShareGPT_Vicuna_unfiltered": os.path.join(SAVE_PATH, "vicuna_sharegpt"),
    "WizardLM/WizardLM_evol_instruct_V2_196k": os.path.join(SAVE_PATH, "evol_instruct"),
    "Open-Orca/SlimOrca": os.path.join(SAVE_PATH, "slim_orca"),
    "meta-math/MetaMathQA": os.path.join(SAVE_PATH, "meta_math_qa"),
    "theblackcat102/evol-codealpaca-v1": os.path.join(SAVE_PATH, "evol_code_alpaca"),
}


def save_dataset(dataset_name, dataset_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=dataset_name,
        local_dir=dataset_dir,
        local_dir_use_symlinks=False,
        force_download=False,
        resume_download=True,
        repo_type="dataset",
    )


if __name__ == "__main__":
    processes = []

    for dataset_name, dataset_dir in datasets.items():
        process = multiprocessing.Process(target=save_dataset, args=(dataset_name, dataset_dir))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
