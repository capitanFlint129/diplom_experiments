import itertools
import os
import random
from typing import Union

import compiler_gym
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from config import TrainConfig, MODELS_DIR


def save_model(state_dict, model_name, replace=True):
    if not replace and os.path.exists(f"./{MODELS_DIR}/{model_name}.pth"):
        return
    if not os.path.exists(f"./{MODELS_DIR}"):
        os.makedirs(f"./{MODELS_DIR}")
    torch.save(state_dict, f"./{MODELS_DIR}/{model_name}.pth")


def make_env(config: TrainConfig):
    return compiler_gym.make(
        config.compiler_gym_env,
        reward_space=config.reward_space,
    )


def fix_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def prepare_datasets(
    env,
    datasets: list[Union[str, tuple[str, int]]],
    train_val_test_split: bool = True,
    skipped: set[str] = None,
) -> tuple[list, dict, dict]:
    if skipped is None:
        skipped = set()
    if not train_val_test_split:
        train_benchmarks = []
        test_and_val_benchmarks = {}
        for dataset_config in datasets:
            benchmarks = _get_benchmarks(env, dataset_config, skipped)
            test_and_val_benchmarks[dataset_config] = benchmarks
            train_benchmarks.extend(test_and_val_benchmarks[dataset_config])
        random.shuffle(train_benchmarks)
        return train_benchmarks, test_and_val_benchmarks, test_and_val_benchmarks
    train_benchmarks = []
    val_benchmarks = {}
    test_benchmarks = {}
    for dataset_config in datasets:
        benchmarks = _get_benchmarks(env, dataset_config, skipped)
        train, test = train_test_split(benchmarks, test_size=0.2, shuffle=False)
        train, val = train_test_split(benchmarks, test_size=0.125, shuffle=False)
        train_benchmarks.extend(train)
        val_benchmarks[dataset_config] = val
        test_benchmarks[dataset_config] = test
    random.shuffle(train_benchmarks)
    return train_benchmarks, val_benchmarks, test_benchmarks


def get_last_model_wandb_naming(models_dir: str) -> str:
    models_files_with_run_id = [
        (filename.split("-")[-1].split(".")[0], filename)
        for filename in list(os.listdir(models_dir))
    ]
    models_files_with_run_id = [
        (int(int_run_id), filename)
        for (int_run_id, filename) in models_files_with_run_id
        if int_run_id.isdecimal()
    ]
    return sorted(models_files_with_run_id)[-1][1]


def _filter_benchmarks(dataset, skipped):
    return [becnhmark for becnhmark in dataset if str(becnhmark) not in skipped]


def _get_benchmarks(env, dataset_config, skipped) -> list:
    if isinstance(dataset_config, tuple):
        dataset_config, dataset_size = dataset_config
        benchmarks = list(
            itertools.islice(env.datasets[dataset_config].benchmarks(), dataset_size)
        )
    else:
        benchmarks = list(env.datasets[dataset_config].benchmarks())
    benchmarks = _filter_benchmarks(benchmarks, skipped)
    return benchmarks
