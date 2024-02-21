import random

import compiler_gym
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def make_env(config):
    return compiler_gym.make(
        config["compiler_gym_env"],
        reward_space=config["reward_space"],
    )


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def prepare_datasets(
    env,
    datasets: list[str],
    train_val_test_split: bool = True,
    skipped: set[str] = None,
) -> tuple[list, dict, dict]:
    if skipped is None:
        skipped = set()
    if not train_val_test_split:
        train_benchmarks = []
        test_and_val_benchmarks = {}
        for dataset_name in datasets:
            benchmarks = _filter_benchmarks(
                list(env.datasets[dataset_name].benchmarks()), skipped
            )
            test_and_val_benchmarks[dataset_name] = benchmarks
            train_benchmarks.extend(test_and_val_benchmarks[dataset_name])
        random.shuffle(train_benchmarks)
        return train_benchmarks, test_and_val_benchmarks, test_and_val_benchmarks
    train_benchmarks = []
    val_benchmarks = {}
    test_benchmarks = {}
    for dataset_name in datasets:
        benchmarks = _filter_benchmarks(
            list(env.datasets[dataset_name].benchmarks()), skipped
        )
        train, test = train_test_split(benchmarks, test_size=0.2, shuffle=False)
        train, val = train_test_split(benchmarks, test_size=0.125, shuffle=False)
        train_benchmarks.extend(train)
        val_benchmarks[dataset_name] = val
        test_benchmarks[dataset_name] = test
    random.shuffle(train_benchmarks)
    return train_benchmarks, val_benchmarks, test_benchmarks


def _filter_benchmarks(dataset, skipped):
    return [becnhmark for becnhmark in dataset if str(becnhmark) not in skipped]
