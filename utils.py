import random

# noinspection PyUnresolvedReferences
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
    env, datasets: list[str], no_split: bool = False
) -> tuple[list, dict, dict]:
    # Пока такое решение работает очень медленно
    # Пример бенчмарка с плохим векторным представлением где все 0: benchmark://tensorflow-v0/1786
    # def filter_zero(env, benchmarks):
    #     result = []
    #     for benchmark in benchmarks:
    #         env.reset(benchmark=benchmark)
    #         if np.sum(env.observation[config["observation_space"]] > 0) != 0:
    #             result.append(benchmark)
    #     return result

    if no_split:
        train_benchmarks = []
        test_and_val_benchmarks = {}
        for dataset_name in datasets:
            benchmarks = list(env.datasets[dataset_name].benchmarks())
            test_and_val_benchmarks[dataset_name] = benchmarks
            train_benchmarks.extend(test_and_val_benchmarks[dataset_name])
        random.shuffle(train_benchmarks)
        return train_benchmarks, test_and_val_benchmarks, test_and_val_benchmarks
    train_benchmarks = []
    val_benchmarks = {}
    test_benchmarks = {}
    for dataset_name in datasets:
        benchmarks = list(env.datasets[dataset_name].benchmarks())
        train, test = train_test_split(benchmarks, test_size=0.2, shuffle=False)
        train, val = train_test_split(benchmarks, test_size=0.125, shuffle=False)
        train_benchmarks.extend(train)
        val_benchmarks[dataset_name] = val
        test_benchmarks[dataset_name] = test
    random.shuffle(train_benchmarks)
    return train_benchmarks, val_benchmarks, test_benchmarks
