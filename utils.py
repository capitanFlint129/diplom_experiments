from itertools import islice
from typing import Any

import compiler_gym
from compiler_gym.wrappers import (
    ConstrainedCommandline,
    TimeLimit,
)
from sklearn.model_selection import train_test_split


def make_env(config: dict[str, Any]) -> compiler_gym.envs.CompilerEnv:
    env = compiler_gym.make(
        config["compiler_gym_env"],
        reward_space=config["reward_space"],
    )
    if config["actions"] is not None:
        env = ConstrainedCommandline(
            env,
            flags=config["actions"],
        )
    if config["episode_length"] is not None:
        env = TimeLimit(env, max_episode_steps=config["episode_length"])
    return env


def prepare_datasets(
    env: compiler_gym.envs.CompilerEnv, config: dict[str, Any]
) -> tuple:
    dataset_limit = config["dataset_limit"]
    train_benchmarks = []
    val_benchmarks = []
    test_benchmarks = []
    for dataset_name in config["datasets"]:
        benchmarks = list(
            islice(env.datasets[dataset_name].benchmarks(), dataset_limit)
        )
        train, test = train_test_split(
            benchmarks, test_size=0.2, random_state=config["random_state"]
        )
        train, val = train_test_split(
            benchmarks, test_size=0.125, random_state=config["random_state"]
        )
        train_benchmarks.extend(train)
        val_benchmarks.extend(val)
        test_benchmarks.extend(test)
    return train_benchmarks, val_benchmarks, test_benchmarks
