import itertools
import os
import random
from dataclasses import dataclass
from typing import Optional, Union

import compiler_gym
import numpy as np
import plotly.graph_objects as go
import torch
from compiler_gym.datasets import FilesDataset
from sklearn.model_selection import train_test_split

from config.config import MODELS_DIR, WANDB_PROJECT_NAME, TrainConfig
from dqn.dqn import DoubleDQNAgent, DQNAgent, LstmDQNAgent, SimpleDQNAgent, TwinDQNAgent


@dataclass
class BinnedStatistic:
    mean: np.ndarray
    std: np.ndarray
    bin_edges: np.ndarray
    binnumber: np.ndarray


@dataclass
class ValidationResult:
    geomean_reward: float
    mean_geomean_reward: float
    geomean_reward_per_dataset: dict[str, float]
    mean_walltime: float
    rewards_sum_by_codesize_bins: BinnedStatistic
    rewards_sum_by_codesize_bins_per_dataset: dict[str, BinnedStatistic]


def get_agent(config: TrainConfig, device, policy_net_path: Optional[str]) -> DQNAgent:
    if config.algorithm == "DQN":
        agent = SimpleDQNAgent(
            observation_size=config.observation_size,
            n_actions=len(config.actions),
            config=config,
            device=device,
            enable_dueling_dqn=config.enable_dueling_dqn,
        )
    elif config.algorithm == "DoubleDQN":
        agent = DoubleDQNAgent(
            observation_size=config.observation_size,
            n_actions=len(config.actions),
            config=config,
            device=device,
            enable_dueling_dqn=config.enable_dueling_dqn,
        )
    elif config.algorithm == "TwinDQN":
        agent = TwinDQNAgent(
            observation_size=config.observation_size,
            n_actions=len(config.actions),
            config=config,
            device=device,
            enable_dueling_dqn=config.enable_dueling_dqn,
        )
    elif config.algorithm == "LstmDQN":
        if config.enable_dueling_dqn:
            raise NotImplemented("LSTM + Dueling")

        agent = LstmDQNAgent(
            observation_size=config.observation_size,
            n_actions=len(config.actions),
            config=config,
            device=device,
        )
    else:
        raise Exception("unknown algorithm used")
    if policy_net_path is not None:
        agent.policy_net.load_state_dict(torch.load(policy_net_path))
        agent.policy_net.eval()
    return agent


def save_model(state_dict, model_name: str, replace: bool = True):
    model_path = get_model_path(model_name)
    models_dir = os.path.join(MODELS_DIR, WANDB_PROJECT_NAME)
    if not replace and os.path.exists(model_path):
        return
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    torch.save(state_dict, get_model_path(model_name))


def get_model_path(model_name: str) -> str:
    return os.path.join(MODELS_DIR, WANDB_PROJECT_NAME, f"{model_name}.pth")


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
    random_state: int,
    train_val_test_split: bool,
    skipped: set[str],
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
        train, test = train_test_split(
            benchmarks, test_size=0.2, random_state=random_state
        )
        train, val = train_test_split(train, test_size=0.125, random_state=random_state)
        train_benchmarks.extend(train)
        val_benchmarks[dataset_config] = val
        test_benchmarks[dataset_config] = test
    # random.shuffle(train_benchmarks)
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


def get_binned_statistics_plot(rewards_sum_by_codesize_bins: BinnedStatistic):
    means = rewards_sum_by_codesize_bins.mean
    stds = rewards_sum_by_codesize_bins.std
    bin_edges = rewards_sum_by_codesize_bins.bin_edges
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2
    not_nan_mask = ~np.isnan(means)
    fig = go.Scatter(
        x=bin_centers[not_nan_mask],
        y=means[not_nan_mask],
        mode="markers",
        name="measured",
        error_y=dict(
            type="data",
            array=stds[not_nan_mask],
            color="purple",
            thickness=1.5,
            width=3,
        ),
        error_x=dict(
            type="constant",
            value=bin_width / 2,
            color="purple",
            thickness=1.5,
            width=3,
        ),
        marker=dict(color="purple", size=8),
    )
    return fig


def _filter_benchmarks(dataset, skipped):
    return [becnhmark for becnhmark in dataset if str(becnhmark) not in skipped]


def _get_benchmarks(env, dataset_config, skipped) -> list:
    if isinstance(dataset_config, tuple):
        dataset_name, dataset_size = dataset_config
    else:
        dataset_name, dataset_size = dataset_config, None
    dataset = _load_dataset(env, dataset_name)
    if dataset_size is not None:
        benchmarks = list(itertools.islice(dataset.benchmarks(), dataset_size))
    else:
        benchmarks = list(env.datasets[dataset_config].benchmarks())
    benchmarks = _filter_benchmarks(benchmarks, skipped)
    return benchmarks


def _load_dataset(env, dataset_name):
    if os.path.exists(os.path.dirname(dataset_name)):
        return FilesDataset(
            dataset_root=dataset_name,
            benchmark_file_suffix=".bc",
            name="custom_dataset",
            description="",
            license="",
        )
    else:
        return env.datasets[dataset_name]
