from dataclasses import dataclass, field
from typing import Optional, Any

import wandb
import numpy as np
import scipy

from config.config import TrainConfig
from utils import BinnedStatistic


@dataclass
class StepResult:
    action: int
    reward: float
    value: float
    new_observation: np.ndarray
    new_base_observation: np.ndarray
    flags: list[str]
    info: dict[str, Any]
    done: bool


@dataclass
class EpisodeData:
    done: bool = False
    total_reward: float = 0
    total_negative_reward: float = 0
    actions_count: int = 0
    patience_count: int = 0
    losses: list[float] = field(default_factory=lambda: [])
    values: list[float] = field(default_factory=lambda: [])
    rewards: list[float] = field(default_factory=lambda: [])
    chosen_flags: list[list[str]] = field(default_factory=lambda: [])
    forbidden_actions: set[int] = field(default_factory=lambda: set())
    remains: int = TrainConfig.episode_length

    def update_after_episode_step(
        self,
        step_result: StepResult,
        loss_value: Optional[float],
    ) -> None:
        self.chosen_flags.append(step_result.flags)
        self.actions_count += 1
        self.remains -= 1
        self.rewards.append(step_result.reward)
        self.total_reward += step_result.reward
        if step_result.reward < 0:
            self.total_negative_reward += step_result.reward

        # if step_result.info.get("action_had_no_effect", True):
        if step_result.reward == 0:
            self.patience_count += 1
            self.forbidden_actions.add(step_result.action)
        else:
            self.patience_count = 0
            self.forbidden_actions = set()

        if loss_value is not None:
            self.losses.append(loss_value)
        self.values.append(step_result.value)


@dataclass
class TrainHistory:
    logging_history_size: int
    rewards_history: list[float] = field(default_factory=lambda: [])
    # rewards_histogram: list[list[float]] = field(default_factory=lambda: [])
    negative_rewards_history: list[float] = field(default_factory=lambda: [])
    best_val_mean: float = 0

    def get_average_rewards_sum(self, log_size: Optional[int] = None) -> float:
        if log_size is None:
            log_size = self.logging_history_size
        return np.mean(self.rewards_history[-log_size:])

    def get_average_negative_rewards_sum(self, log_size: Optional[int] = None) -> float:
        if log_size is None:
            log_size = self.logging_history_size
        return np.mean(self.negative_rewards_history[-log_size:])

    def get_rewards_std(self, log_size: Optional[int] = None) -> float:
        if log_size is None:
            log_size = self.logging_history_size
        return np.std(self.rewards_history[-log_size:])

    # def reward_hist_for_step(self, log_size: Optional[int] = None):
    #     if log_size is None:
    #         log_size = self.logging_history_size
    #     data = [
    #         [i, np.mean(step_data[-log_size:])]
    #         for i, step_data in enumerate(self.rewards_histogram)
    #     ]
    #     table = wandb.Table(data=data, columns=["step", "reward"])
    #     reward_hist = wandb.plot.scatter(table, "step", "reward")
    #
    #     data = [
    #         [i, np.std(step_data[-log_size:])]
    #         for i, step_data in enumerate(self.rewards_histogram)
    #     ]
    #     table = wandb.Table(data=data, columns=["step", "reward_std"])
    #     reward_hist_std = wandb.plot.scatter(table, "step", "reward_std")
    #     return reward_hist, reward_hist_std

    def update(self, episode_data: EpisodeData) -> None:
        self.rewards_history.append(episode_data.total_reward)
        self.negative_rewards_history.append(episode_data.total_negative_reward)
        # if len(self.rewards_histogram) == 0:
        #     self.rewards_histogram = [[reward] for reward in episode_data.rewards]
        # else:
        #     for i, reward in enumerate(episode_data.rewards):
        #         self.rewards_histogram[i].append(reward)


def get_binned_statistics(
    binned_statistic_data: dict[str, tuple[list[int], list[float]]]
) -> tuple[BinnedStatistic, dict[str, BinnedStatistic]]:
    rewards_sum_by_codesize_bins_per_dataset = {}
    all_codesizes = []
    all_rewards = []
    for dataset_name, (
        dataset_codesizes,
        dataset_rewards,
    ) in binned_statistic_data.items():
        all_codesizes.extend(dataset_codesizes)
        all_rewards.extend(dataset_rewards)
        rewards_sum_by_codesize_bins_per_dataset[
            dataset_name
        ] = _get_mean_and_std_binned(dataset_codesizes, dataset_rewards)
    rewards_sum_by_codesize_bins = _get_mean_and_std_binned(all_codesizes, all_rewards)
    return rewards_sum_by_codesize_bins, rewards_sum_by_codesize_bins_per_dataset


def _get_mean_and_std_binned(dataset_codesizes, dataset_rewards):
    mean, bin_edges, binnumber = scipy.stats.binned_statistic(
        x=dataset_codesizes,
        values=dataset_rewards,
        bins=TrainConfig.codesize_bins_number,
        statistic="mean",
    )
    std, _, _ = scipy.stats.binned_statistic(
        x=dataset_codesizes,
        values=dataset_rewards,
        bins=TrainConfig.codesize_bins_number,
        statistic="std",
    )
    return BinnedStatistic(
        mean=mean,
        std=std,
        bin_edges=bin_edges,
        binnumber=binnumber,
    )
