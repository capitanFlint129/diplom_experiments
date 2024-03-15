from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy

from config import TrainConfig
from utils import BinnedStatistic


@dataclass
class EpisodeData:
    base_observations_history: list[np.ndarray]
    done: bool = False
    total_reward: float = 0
    total_negative_reward: float = 0
    actions_count: int = 0
    patience_count: int = 0
    losses: list[float] = field(default_factory=lambda: [])
    chosen_flags: list[str] = field(default_factory=lambda: [])
    forbidden_actions: set[int] = field(default_factory=lambda: set())
    remains: int = TrainConfig.episode_length

    def update_after_episode(
        self,
        action: int,
        reward: float,
        base_observation: np.ndarray,
        flags: list[str],
        loss_val: Optional[float],
    ) -> None:
        self.chosen_flags.extend(flags)
        self.actions_count += 1
        self.remains -= 1
        self.total_reward += reward
        if reward < 0:
            self.total_negative_reward += reward

        if reward == 0:
            self.patience_count += 1
            self.forbidden_actions.add(action)
        else:
            self.patience_count = 0
            self.forbidden_actions = set()

        self.base_observations_history.append(base_observation)
        if loss_val is not None:
            self.losses.append(loss_val)


def apply_modifiers(
    observation, modifiers, episode_data: EpisodeData, config: TrainConfig
):
    for modifier in modifiers:
        if modifier.startswith("remains-counter"):
            counter = episode_data.remains
            if modifier == "remains-counter-normalized":
                counter /= config.episode_length
            observation = np.concatenate((observation, np.array([counter])))
        elif modifier.startswith("prev"):
            prev_n = int(modifier.split("-")[1])
            prev = []
            for i in range(prev_n - 1):
                index = max(-i, 0)
                prev.append(episode_data.base_observations_history[index])
            observation = np.concatenate(prev + [observation])
    return observation


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
