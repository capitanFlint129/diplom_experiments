import itertools
import sys

import numpy as np
import torch
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer
from compiler_gym.wrappers.datasets import RandomOrderBenchmarks

from config import TrainConfig
from dqn.dqn import DQNAgent
from observation import get_observation
from train_utils import EpisodeData, apply_modifiers, get_binned_statistics
from utils import save_model, ValidationResult


def train(
    run,
    agent: DQNAgent,
    env,
    config: TrainConfig,
    train_benchmarks: list,
    val_benchmarks: dict,
    enable_validation: bool = True,
    enable_validation_logs: bool = False,
) -> None:
    train_env = RandomOrderBenchmarks(
        env.fork(),
        benchmarks=train_benchmarks,
        rng=np.random.default_rng(config.random_state),
    )
    rewards_history = []
    # используем среднее из средних геометрических по всем датасетам,
    # потому что есть неудобные датасеты, на которых среднее геометрическое почти всегда 0
    best_val_mean_geomean = 0

    for episode_i in range(config.episodes):
        agent.policy_net.train()
        while True:
            train_env.reset()
            base_observation, is_observation_correct = get_observation(
                train_env, config
            )
            if is_observation_correct:
                break
            else:
                print(
                    f"Skip {env.benchmark} during training. It produces incorrect initial observation",
                    file=sys.stderr,
                )

        episode_data = EpisodeData(
            base_observations_history=[base_observation], remains=config.episode_length
        )
        agent.episode_reset()
        observation = apply_modifiers(
            base_observation, config.observation_modifiers, episode_data
        )
        while (
            not episode_data.done
            and episode_data.actions_count < config.episode_length
            and episode_data.patience_count < config.patience
        ):
            action = agent.choose_action(
                observation, forbidden_actions=episode_data.forbidden_actions
            )
            flags = config.actions[action]
            if " " in flags:
                flags = flags.split()
            else:
                flags = [flags]
            _, reward, episode_data.done, info = train_env.multistep(
                [train_env.action_space.flags.index(f) for f in flags]
            )
            new_base_observation, _ = get_observation(env, config)

            new_observation = apply_modifiers(
                new_base_observation, config.observation_modifiers, episode_data
            )

            agent.store_transition(
                action, observation, reward, new_observation, episode_data.done
            )
            loss_val = agent.learn()
            episode_data.update_after_episode(
                action=action,
                reward=reward,
                base_observation=new_base_observation,
                flags=flags,
                loss_val=loss_val,
            )
            observation = new_observation

        agent.episode_done()
        rewards_history.append(episode_data.total_reward)
        average_rewards_sum = np.mean(rewards_history[-config.logging_history_size :])
        std_rewards_sum = np.std(rewards_history[-config.logging_history_size :])
        _log_episode_results(
            run,
            train_env,
            agent.epsilon,
            episode_i,
            average_rewards_sum,
            std_rewards_sum,
            episode_data,
        )
        if episode_i % config.validation_interval == 0 and enable_validation:
            best_val_mean_geomean = _validation_during_train(
                run,
                episode_i,
                best_val_mean_geomean,
                agent,
                env,
                config,
                val_benchmarks,
                enable_logs=enable_validation_logs,
            )

    if (config.episodes - 1) % config.validation_interval != 0 and enable_validation:
        _validation_during_train(
            run,
            config.episodes - 1,
            best_val_mean_geomean,
            agent,
            env,
            config,
            val_benchmarks,
            enable_logs=enable_validation_logs,
        )
    save_model(agent.policy_net.state_dict(), run.name, replace=False)


def _validation_during_train(
    run,
    episode_i,
    best_val_mean_geomean,
    agent: DQNAgent,
    env,
    config: TrainConfig,
    val_benchmarks: dict,
    enable_logs: bool = False,
) -> float:
    validation_result = validate(
        agent, env, config, val_benchmarks, enable_logs=enable_logs
    )
    log_data = {
        f"val_geomean_reward_{dataset_name}": geomean_reward
        for dataset_name, geomean_reward in validation_result.geomean_reward_per_dataset.items()
    }
    log_data["val_geomean_reward"] = validation_result.geomean_reward
    run.log(
        log_data,
        step=episode_i,
    )
    if validation_result.mean_geomean_reward > best_val_mean_geomean:
        print(
            f"Save model. New best geomean: {validation_result.mean_geomean_reward}, previous best geomean: {best_val_mean_geomean}"
        )
        save_model(agent.policy_net.state_dict(), f"{run.name}")
        return validation_result.mean_geomean_reward
    return best_val_mean_geomean


def _log_episode_results(
    run,
    env,
    epsilon: float,
    episode_i: int,
    average_rewards_sum: float,
    std_rewards_sum: float,
    episode_data: EpisodeData,
) -> None:
    print(
        f"{episode_i} - {env.benchmark}\n"
        + "Total: {:.4f}".format(episode_data.total_reward)
        + " Total neg: {:.4f}".format(episode_data.total_negative_reward)
        + " Epsilon: {:.4f}".format(epsilon)
        + f" Average rewards sum: {average_rewards_sum}"
        + f" Action: {' '.join(episode_data.chosen_flags)}"
    )
    run.log(
        {
            "average_rewards_sum_for_last_episodes": average_rewards_sum,
            "std_rewards_sum_for_last_episodes": std_rewards_sum,
            "average_episode_loss": np.mean(episode_data.losses or [0]),
            "total_episode_reward": episode_data.total_reward,
        },
        step=episode_i,
    )


def validate(
    agent,
    env,
    config: TrainConfig,
    val_benchmarks: dict[str, list],
    enable_logs: bool = False,
) -> ValidationResult:
    agent.eval()
    rewards = {}
    times = []
    binned_statistic_data = {}
    for dataset_name, benchmarks in val_benchmarks.items():
        codesize = []
        rewards[dataset_name] = []
        for i, benchmark in enumerate(benchmarks):
            env.reset(benchmark=benchmark)
            observation, is_observation_correct = get_observation(env, config)
            codesize.append(env.observation["IrInstructionCount"])
            if is_observation_correct:
                with Timer() as timer:
                    episode_data = rollout(agent, env, config)
                rewards[dataset_name].append(episode_data.total_reward)
                times.append(timer.time)
                if enable_logs:
                    print(
                        f"{i} - {benchmark} - reward: {episode_data.total_reward} - time: {timer.time} - actions: {' '.join(episode_data.chosen_flags)}"
                    )
            else:
                print(
                    f"Skip {benchmark} during validation. It produces incorrect initial observation",
                    file=sys.stderr,
                )
        binned_statistic_data[dataset_name] = (codesize, rewards[dataset_name])
    geomean_reward = geometric_mean(
        list(itertools.chain.from_iterable(rewards.values()))
    )
    geomean_reward_per_dataset = {
        dataset_name: geometric_mean(dataset_rewards)
        for dataset_name, dataset_rewards in rewards.items()
    }
    mean_walltime = arithmetic_mean(times)
    mean_geomean_reward = arithmetic_mean(list(geomean_reward_per_dataset.values()))
    (
        rewards_sum_by_codesize_bins,
        rewards_sum_by_codesize_bins_per_dataset,
    ) = get_binned_statistics(binned_statistic_data)
    return ValidationResult(
        geomean_reward,
        mean_geomean_reward,
        geomean_reward_per_dataset,
        mean_walltime,
        rewards_sum_by_codesize_bins,
        rewards_sum_by_codesize_bins_per_dataset,
    )


@torch.no_grad()
def rollout(agent: DQNAgent, env, config: TrainConfig) -> EpisodeData:
    agent.episode_reset()

    base_observation, _ = get_observation(env, config)
    episode_data = EpisodeData(
        base_observations_history=[base_observation], remains=config.episode_length
    )
    observation = apply_modifiers(
        base_observation, config.observation_modifiers, episode_data
    )

    while (
        not episode_data.done
        and episode_data.actions_count < config.episode_length
        and episode_data.patience_count < config.val_patience
    ):
        if config.eval_with_forbidden_actions:
            action = agent.choose_action(
                observation,
                enable_epsilon_greedy=False,
                forbidden_actions=episode_data.forbidden_actions,
            )
        else:
            action = agent.choose_action(observation, enable_epsilon_greedy=False)
        flags = config.actions[action]
        if " " in flags:
            flags = flags.split()
        else:
            flags = [flags]
        _, reward, episode_data.done, info = env.multistep(
            [env.action_space.flags.index(f) for f in flags]
        )
        base_observation, _ = get_observation(env, config)
        observation = apply_modifiers(
            base_observation, config.observation_modifiers, episode_data
        )
        episode_data.update_after_episode(
            action=action,
            reward=reward,
            base_observation=base_observation,
            flags=flags,
            loss_val=None,
        )

    return episode_data
