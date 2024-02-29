from dataclasses import dataclass, field
import itertools
import os
import sys

import numpy as np
import torch
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer
from compiler_gym.wrappers.datasets import RandomOrderBenchmarks

from config import TrainConfig
from dqn.dqn import Agent
from observation import get_observation

MODELS_DIR = "_models"


@dataclass
class EpisodeData:
    done: bool = False
    total_reward: float = 0
    actions_count: int = 0
    patience_count: int = 0
    losses: list[float] = field(default_factory=lambda: [])
    chosen_flags: list[str] = field(default_factory=lambda: [])


def train(
    run,
    agent: Agent,
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
    history = []
    # используем среднее из средних геометрических по всем датасетам,
    # потому что есть неудобные датасеты, на которых среднее геометрическое почти всегда 0
    best_val_mean_geomean = 0

    for episode_i in range(config.episodes):
        agent.Q_eval.train()
        while True:
            train_env.reset()
            observation, is_observation_correct = get_observation(train_env, config)
            if is_observation_correct:
                break
            else:
                print(
                    f"Skip {env.benchmark} during training. It produces incorrect initial observation",
                    file=sys.stderr,
                )

        episode_data = EpisodeData()
        agent.episode_reset()
        forbidden_actions = set()
        while (
            not episode_data.done
            and episode_data.actions_count < config.episode_length
            and episode_data.patience_count < config.patience
        ):
            action = agent.choose_action(
                observation, forbidden_actions=forbidden_actions
            )
            flag = config.actions[action]
            episode_data.chosen_flags.append(flag)
            _, reward, episode_data.done, info = train_env.step(
                train_env.action_space.flags.index(flag)
            )
            new_observation, _ = get_observation(env, config)
            episode_data.actions_count += 1
            episode_data.total_reward += reward

            if reward == 0:
                episode_data.patience_count += 1
                forbidden_actions.add(action)
            else:
                episode_data.patience_count = 0
                forbidden_actions = set()

            agent.store_transition(
                action, observation, reward, new_observation, episode_data.done
            )
            loss_val = agent.learn()
            if loss_val is not None:
                episode_data.losses.append(loss_val)
            observation = new_observation

            if len(agent.actions_taken) == len(config.actions):
                episode_data.done = True

        agent.episode_done()
        history.append(episode_data.total_reward)
        average_rewards_sum = np.mean(history[-config.logging_history_size :])
        print(
            f"{episode_i} - {train_env.benchmark}\n"
            + "Total: {:.4f}".format(episode_data.total_reward)
            + " Epsilon: {:.4f}".format(agent.epsilon)
            + f" Average rewards sum: {average_rewards_sum}"
            + f" Action: {' '.join(episode_data.chosen_flags)}"
        )
        run.log(
            {
                "average_rewards_sum_for_last_episodes": average_rewards_sum,
                "std_rewards_sum_for_last_episodes": np.std(history),
                "average_episode_loss": np.mean(episode_data.losses or [0]),
                "total_episode_reward": episode_data.total_reward,
            },
            step=episode_i,
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
    save_model(agent.Q_eval.state_dict(), run.name, replace=False)


def _validation_during_train(
    run,
    episode_i,
    best_val_mean_geomean,
    agent: Agent,
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
        save_model(agent.Q_eval.state_dict(), f"{run.name}")
        return validation_result.mean_geomean_reward
    return best_val_mean_geomean


@dataclass
class ValidationResult:
    geomean_reward: float
    mean_geomean_reward: float
    geomean_reward_per_dataset: dict[str, float]
    mean_walltime: float


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
    for dataset_name, benchmarks in val_benchmarks.items():
        rewards[dataset_name] = []
        for benchmark in benchmarks:
            env.reset(benchmark=benchmark)
            observation, is_observation_correct = get_observation(env, config)
            if is_observation_correct:
                with Timer() as timer:
                    reward, applied_actions = rollout(agent, env, config)
                rewards[dataset_name].append(reward)
                times.append(timer.time)
                if enable_logs:
                    applied_actions = [
                        config.actions[action_i] for action_i in applied_actions
                    ]
                    print(
                        f"{benchmark} - reward: {reward} - time: {timer.time} - actions: {' '.join(applied_actions)}"
                    )
            else:
                print(
                    f"Skip {benchmark} during validation. It produces incorrect initial observation",
                    file=sys.stderr,
                )
    geomean_reward = geometric_mean(
        list(itertools.chain.from_iterable(rewards.values()))
    )
    geomean_reward_per_dataset = {
        dataset_name: geometric_mean(dataset_rewards)
        for dataset_name, dataset_rewards in rewards.items()
    }
    mean_walltime = arithmetic_mean(times)
    mean_geomean_reward = arithmetic_mean(list(geomean_reward_per_dataset.values()))
    return ValidationResult(
        geomean_reward,
        mean_geomean_reward,
        geomean_reward_per_dataset,
        mean_walltime,
    )


@torch.no_grad()
def rollout(agent: Agent, env, config: TrainConfig) -> tuple[float, list[str]]:
    observation, _ = get_observation(env, config)
    action_seq, rewards = [], []
    agent.episode_reset()
    change_count = 0

    for i in range(config.episode_length):
        action = agent.choose_action(observation, enable_epsilon_greedy=False)
        flag = config.actions[action]
        action_seq.append(action)
        _, reward, done, info = env.step(env.action_space.flags.index(flag))
        observation, _ = get_observation(env, config)
        rewards.append(reward)

        if reward == 0:
            change_count += 1
        else:
            change_count = 0

        if len(agent.actions_taken) == len(config.actions):
            done = True

        if done or change_count > config.patience:
            break

    return sum(rewards), action_seq


def save_model(state_dict, model_name, replace=True):
    if not replace and os.path.exists(f"./{MODELS_DIR}/{model_name}.pth"):
        return
    if not os.path.exists(f"./{MODELS_DIR}"):
        os.makedirs(f"./{MODELS_DIR}")
    torch.save(state_dict, f"./{MODELS_DIR}/{model_name}.pth")
