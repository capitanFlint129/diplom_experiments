import dataclasses
import itertools
import os
import sys

import numpy as np
import torch
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer
from compiler_gym.wrappers.datasets import RandomOrderBenchmarks

from dqn.dqn import Agent

MODELS_DIR = "../models"


def process_observation(observation):
    # Autophase
    # return observation / observation[51]
    # InstCountNorm
    return observation


def save_model(state_dict, model_name, replace=True):
    if not replace and os.path.exists(f"./{MODELS_DIR}/{model_name}.pth"):
        return
    if not os.path.exists(f"./{MODELS_DIR}"):
        os.makedirs(f"./{MODELS_DIR}")
    torch.save(state_dict, f"./{MODELS_DIR}/{model_name}.pth")


def is_observation_correct(observation):
    return (observation > 0).sum() != 0 and not np.any(np.isnan(observation))


def train(
    run,
    agent: Agent,
    env,
    config,
    train_benchmarks: list,
    val_benchmarks: dict,
) -> None:
    env.observation_space = config["observation_space"]

    train_env = RandomOrderBenchmarks(
        env.fork(),
        benchmarks=train_benchmarks,
        rng=np.random.default_rng(config["random_state"]),
    )

    history_size = config["logging_history_size"]
    mem_cntr = 0
    history = np.zeros(history_size)
    # используем среднее из средних геометрических по всем датасетам,
    # потому что есть неудобные датасеты, на которых среднее геометрическое - 0
    best_val_mean_geomean = 0

    for episode_i in range(config["episodes"]):
        agent.Q_eval.train()
        observation = np.zeros(1)
        # skip zero vectors
        while not is_observation_correct(observation):
            train_env.reset()
            observation = process_observation(
                train_env.observation[config["observation_space"]]
            )
        done = False
        total = 0
        actions_taken = 0
        agent.actions_taken = []
        change_count = 0

        losses = []
        chosen_flags = []
        while (
            not done
            and actions_taken < config["episode_length"]
            and change_count < config["patience"]
        ):
            action = agent.choose_action(observation)
            flag = config["actions"][action]
            chosen_flags.append(flag)
            new_observation, reward, done, info = train_env.step(
                train_env.action_space.flags.index(flag),
                observation_spaces=[config["observation_space"]],
            )
            new_observation = process_observation(new_observation[0])
            actions_taken += 1
            total += reward

            if reward == 0:
                change_count += 1
            else:
                change_count = 0

            agent.store_transition(action, observation, reward, new_observation, done)
            loss_val = agent.learn()
            if loss_val is not None:
                losses.append(loss_val)
            observation = new_observation

            if len(agent.actions_taken) == len(config["actions"]):
                done = True

        index = mem_cntr % history_size
        history[index] = total
        mem_cntr += 1
        print(f"{episode_i} - {train_env.benchmark}")
        print(
            "Total: {:.4f}".format(total)
            + " Epsilon: {:.4f}".format(agent.epsilon)
            + f" Average rewards sum: {str(np.mean(history))}"
            + f" Action: {' '.join(chosen_flags)}"
        )
        run.log(
            {
                "average_rewards_sum_last_100": np.mean(history),
                "std_rewards_sum_last_100": np.std(history),
                "average_episode_loss": np.mean(losses or [0]),
                "total_episode_reward": total,
            },
            step=episode_i,
        )
        if episode_i % 500 == 0:
            best_val_mean_geomean = _validation(
                run,
                episode_i,
                best_val_mean_geomean,
                agent,
                env,
                config,
                val_benchmarks,
            )

    if (config["episodes"] - 1) % 500 != 0:
        _validation(
            run,
            config["episodes"] - 1,
            best_val_mean_geomean,
            agent,
            env,
            config,
            val_benchmarks,
        )
    save_model(agent.Q_eval.state_dict(), run.name, replace=False)


def _validation(
    run, episode_i, best_val_mean_geomean, agent, env, config, val_benchmarks: dict
) -> float:
    validation_result = validate(agent, env, config, val_benchmarks)
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


@dataclasses.dataclass
class ValidationResult:
    geomean_reward: float
    mean_geomean_reward: float
    geomean_reward_per_dataset: dict[str, float]
    mean_walltime: float


def validate(agent, env, config, val_benchmarks: dict[str, list]) -> ValidationResult:
    agent.Q_eval.eval()
    rewards = {}
    times = []
    for dataset_name, benchmarks in val_benchmarks.items():
        rewards[dataset_name] = []
        for benchmark in benchmarks:
            env.reset(benchmark=benchmark)
            observation = env.observation[config["observation_space"]]
            if is_observation_correct(observation):
                with Timer() as timer:
                    reward = rollout(agent, env, config)
                rewards[dataset_name].append(reward)
                times.append(timer.time)
            else:
                print(f"{benchmark} skipped during validation", file=sys.stderr)
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
def rollout(agent: Agent, env, config):
    observation = process_observation(env.observation[config["observation_space"]])
    action_seq, rewards = [], []
    agent.actions_taken = []
    change_count = 0

    for i in range(config["episode_length"]):
        action = agent.choose_action(observation, disable_epsilon_greedy=True)
        flag = config["actions"][action]
        action_seq.append(action)
        observation, reward, done, info = env.step(
            env.action_space.flags.index(flag),
            observation_spaces=[config["observation_space"]],
        )
        observation = process_observation(observation[0])
        rewards.append(reward)

        if reward == 0:
            change_count += 1
        else:
            change_count = 0

        if len(agent.actions_taken) == len(config["actions"]):
            done = True

        if done or change_count > config["patience"]:
            break

    return sum(rewards)