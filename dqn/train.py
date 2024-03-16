import itertools
import sys
from typing import Optional

import numpy as np
import torch
from compiler_gym.envs import CompilerEnv
from compiler_gym.errors import SessionNotFound
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer
from compiler_gym.wrappers.datasets import RandomOrderBenchmarks

from config import TrainConfig
from dqn.dqn import DQNAgent
from dqn.train_utils import EpisodeData, get_binned_statistics, StepResult
from observation import get_observation, ObservationModifier
from utils import save_model, ValidationResult


def train(
    run,
    agent: DQNAgent,
    train_env: CompilerEnv,
    config: TrainConfig,
    train_benchmarks: list,
    val_benchmarks: dict,
    enable_validation: bool = True,
    enable_validation_logs: bool = False,
) -> None:
    validation_env = train_env.fork()
    train_env = RandomOrderBenchmarks(
        train_env,
        benchmarks=train_benchmarks,
        rng=np.random.default_rng(config.random_state),
    )
    rewards_history = []
    negative_rewards_history = []
    # используем среднее из средних геометрических по всем датасетам,
    # потому что есть неудобные датасеты, на которых среднее геометрическое почти всегда 0
    best_val_mean_geomean = 0

    for episode_i in range(config.episodes):
        train_env.reset()
        agent.episode_reset()
        episode_data = EpisodeData(remains=config.episode_length)

        observation_modifier = ObservationModifier(
            train_env, config.observation_modifiers, config.episode_length
        )
        base_observation = get_observation(train_env, config)
        observation = observation_modifier.modify(
            base_observation, episode_data.remains
        )
        prev_action = config.actions.index("noop")
        try:
            while (
                not episode_data.done
                and episode_data.actions_count < config.episode_length
                and episode_data.patience_count < config.patience
            ):
                step_result = episode_step(
                    env=train_env,
                    config=config,
                    agent=agent,
                    episode_data=episode_data,
                    observation=observation,
                    observation_modifier=observation_modifier,
                    forbidden_actions=episode_data.forbidden_actions,
                )

                agent.store_transition(
                    prev_action=prev_action,
                    action=step_result.action,
                    observation=observation,
                    reward=step_result.reward,
                    new_observation=step_result.new_observation,
                    done=step_result.done,
                )
                loss_value = agent.learn()
                episode_data.update_after_episode_step(
                    step_result=step_result,
                    loss_value=loss_value,
                )
                observation = step_result.new_observation
                prev_action = step_result.action
        except SessionNotFound as e:
            print(f"Warning! SessionNotFound error occured {e}", file=sys.stderr)
        agent.episode_done()
        rewards_history.append(episode_data.total_reward)
        negative_rewards_history.append(episode_data.total_negative_reward)
        average_rewards_sum = np.mean(rewards_history[-config.logging_history_size :])
        average_negative_rewards_sum = np.mean(
            negative_rewards_history[-config.logging_history_size :]
        )
        std_rewards_sum = np.std(rewards_history[-config.logging_history_size :])
        _log_episode_results(
            run=run,
            env=train_env,
            epsilon=agent.get_epsilon(),
            episode_i=episode_i,
            average_rewards_sum=average_rewards_sum,
            average_negative_rewards_sum=average_negative_rewards_sum,
            std_rewards_sum=std_rewards_sum,
            episode_data=episode_data,
        )
        if episode_i % config.validation_interval == 0 and enable_validation:
            best_val_mean_geomean = _validation_during_train(
                run=run,
                episode_i=episode_i,
                best_val_mean_geomean=best_val_mean_geomean,
                agent=agent,
                env=validation_env,
                config=config,
                val_benchmarks=val_benchmarks,
                enable_logs=enable_validation_logs,
            )

    if (config.episodes - 1) % config.validation_interval != 0 and enable_validation:
        _validation_during_train(
            run=run,
            episode_i=config.episodes - 1,
            best_val_mean_geomean=best_val_mean_geomean,
            agent=agent,
            env=validation_env,
            config=config,
            val_benchmarks=val_benchmarks,
            enable_logs=enable_validation_logs,
        )
    save_model(
        state_dict=agent.get_policy_net_state_dict(), model_name=run.name, replace=False
    )


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
        agent=agent,
        env=env,
        config=config,
        val_benchmarks=val_benchmarks,
        enable_logs=enable_logs,
        use_actions_masking=config.eval_with_forbidden_actions,
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
        save_model(agent.get_policy_net_state_dict(), f"{run.name}")
        return validation_result.mean_geomean_reward
    return best_val_mean_geomean


def _log_episode_results(
    run,
    env,
    epsilon: float,
    episode_i: int,
    average_rewards_sum: float,
    average_negative_rewards_sum: float,
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
            "average_negative_rewards_sum_for_last_episodes": average_negative_rewards_sum,
            "std_rewards_sum_for_last_episodes": std_rewards_sum,
            "average_episode_loss": np.mean(episode_data.losses or [0]),
            "total_episode_reward": episode_data.total_reward,
            "episode_length": episode_data.actions_count,
        },
        step=episode_i,
    )


def validate(
    agent,
    env,
    config: TrainConfig,
    val_benchmarks: dict[str, list],
    enable_logs: bool = False,
    use_actions_masking: bool = False,
) -> ValidationResult:
    rewards = {}
    times = []
    binned_statistic_data = {}
    for dataset_name, benchmarks in val_benchmarks.items():
        codesize = []
        rewards[dataset_name] = []
        for i, benchmark in enumerate(benchmarks):
            env.reset(benchmark=benchmark)
            codesize.append(env.observation["IrInstructionCount"])
            with Timer() as timer:
                episode_data = rollout(
                    agent, env, config, use_actions_masking=use_actions_masking
                )
            rewards[dataset_name].append(episode_data.total_reward)
            times.append(timer.time)
            if enable_logs:
                print(
                    f"{i} - {benchmark} - reward: {episode_data.total_reward} - time: {timer.time} - actions: {' '.join(episode_data.chosen_flags)}"
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
def rollout(
    agent: DQNAgent, env, config: TrainConfig, use_actions_masking
) -> EpisodeData:
    agent.episode_reset()
    base_observation = get_observation(env, config)
    episode_data = EpisodeData(remains=config.episode_length)
    observation_modifier = ObservationModifier(
        env, config.observation_modifiers, config.episode_length
    )
    observation = observation_modifier.modify(base_observation, episode_data.remains)
    while (
        not episode_data.done
        and episode_data.actions_count < config.episode_length
        and episode_data.patience_count < config.val_patience
    ):
        if use_actions_masking:
            step_result = episode_step(
                env,
                config,
                agent,
                episode_data,
                observation,
                observation_modifier,
                episode_data.forbidden_actions,
            )
        else:
            step_result = episode_step(
                env, config, agent, episode_data, observation, observation_modifier
            )
        episode_data.update_after_episode_step(
            step_result=step_result,
            loss_value=None,
        )

    return episode_data


def episode_step(
    env: CompilerEnv,
    config: TrainConfig,
    agent: DQNAgent,
    episode_data: EpisodeData,
    observation: np.ndarray,
    observation_modifier: ObservationModifier,
    forbidden_actions: Optional[set[int]] = None,
    enable_epsilon_greedy: bool = True,
) -> StepResult:
    if forbidden_actions is not None:
        action = agent.choose_action(
            observation,
            enable_epsilon_greedy=enable_epsilon_greedy,
            forbidden_actions=episode_data.forbidden_actions,
        )
    else:
        action = agent.choose_action(
            observation, enable_epsilon_greedy=enable_epsilon_greedy
        )
    flags = config.actions[action]
    if flags == "noop":
        flags = [flags]
        reward, done, info = 0, False, {"action_had_no_effect": True}
    elif flags == "terminate":
        flags = [flags]
        reward, done, info = 0, True, {"action_had_no_effect": True}
    else:
        if " " in flags:
            flags = flags.split()
        else:
            flags = [flags]
        _, reward, done, info = env.multistep(
            [env.action_space.flags.index(f) for f in flags]
        )
    base_observation = get_observation(env, config)
    observation = observation_modifier.modify(base_observation, episode_data.remains)
    return StepResult(
        action=action,
        reward=reward,
        new_observation=observation,
        new_base_observation=base_observation,
        flags=flags,
        info=info,
        done=done,
    )
