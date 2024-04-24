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

from config.config import TrainConfig
from dqn.dqn import DQNAgent
from dqn.train_utils import EpisodeData, StepResult, TrainHistory
from observation.utils import get_observation, ObservationModifier
from utils import save_model, ValidationResult
from env.performance_optimization import MyEnv, LlvmMcaEnv, CgLlvmMcaEnv, CfgGridEnv


def train(
    run,
    agent: DQNAgent,
    train_env: CompilerEnv,
    config: TrainConfig,
    train_benchmarks: list,
    val_benchmarks: list,
    enable_validation: bool,
    enable_validation_logs: bool = False,
) -> None:
    validation_env = train_env.fork()
    train_env = RandomOrderBenchmarks(
        train_env,
        benchmarks=train_benchmarks,
        rng=np.random.default_rng(config.random_state),
    )
    # mca_train_env = CgLlvmMcaEnv(config, train_env)
    # mca_validation_env = CgLlvmMcaEnv(config, validation_env)

    cfg_train_env = CfgGridEnv(config, train_env)
    cfg_validation_env = CfgGridEnv(config, validation_env)

    train_history = TrainHistory(logging_history_size=config.logging_history_size)

    for episode_i in range(config.episodes):
        cfg_train_env.reset()
        agent.episode_reset()
        episode_data = EpisodeData(remains=config.episode_length)

        observation_modifier = ObservationModifier(
            train_env, config.observation_modifiers, config.episode_length
        )
        # base_observation = get_observation(train_env, config)
        base_observation = cfg_train_env.get_observation(config.observation_space)
        # observation = observation_modifier.modify(
        #     base_observation, episode_data.remains
        # )
        observation = base_observation
        prev_action = 0
        if "noop" in config.special_actions:
            prev_action = config.actions.index("noop")
        try:
            while (
                not episode_data.done
                and episode_data.actions_count < config.episode_length
                # and episode_data.patience_count < config.patience
            ):
                step_result = episode_step(
                    env=cfg_train_env,
                    config=config,
                    agent=agent,
                    remains_steps=episode_data.remains,
                    observation=observation,
                    observation_modifier=observation_modifier,
                    forbidden_actions=set(),
                    enable_epsilon_greedy=True,
                    eval_mode=False,
                )
                loss_value = agent.learn()
                episode_data.update_after_episode_step(
                    step_result=step_result,
                    loss_value=loss_value,
                )
                done = (
                    step_result.done
                    or episode_data.actions_count >= config.episode_length - 1
                    # or episode_data.patience_count > config.patience
                )
                train_reward = step_result.reward
                train_reward *= config.reward_scale
                if train_reward < -14.767:
                    train_reward = -np.emath.logn(1.2, -train_reward)
                agent.store_transition(
                    prev_action=prev_action,
                    action=step_result.action,
                    observation=observation,
                    reward=train_reward,
                    new_observation=step_result.new_observation,
                    done=done,
                )
                observation = step_result.new_observation
                prev_action = step_result.action
        except SessionNotFound as e:
            print(f"Warning! SessionNotFound error occured {e}", file=sys.stderr)
        agent.episode_done()
        train_history.update(episode_data)

        _log_episode_results(
            run=run,
            env=train_env,
            epsilon=agent.get_epsilon(),
            episode_i=episode_i,
            train_history=train_history,
            episode_data=episode_data,
        )
        if (
            episode_i % config.validation_interval == 0
            or episode_i == config.episodes - 1
        ) and enable_validation:
            train_history.best_val_mean = _validation_during_train(
                run=run,
                episode_i=episode_i,
                best_val_mean=train_history.best_val_mean,
                agent=agent,
                env=cfg_validation_env,
                config=config,
                val_benchmarks=val_benchmarks,
                enable_logs=enable_validation_logs,
            )
    save_model(
        state_dict=agent.get_policy_net_state_dict(), model_name=run.name, replace=False
    )
    validation_env.close()


def _validation_during_train(
    run,
    episode_i: int,
    best_val_mean: float,
    agent: DQNAgent,
    env: MyEnv,
    config: TrainConfig,
    val_benchmarks: list,
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
    log_data = {}
    log_data["val_geomean_reward"] = validation_result.geomean_reward
    log_data["val_mean_reward"] = validation_result.mean_reward
    run.log(
        log_data,
        step=episode_i,
    )
    if validation_result.mean_reward > best_val_mean:
        print(
            f"Save model. New best mean: {validation_result.mean_reward},"
            f" previous best mean: {best_val_mean}"
        )
        save_model(agent.get_policy_net_state_dict(), f"{run.name}", replace=True)
        return validation_result.mean_reward
    return best_val_mean


def _log_episode_results(
    run,
    env,
    epsilon: float,
    episode_i: int,
    train_history: TrainHistory,
    episode_data: EpisodeData,
) -> None:
    average_rewards_sum = train_history.get_average_rewards_sum()
    average_negative_rewards_sum = train_history.get_average_negative_rewards_sum()
    std_rewards_sum = train_history.get_rewards_std()
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
            "mean_values": np.mean(episode_data.values),
        },
        step=episode_i,
    )


def validate(
    agent,
    env: MyEnv,
    config: TrainConfig,
    val_benchmarks: list,
    use_actions_masking: bool,
    enable_logs: bool = False,
) -> ValidationResult:
    times = []
    codesize = []
    rewards = []
    for i, benchmark in enumerate(val_benchmarks):
        env.reset(benchmark=benchmark)
        codesize.append(env._cg_env.observation["IrInstructionCount"])
        with Timer() as timer:
            episode_data = rollout(
                agent, env, config, use_actions_masking=use_actions_masking
            )
        rewards.append(episode_data.total_reward)
        times.append(timer.time)
        if enable_logs:
            print(
                f"{i} - {benchmark} - reward: {episode_data.total_reward} - time: {timer.time} - actions: {' '.join(episode_data.chosen_flags)}"
            )

    geomean_reward = geometric_mean(rewards)
    mean_reward = arithmetic_mean(rewards)
    mean_walltime = arithmetic_mean(times)
    return ValidationResult(
        geomean_reward=geomean_reward,
        mean_reward=mean_reward,
        mean_walltime=mean_walltime,
    )


@torch.no_grad()
def rollout(
    agent: DQNAgent, env, config: TrainConfig, use_actions_masking
) -> EpisodeData:
    agent.episode_reset()
    episode_data = EpisodeData(remains=config.episode_length)
    # base_observation = get_observation(env, config)
    base_observation = env.get_observation(config.observation_space)
    observation_modifier = ObservationModifier(
        env, config.observation_modifiers, config.episode_length
    )
    # observation = observation_modifier.modify(base_observation, episode_data.remains)
    observation = base_observation
    best_reward = float("-inf")
    best_sequence = []
    while (
        not episode_data.done
        and episode_data.actions_count < config.episode_length
        # and episode_data.patience_count < config.val_patience
    ):
        # if use_actions_masking:
        #     step_result = episode_step(
        #         env=env,
        #         config=config,
        #         agent=agent,
        #         remains_steps=episode_data.remains,
        #         observation=observation,
        #         observation_modifier=observation_modifier,
        #         forbidden_actions=episode_data.forbidden_actions,
        #         enable_epsilon_greedy=False,
        #         eval_mode=True,
        #     )
        # else:
        step_result = episode_step(
            env=env,
            config=config,
            agent=agent,
            remains_steps=episode_data.remains,
            observation=observation,
            observation_modifier=observation_modifier,
            enable_epsilon_greedy=False,
            forbidden_actions=None,
            eval_mode=False,
        )
        episode_data.update_after_episode_step(
            step_result=step_result,
            loss_value=None,
        )
        if episode_data.total_reward > best_reward:
            best_reward = episode_data.total_reward
            best_sequence.extend(step_result.flags)
    if config.eval_with_bestsequence:
        episode_data.total_reward = best_reward
        episode_data.chosen_flags = best_sequence
    return episode_data


def episode_step(
    env: MyEnv,
    config: TrainConfig,
    agent: DQNAgent,
    remains_steps: int,
    observation: np.ndarray,
    observation_modifier: ObservationModifier,
    forbidden_actions: Optional[set[int]],
    enable_epsilon_greedy: bool,
    eval_mode: bool,
) -> StepResult:
    action, value = agent.choose_action(
        observation,
        enable_epsilon_greedy=enable_epsilon_greedy,
        forbidden_actions=forbidden_actions,
        eval_mode=eval_mode,
    )
    flags = config.actions[action]
    # if flags == "noop":
    #     flags = [flags]
    #     reward, done, info = 0, False, {"action_had_no_effect": True}
    # elif flags == "terminate":
    #     flags = [flags]
    #     reward, done, info = 0, True, {"action_had_no_effect": True}
    # else:
    #     if " " in flags:
    #         flags = flags.split()
    #         _, reward, done, info = env.multistep(
    #             [env.action_space.flags.index(f) for f in flags]
    #         )
    #     else:
    reward = env.step(env._cg_env.action_space.flags.index(flags))
    flags = [flags]

    base_observation = env.get_observation(config.observation_space)
    # observation = observation_modifier.modify(base_observation, remains_steps)
    observation = base_observation
    return StepResult(
        action=action,
        reward=reward,
        value=value,
        new_observation=observation,
        new_base_observation=base_observation,
        flags=flags,
        info={},
        done=False,
    )
