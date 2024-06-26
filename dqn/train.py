import json
import os
import sys
from subprocess import TimeoutExpired
from typing import Optional

import numpy as np
import torch
from compiler_gym.envs import CompilerEnv
from compiler_gym.errors import SessionNotFound
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer
from compiler_gym.wrappers.datasets import RandomOrderBenchmarks
from tqdm import tqdm

from config.action_config import _O23_SUBSEQ_CBENCH_MINS_O3
from config.config import TrainConfig
from dqn.dqn import DQNAgent
from dqn.train_utils import EpisodeData, StepResult, TrainHistory
from env.my_env import MyEnv
from env.performance_optimization.cfg_grind_env import CfgGridEnv, CfgGridSubsetEnv
from env.performance_optimization.mca_env import CgLlvmMcaEnv
from env.performance_optimization.runtime_env import RuntimeEnv
from observation.utils import ObservationModifier
from other_experiments.classifier.dataset import Dataset
from utils import save_model, ValidationResult


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
    runtime_validation_env = train_env.fork()
    prefill_env = RandomOrderBenchmarks(
        train_env.fork(),
        benchmarks=train_benchmarks,
        rng=np.random.default_rng(config.random_state + 100),
    )
    # RESUME:
    # from dqn.replay_buffer import ReplayBufferForLSTM
    # loaded = np.load("/home/flint/diplom/diplom_experiments/_FOR_RESUME_replay_buffer_cache/buffer_cache.npz")
    # agent._replay_buffer.from_npz_loaded(loaded, 15010)

    train_env = RandomOrderBenchmarks(
        train_env,
        benchmarks=train_benchmarks,
        rng=np.random.default_rng(config.random_state),
    )
    custom_train_env, custom_validation_env, custom_runtime_validation_env = _get_envs(
        config,
        train_env=train_env,
        validation_env=validation_env,
        runtime_validation_env=runtime_validation_env,
    )
    train_history = TrainHistory(logging_history_size=config.logging_history_size)

    observation_modifier = ObservationModifier(
        train_env, config.observation_modifiers, config.episode_length
    )

    if config.prefill > 0:
        cfg_prefill_env = CfgGridEnv(config, prefill_env)
        _prefill(cfg_prefill_env, config, agent, observation_modifier)
        prefill_env.close()

    dataset = Dataset(name=run.name)

    # RESUME set episode_i
    for episode_i in range(config.episodes):
        try:
            while True:
                reset_res = custom_train_env.reset()
                if reset_res:
                    break
        except Exception as e:
            print(f"train step failed skip it: {e}")
            continue
        agent.episode_reset()
        episode_data = EpisodeData(remains=config.episode_length)

        # base_observation = get_observation(train_env, config)
        base_observation = custom_train_env.get_observation(config.observation_space)
        observation = observation_modifier.modify(
            base_observation,
            episode_data.remains,
            env=custom_train_env,
        )
        x_data = observation.copy()
        # observation = base_observation
        prev_action = 0
        if "noop" in config.special_actions:
            prev_action = config.actions.index("noop")
        elif config.task == "subset":
            prev_action = config.actions.index("skip")
        try:
            while (
                not episode_data.done
                and episode_data.actions_count < config.episode_length
                # and episode_data.patience_count < config.patience
            ):
                step_result = episode_step(
                    env=custom_train_env,
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
                # if train_reward < -14.767:
                #     train_reward = -np.emath.logn(1.2, -train_reward)
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
        except TimeoutExpired as e:
            print(f"Timeout: {e}", file=sys.stderr)
        except Exception as e:
            print(f"train step failed skip it: {e}")

        y_data = custom_train_env.gather_data()
        if None not in y_data:
            dataset.add_example(
                x_data,
                np.array(y_data),
            )

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
        if episode_i % 100 == 0:
            dataset.save()
            cache_dir = "_replay_buffer_cache"
            # data_file = os.path.join(cache_dir, f"{run.name}")
            data_file = os.path.join(cache_dir, "buffer_cache")
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, "meta.json"), "w") as ouf:
                if config.algorithm.startswith("Lstm"):
                    ouf.write(
                        json.dumps(
                            {
                                "mem_counter": agent._replay_buffer._mem_counter,
                                "ready_data_size": agent._replay_buffer._ready_data_size,
                            }
                        )
                    )
                else:
                    ouf.write(
                        json.dumps(
                            {
                                "mem_counter": agent._replay_buffer._mem_counter,
                            }
                        )
                    )
            agent._replay_buffer.save_to_npz(data_file)
            save_model(
                state_dict=agent.get_policy_net_state_dict(),
                model_name=f"{episode_i}_{run.name}",
                replace=True,
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
                env=custom_validation_env,
                runtime_env=custom_runtime_validation_env,
                config=config,
                val_benchmarks=val_benchmarks,
                enable_logs=enable_validation_logs,
            )
    save_model(
        state_dict=agent.get_policy_net_state_dict(), model_name=run.name, replace=False
    )
    validation_env.close()
    runtime_validation_env.close()


def _get_envs(
    config: TrainConfig,
    train_env: CompilerEnv,
    validation_env: CompilerEnv,
    runtime_validation_env: CompilerEnv,
) -> tuple[MyEnv, MyEnv, MyEnv]:
    if config.task == "subset":
        if config.reward_space == "CfgInstructions":
            custom_train_env = CfgGridSubsetEnv(config, train_env)
            custom_validation_env = CfgGridSubsetEnv(config, validation_env)
            custom_runtime_validation_env = None
        else:
            raise NotImplementedError()
    elif config.task == "classic_phase_ordering":
        custom_runtime_validation_env = RuntimeEnv(config, runtime_validation_env)
        if config.reward_space == "MCA":
            custom_train_env = CgLlvmMcaEnv(config, train_env)
            custom_validation_env = CgLlvmMcaEnv(config, validation_env)
        elif config.reward_space == "CfgInstructions":
            custom_train_env = CfgGridEnv(config, train_env)
            custom_validation_env = CfgGridEnv(config, validation_env)
        elif config.reward_space == "Runtime":
            custom_train_env = RuntimeEnv(config, train_env)
            custom_validation_env = RuntimeEnv(config, validation_env)
        else:
            raise NotImplementedError()
        if config.llvm_test_suite_runtime_validation:
            custom_runtime_validation_env = RuntimeEnv(
                config, validation_env, llvm_test_suite_env=True
            )
    else:
        raise Exception("unknown reward space")
    return custom_train_env, custom_validation_env, custom_runtime_validation_env


def _prefill(
    prefill_env: MyEnv,
    config: TrainConfig,
    agent: DQNAgent,
    observation_modifier: ObservationModifier,
):
    prefill_dir = "_prefill_cache"
    prefill_data_file = os.path.join(
        prefill_dir, f"{config.observation_space}_prefill_o3"
    )
    os.makedirs(prefill_dir, exist_ok=True)
    if os.path.isfile(f"{prefill_data_file}.npz"):
        loaded = np.load(f"{prefill_data_file}.npz")
        agent._replay_buffer.from_npz_loaded(loaded, config.prefill)
        return
    action_seq = [len(config.special_actions) + el for el in _O23_SUBSEQ_CBENCH_MINS_O3]
    # o3_seq = [el for el in O3_SEQ if el in cfg_prefill_env._cg_env.action_space.flags]
    rewards = []
    for _ in tqdm(range(config.prefill)):
        try:
            prefill_env.reset()
        except Exception as e:
            print(f"train step failed skip it: {e}")
            continue
        agent.episode_reset()

        base_observation = prefill_env.get_observation(config.observation_space)
        observation = observation_modifier.modify(
            base_observation, config.episode_length, env=prefill_env
        )
        prev_action = 0
        if "noop" in config.special_actions:
            prev_action = config.actions.index("noop")
        remains = config.episode_length
        # choosen_flags = []
        reward_sum = 0
        try:
            for action in action_seq:
                # choosen_flags.append(config.actions[action])
                flags = config.actions[action]
                flags = flags.split()
                reward = prefill_env.step(flags)
                reward_sum += reward

                base_observation = prefill_env.get_observation(config.observation_space)
                new_observation = observation_modifier.modify(
                    base_observation, remains, env=prefill_env
                )
                remains -= 1
                agent.store_transition(
                    action,
                    observation=observation,
                    reward=reward,
                    new_observation=new_observation,
                    done=False,
                    prev_action=prev_action,
                )
                prev_action = action
                observation = new_observation

            # tmp_choosen_flags = []
            # for el in choosen_flags:
            #     tmp_choosen_flags.extend(el.split())
            # assert tmp_choosen_flags == o3_seq
            action = config.actions.index("noop")
            while remains > 0:
                agent.store_transition(
                    action,
                    observation=observation,
                    reward=0,
                    new_observation=observation,
                    done=remains < 2,
                    prev_action=prev_action,
                )
                prev_action = action
                remains -= 1
        except SessionNotFound as e:
            print(f"Warning! SessionNotFound error occured {e}", file=sys.stderr)
        except TimeoutExpired as e:
            print(f"Timeout: {e}", file=sys.stderr)
        except Exception as e:
            print(f"prefill step failed skip it: {e}")
        agent.episode_done()
        rewards.append(reward_sum)
        print(np.mean(rewards))
    agent._replay_buffer.save_to_npz(prefill_data_file)


def _validation_during_train(
    run,
    episode_i: int,
    best_val_mean: float,
    agent: DQNAgent,
    env: MyEnv,
    runtime_env: MyEnv,
    config: TrainConfig,
    val_benchmarks: list,
    enable_logs: bool = False,
) -> float:
    validation_result = validate(
        agent=agent,
        env=env,
        runtime_env=runtime_env,
        config=config,
        val_benchmarks=val_benchmarks,
        enable_logs=enable_logs,
        use_actions_masking=config.eval_with_forbidden_actions,
    )
    log_data = {}
    log_data["val_geomean_reward"] = validation_result.geomean_reward
    log_data["val_mean_reward"] = validation_result.mean_reward
    log_data["val_runtime_mean_reward"] = validation_result.mean_runtime_reward
    # log_data["val_reward_for_step"] = validation_result.step_reward_hist
    # log_data["val_reward_for_step_std"] = validation_result.step_reward_hist_std
    run.log(
        log_data,
        step=episode_i,
    )
    save_model(
        agent.get_policy_net_state_dict(), f"{episode_i}_{run.name}", replace=True
    )
    if validation_result.mean_reward > best_val_mean:
        print(
            f"Save model. New best mean: {validation_result.mean_reward},"
            f" previous best mean: {best_val_mean}"
        )
        save_model(
            agent.get_policy_net_state_dict(), f"best_val_{run.name}", replace=True
        )
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
    actions_str = " ".join(
        [
            f"{' '.join(flags)}({round(reward, 7)})"
            for flags, reward in zip(episode_data.chosen_flags, episode_data.rewards)
        ]
    )
    print(
        f"{episode_i} - {env.benchmark}\n"
        + "Total: {:.4f}".format(episode_data.total_reward)
        + " Total neg: {:.4f}".format(episode_data.total_negative_reward)
        + " Epsilon: {:.4f}".format(epsilon)
        + f" Average rewards sum: {average_rewards_sum}"
        + f" Action: {actions_str}"
    )
    # reward_hist, reward_hist_std = train_history.reward_hist_for_step()
    run.log(
        {
            "average_rewards_sum_for_last_episodes": average_rewards_sum,
            "average_negative_rewards_sum_for_last_episodes": average_negative_rewards_sum,
            "std_rewards_sum_for_last_episodes": std_rewards_sum,
            "average_episode_loss": np.mean(episode_data.losses or [0]),
            "total_episode_reward": episode_data.total_reward,
            "episode_length": episode_data.actions_count,
            "mean_values": np.mean(episode_data.values),
            # "reward_for_step": reward_hist,
            # "reward_for_step_std": reward_hist,
        },
        step=episode_i,
    )


def validate(
    agent,
    env: MyEnv,
    runtime_env: MyEnv,
    config: TrainConfig,
    val_benchmarks: list,
    use_actions_masking: bool,
    enable_logs: bool = False,
) -> ValidationResult:
    times = []
    # codesize = []
    rewards = []
    runtime_rewards = []
    train_history = TrainHistory(logging_history_size=config.logging_history_size)
    for i, benchmark in enumerate(val_benchmarks):
        # codesize.append(env._cg_env.observation["IrInstructionCount"])
        try:
            env.reset(benchmark=benchmark, val=True)
            runtime_env.reset(benchmark=benchmark, val=True)
            with Timer() as timer:
                episode_data = rollout(
                    agent, env, config, use_actions_masking=use_actions_masking
                )
                if env.is_runtime():
                    runtime_episode_data = episode_data
                else:
                    runtime_episode_data = rollout(
                        agent,
                        runtime_env,
                        config,
                        use_actions_masking=use_actions_masking,
                    )
            rewards.append(episode_data.total_reward)
            runtime_rewards.append(runtime_episode_data.total_reward)
            times.append(timer.time)
            train_history.update(episode_data)
            if enable_logs:
                # actions_str = " ".join(
                #     [
                #         f"{' '.join(flags)}({round(reward, 7)})"
                #         for flags, reward in zip(
                #             episode_data.chosen_flags, episode_data.rewards
                #         )
                #     ]
                # )
                print(
                    f"{i} - {benchmark} - reward: {episode_data.total_reward} - runtime_reward: {runtime_episode_data.total_reward} - time: {timer.time}"
                    # f"{i} - {benchmark} - reward: {episode_data.total_reward} - time: {timer.time} - actions: {actions_str}"
                )
        except TimeoutExpired as e:
            print(f"Timeout skip benchmark {str(benchmark)}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"benchmark {benchmark} validation failed skip it: {e}")

    geomean_reward = geometric_mean(rewards)
    mean_reward = arithmetic_mean(rewards)
    mean_walltime = arithmetic_mean(times)
    # step_reward_hist, step_reward_hist_std = train_history.reward_hist_for_step(
    #     log_size=len(val_benchmarks)
    # )
    return ValidationResult(
        geomean_reward=geomean_reward,
        mean_reward=mean_reward,
        mean_walltime=mean_walltime,
        mean_runtime_reward=arithmetic_mean(runtime_rewards),
        # step_reward_hist=step_reward_hist,
        # step_reward_hist_std=step_reward_hist_std,
    )


@torch.no_grad()
def rollout(
    agent: DQNAgent,
    env: MyEnv,
    config: TrainConfig,
    use_actions_masking,
    print_debug=True,
) -> EpisodeData:
    agent.episode_reset()
    episode_data = EpisodeData(remains=config.episode_length)
    # base_observation = get_observation(env, config)
    base_observation = env.get_observation(config.observation_space)
    observation_modifier = ObservationModifier(
        env, config.observation_modifiers, config.episode_length
    )
    observation = observation_modifier.modify(
        base_observation, episode_data.remains, env=env
    )
    # observation = base_observation
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
            eval_mode=True,
            ignore_reward=True,
        )
        if step_result.value <= 0:
            episode_data.done = True
        episode_data.update_after_episode_step(
            step_result=step_result,
            loss_value=None,
        )
        if episode_data.total_reward > best_reward:
            best_reward = episode_data.total_reward
            best_sequence.append(step_result.flags)
        if print_debug:
            if "mca" in config.observation_modifiers:
                print(
                    f"{config.actions[step_result.action]}({step_result.action}, {step_result.value}, {observation[:3]})",
                    end=" ",
                )
            else:
                print(
                    f"{config.actions[step_result.action]}({step_result.action}, {step_result.value})",
                    end=" ",
                )
    episode_data.total_reward = env.get_final_reward()
    if print_debug:
        print()
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
    ignore_reward=False,
) -> StepResult:
    action, value = agent.choose_action(
        observation,
        enable_epsilon_greedy=enable_epsilon_greedy,
        forbidden_actions=forbidden_actions,
        eval_mode=eval_mode,
    )
    flags = config.actions[action]
    flags = flags.split()
    if ignore_reward:
        reward = env.step_ignore_reward(flags)
    else:
        reward = env.step(flags)

    base_observation = env.get_observation(config.observation_space)
    observation = observation_modifier.modify(base_observation, remains_steps, env=env)
    # observation = base_observation
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
