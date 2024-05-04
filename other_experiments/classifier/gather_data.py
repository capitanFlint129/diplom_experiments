import argparse
import sys
from subprocess import TimeoutExpired
from typing import Optional

# noinspection PyUnresolvedReferences
import compiler_gym
import numpy as np
import torch
from compiler_gym import CompilerEnv
from compiler_gym.errors import SessionNotFound
from compiler_gym.wrappers import RandomOrderBenchmarks
from tqdm import tqdm

from config.config import TrainConfig
from dqn.dqn import DQNAgent
from dqn.train_utils import EpisodeData, StepResult
from env.my_env import MyEnv
from env.performance_optimization.cfg_grind_env import CfgGridSubsetEnv, CfgGridEnv
from env.performance_optimization.mca_env import CgLlvmMcaEnv
from env.performance_optimization.runtime_env import RuntimeEnv
from observation.utils import ObservationModifier
from other_experiments.classifier.dataset import Dataset
from utils import (
    fix_seed,
    get_agent,
    make_env,
    prepare_datasets,
    get_model_path,
)


def _get_envs(
    config: TrainConfig,
    train_env: CompilerEnv,
) -> MyEnv:
    if config.task == "subset":
        if config.reward_space == "CfgInstructions":
            custom_train_env = CfgGridSubsetEnv(config, train_env)
        else:
            raise NotImplementedError()
    elif config.task == "classic_phase_ordering":
        if config.reward_space == "MCA":
            custom_train_env = CgLlvmMcaEnv(config, train_env)
        elif config.reward_space == "CfgInstructions":
            custom_train_env = CfgGridEnv(config, train_env)
        elif config.reward_space == "Runtime":
            custom_train_env = RuntimeEnv(config, train_env)
        else:
            raise NotImplementedError()
    else:
        raise Exception("unknown reward space")
    return custom_train_env


def _episode_step(
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
    flags = flags.split()
    reward = env.step_ignore_reward(flags)

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


def gather_data(
    agent: DQNAgent,
    train_env: CompilerEnv,
    config: TrainConfig,
    train_benchmarks: list,
    size: int,
) -> None:
    train_env = RandomOrderBenchmarks(
        train_env,
        benchmarks=train_benchmarks,
        rng=np.random.default_rng(config.random_state + 100),
    )
    custom_train_env = _get_envs(config, train_env)

    observation_modifier = ObservationModifier(
        train_env, config.observation_modifiers, config.episode_length
    )

    dataset = Dataset(name=args.run_name)

    names = ["O0", "O2", "O3", "model"]
    best = {"O0": 0, "O2": 0, "O3": 0, "model": 0}
    for episode_i in tqdm(range(size)):
        try:
            custom_train_env.reset()
        except Exception as e:
            print(f"train step failed skip it: {e}")
            continue
        agent.episode_reset()
        episode_data = EpisodeData(remains=config.episode_length)

        base_observation = custom_train_env.get_observation(config.observation_space)
        observation = observation_modifier.modify(
            base_observation,
            episode_data.remains,
            env=custom_train_env,
        )
        x_data = observation.copy()
        try:
            while (
                not episode_data.done
                and episode_data.actions_count < config.episode_length
            ):
                step_result = _episode_step(
                    env=custom_train_env,
                    config=config,
                    agent=agent,
                    remains_steps=episode_data.remains,
                    observation=observation,
                    observation_modifier=observation_modifier,
                    forbidden_actions=set(),
                    enable_epsilon_greedy=False,
                    eval_mode=True,
                )
                episode_data.update_after_episode_step(
                    step_result=step_result,
                    loss_value=None,
                )
                observation = step_result.new_observation
        except SessionNotFound as e:
            print(f"Warning! SessionNotFound error occured {e}", file=sys.stderr)
        except TimeoutExpired as e:
            print(f"Timeout: {e}", file=sys.stderr)
        except Exception as e:
            print(f"train step failed skip it: {e}")

        y_data = custom_train_env.gather_data(without_train=True)
        min_res = np.min(y_data)
        best_names = []
        for i, name in enumerate(names):
            if y_data[i] == min_res:
                best_names.append(name)
                best[name] += 1
        print(best_names, dict(zip(names, y_data)), best)
        if None not in y_data:
            dataset.add_example(
                x_data,
                np.array(y_data),
            )
        agent.episode_done()
        if episode_i % 100 == 0:
            dataset.save()
    dataset.save()
    


def main():
    config = TrainConfig.load_config(args.run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path(
            args.run_name if not args.best_val else f"best_val_{args.run_name}"
        ),
    )

    with make_env(config) as train_env:
        fix_seed(config.random_state)
        train_benchmarks, val_benchmarks, test_benchmarks = prepare_datasets(
            args.run_name,
            train_env,
            random_state=config.random_state,
        )
        gather_data(
            agent,
            train_env,
            config,
            train_benchmarks,
            size=args.size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name")
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--best_val",
        help="best_val",
        action="store_true",
    )
    args = parser.parse_args()

    main()
