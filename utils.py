import itertools
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import compiler_gym
import numpy as np
import plotly.graph_objects as go
import torch
from compiler_gym.datasets import FilesDataset
from compiler_gym.envs import CompilerEnv, LlvmEnv
from sklearn.model_selection import train_test_split

import wandb
from config.config import (
    MODELS_DIR,
    WANDB_PROJECT_NAME,
    TrainConfig,
    TEST_BENCHMARKS_DIR,
    LLVM_TEST_SUITE_DATASET_PATH,
)
from dqn.dqn import DoubleDQNAgent, DQNAgent, LstmDQNAgent, SimpleDQNAgent, TwinDQNAgent
from observation.utils import ObservationModifier


@dataclass
class BinnedStatistic:
    mean: np.ndarray
    std: np.ndarray
    bin_edges: np.ndarray
    binnumber: np.ndarray


@dataclass
class ValidationResult:
    geomean_reward: float
    mean_reward: float
    mean_walltime: float
    mean_runtime_reward: float = None
    step_reward_hist: wandb.Histogram = None
    step_reward_hist_std: wandb.Histogram = None


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
        # reward_space=config.reward_space,
    )


def fix_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def prepare_datasets(
    run_name,
    env: LlvmEnv,
    random_state: int,
    config: TrainConfig,
) -> tuple[list, list, list]:
    if (
        config.dataset == "llvm_test_suite_benchmarks"
        or config.llvm_test_suite_runtime_validation
    ):
        files = os.listdir(LLVM_TEST_SUITE_DATASET_PATH)
        benchmarks = [
            env.make_benchmark(os.path.join(LLVM_TEST_SUITE_DATASET_PATH, f))
            for f in files
        ]
        benchmarks = [
            b
            for b in benchmarks
            if str(b).split("/")[-1].split(".")[0] not in config.skipped_benchmarks
        ]
        print(f"benchmarks number: {len(benchmarks)}")
        if config.dataset == "llvm_test_suite_benchmarks":
            train_llvm, val_llvm = train_test_split(
                benchmarks, test_size=config.val_size, random_state=random_state
            )
        else:
            train_llvm, val_llvm = train_test_split(
                benchmarks, test_size=0.2, random_state=random_state
            )
        if config.dataset == "llvm_test_suite_benchmarks":
            return list(train_llvm), list(val_llvm), []

    # train_dataset_name = "benchmark://anghabench-v1"
    dataset_name = config.dataset
    # dataset_size = 7000
    # test_dataset_name = "benchmark://cbench-v1"
    benchmarks = list(env.datasets[dataset_name].benchmarks())
    # benchmarks = list(
    #     itertools.islice(env.datasets[dataset_name].benchmarks(), dataset_size)
    # )
    benchmarks, test = train_test_split(
        benchmarks, test_size=0.25, random_state=random_state + 10
    )
    # benchmarks = benchmarks[:dataset_size]
    os.makedirs(TEST_BENCHMARKS_DIR, exist_ok=True)
    with open(
        os.path.join(TEST_BENCHMARKS_DIR, f"test_benchmarks_{run_name}.txt"), "w"
    ) as ouf:
        ouf.write(
            "\n".join(
                [str(benchmark).rsplit("/", maxsplit=1)[-1] for benchmark in test]
            )
        )
    train, val = train_test_split(
        benchmarks, test_size=config.val_size, random_state=random_state
    )
    # test = env.datasets[test_dataset_name]
    # train = env.datasets[test_dataset_name]
    # val = env.datasets[test_dataset_name]
    if config.llvm_test_suite_runtime_validation:
        return list(train), list(val_llvm), []
    return list(train), list(val), []


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


# def optimize_with_model_raw(
#     config: TrainConfig,
#     agent: DQNAgent,
#     ir: str,
#     eval_mode,
#     iters=10,
#     print_debug=True,
# ) -> list[str]:
#     flags = []
#     prev_obs = np.zeros((config.observation_size,))
#     agent.episode_reset()
#
#     observation_modifier = ObservationModifier(
#         None, config.observation_modifiers, config.episode_length
#     )
#
#     with tempfile.NamedTemporaryFile("rw", delete=False) as ll_file:
#         ll_file.write(ir)
#
#     for i in range(iters):
#         base_observation = get_observation_from_cg(env, config.observation_space)
#         obs = observation_modifier.modify(
#             base_observation, config.episode_length - i, ir=env.observation["Ir"]
#         )
#         # assert np.any(prev_obs != obs)
#         action, value = 0, 1
#         # agent.choose_action(
#         # obs,
#         # enable_epsilon_greedy=False,
#         # forbidden_actions=set(),
#         # eval_mode=eval_mode,
#         # ))
#
#         if config.task == "subset":
#             if config.actions[action] == "apply":
#                 cur_flags = config.actions_sequence[i].split()
#             elif config.actions[action] == "skip":
#                 cur_flags = []
#             else:
#                 raise Exception("Unknown subset action")
#         else:
#             if value <= 0:
#                 break
#             cur_flags = config.actions[action].split()
#         if len(cur_flags) == 0:
#             pass
#         elif len(cur_flags) > 0 and cur_flags[0] == "noop":
#             pass
#         elif len(cur_flags) > 1:
#             env.multistep([env.action_space.flags.index(f) for f in cur_flags])
#         else:
#             env.step(env.action_space.flags.index(cur_flags[0]))
#
#         if print_debug:
#             if "mca" in config.observation_modifiers:
#                 print(
#                     f"{config.actions[action]}({action}, {value}, {obs[:3]})",
#                     end=" ",
#                 )
#             else:
#                 print(f"{config.actions[action]}({action}, {value})", end=" ")
#         flags.append(config.actions[action])
#         prev_obs = obs
#         if len(cur_flags) > 0 and cur_flags[0] == "noop":
#             break
#
#     if print_debug:
#         print()
#     return flags


def optimize_with_model(
    config: TrainConfig,
    agent: DQNAgent,
    env: CompilerEnv,
    eval_mode,
    iters=10,
    print_debug=True,
    hack: bool = False,
) -> list[str]:
    if print_debug:
        print(f"Eval mode: {eval_mode} - hack: {hack}")
    flags = []
    prev_obs = np.zeros((config.observation_size,))
    agent.episode_reset()

    observation_modifier = ObservationModifier(
        None, config.observation_modifiers, config.episode_length
    )
    for i in range(iters):
        base_observation = get_observation_from_cg(env, config.observation_space)
        obs = observation_modifier.modify(
            base_observation, config.episode_length - i, ir=env.observation["Ir"]
        )
        # assert np.any(prev_obs != obs)
        # action, value = 0, 1
        action, value = agent.choose_action(
            obs,
            enable_epsilon_greedy=False,
            forbidden_actions=set(),
            eval_mode=eval_mode,
            hack=hack,
        )
        if config.task == "subset":
            if config.actions[action] == "apply":
                cur_flags = config.actions_sequence[i].split()
            elif config.actions[action] == "skip":
                cur_flags = []
            else:
                raise Exception("Unknown subset action")
        else:
            if value <= 0:
                break
            cur_flags = config.actions[action].split()
        if len(cur_flags) == 0:
            pass
        elif len(cur_flags) > 0 and cur_flags[0] == "noop":
            pass
        elif len(cur_flags) > 1:
            env.multistep([env.action_space.flags.index(f) for f in cur_flags])
        else:
            env.step(env.action_space.flags.index(cur_flags[0]))

        if print_debug:
            if "mca" in config.observation_modifiers:
                print(
                    f"{config.actions[action]}({action}, {value}, {obs[:3]})", end=" "
                )
            else:
                print(f"{config.actions[action]}({action}, {value})", end=" ")
        flags.append(config.actions[action])
        prev_obs = obs
        if len(cur_flags) > 0 and cur_flags[0] == "noop":
            break
    if print_debug:
        print()
    return flags


def get_observation_from_cg(env: CompilerEnv, obs_name: str) -> np.ndarray:
    if obs_name == "IR2Vec":
        return get_ir2vec(env.observation["Ir"])
    elif obs_name == "IR2Vec+InstCountNorm+AutophaseNorm":
        return np.concatenate(
            [
                get_ir2vec(env.observation["Ir"]),
                env.observation["InstCountNorm"],
                env.observation["Autophase"] / env.observation["IrInstructionCount"],
            ]
        )
    elif obs_name == "InstCountNorm+AutophaseNorm":
        autophase = env.observation["Autophase"]
        return np.concatenate(
            [
                env.observation["InstCountNorm"],
                autophase / autophase[-5],
            ]
        )
    else:
        return env.observation[obs_name]


# def get_observation_from_ll(filepath, obs_name: str) -> np.ndarray:
#     if obs_name == "IR2Vec":
#         return get_ir2vec(env.observation["Ir"])
#     elif obs_name == "IR2Vec+InstCountNorm+AutophaseNorm":
#         autophase = env.observation["Autophase"]
#         return np.concatenate(
#             [
#                 get_ir2vec(env.observation["Ir"]),
#                 env.observation["InstCountNorm"],
#                 autophase / autophase[-5],
#             ]
#         )
#     elif obs_name == "InstCountNorm+AutophaseNorm":
#         autophase = env.observation["Autophase"]
#         return np.concatenate(
#             [
#                 env.observation["InstCountNorm"],
#                 autophase / autophase[-5],
#             ]
#         )
#     else:
#         return env.observation[obs_name]


def _get_obs(env, obs_name):
    if obs_name == "IR2Vec":
        return get_ir2vec(env.observation["Ir"])
    else:
        return env.observation[obs_name]


def get_ir2vec(ir_text: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile("w") as ll_file:
        ll_file.write(ir_text)
        ll_file.flush()
        return get_ir2vec_from_file(ll_file.name)


def get_ir2vec_from_file(filepath: str) -> np.ndarray:
    ir2vec_bin = "/home/flint/diplom/IR2Vec/build/bin/ir2vec"
    seed_emb_path = (
        "/home/flint/diplom/IR2Vec/vocabulary/seedEmbeddingVocab-300-llvm10.txt"
    )
    with tempfile.NamedTemporaryFile("r") as result_file:
        proc = subprocess.run(
            [
                ir2vec_bin,
                "-fa",
                "-vocab",
                seed_emb_path,
                "-o",
                result_file.name,
                "-level",
                "p",
                filepath,
            ],
            capture_output=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise Exception("IR2Vec failed")
        observation = np.loadtxt(result_file.name)
    observation = observation / np.linalg.norm(observation)
    return observation
