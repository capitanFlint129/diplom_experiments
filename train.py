import gym
from absl import app
import torch
import random
import gym

import numpy as np

# noinspection PyUnresolvedReferences
import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks
import wandb

from dqn import train, Agent
from sklearn.model_selection import train_test_split

config = dict(
    # TODO сделать нормально отслеживание данных для обучения
    train_benchmarks="benchmark://cbench-v1",
    eval_benchmarks="benchmark://cbench-v1",
    algorithm="DQN",
    compiler_gym_env="llvm-v0",
    # observation_space="InstCountNorm",
    observation_space=["InstCountNorm", "Autophase"],
    monitoring_baseline_observation_name="IrInstructionCountOz",
    monitoring_observation_space="IrInstructionCount",
    reward_space="IrInstructionCountOz",
    gamma=0.90,  # The percent of how often the actor stays on policy
    epsilon=1.0,  # The starting value for epsilon
    epsilon_end=0.05,  # The ending value for epsilon
    epsilon_dec=5e-5,  # The decrement value for epsilon
    alpha=0.001,  # The learning rate
    batch_size=32,  # The batch size
    max_mem_size=100000,  # The maximum memory size
    replace=500,  # The number of iterations to run before replacing target network
    fc_dim=256,  # The dimension of a fully connected layer
    episodes=4000,  # The number of episodes used to learn
    episode_length=100,  # The (MAX) number of transformation passes per episode
    patience=20,  # The (MAX) number of times to apply a series of transformations without observable change
    learn_memory_threshold=32,  # The number of fully exploratory episodes to run before starting learning
    actions=[  # A list of action names to explore from
        "-break-crit-edges",
        "-early-cse-memssa",
        "-gvn-hoist",
        "-gvn",
        "-instcombine",
        "-instsimplify",
        "-jump-threading",
        "-loop-reduce",
        "-loop-rotate",
        "-loop-versioning",
        "-mem2reg",
        "-newgvn",
        "-reg2mem",
        "-simplifycfg",
        "-sroa",
    ],
    random_state=42,
)


def make_env():
    return gym.make("llvm-ic-v0")


if __name__ == "__main__":
    run = wandb.init(
        project="rl-compilers-experiments",
    )
    env = make_env()
    np.random.seed(config["random_state"])
    torch.manual_seed(config["random_state"])
    random.seed(config["random_state"])
    train_benchmarks, _ = train_test_split(
        env.datasets[config["train_benchmarks"]],
        test_size=0,
        random_state=config["random_state"],
    )
    env = CycleOverBenchmarks(env, train_benchmarks)

    agent = Agent(input_dims=[125], n_actions=len(config["actions"]))
    train(run, agent, env, config)
    env.close()
