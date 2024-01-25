from itertools import islice

# noinspection PyUnresolvedReferences
import compiler_gym
import gym
import torch
from compiler_gym.wrappers import RandomOrderBenchmarks

import wandb
from dqn import train, Agent

config = dict(
    # TODO сделать нормально отслеживание данных для обучения
    train_benchmarks="benchmark://anghabench-v1",
    validation_benchmarks="benchmark://anghabench-v1",
    algorithm="DQN",
    compiler_gym_env="llvm-ic-v0",
    observation_space="InstCountNorm",
    reward_space="IrInstructionCountOz",
    gamma=0.90,  # The percent of how often the actor stays on policy
    epsilon=1.0,  # The starting value for epsilon
    epsilon_end=0.05,  # The ending value for epsilon
    epsilon_dec=5e-5,  # The decrement value for epsilon
    alpha=0.001,  # The learning rate
    batch_size=32,  # The batch size
    max_mem_size=100000,  # The maximum memory size
    replace=500,  # The number of iterations to run before replacing target network
    fc_dim=128,  # The dimension of a fully connected layer
    episodes=4000,  # The number of episodes used to learn
    episode_length=100,  # The (MAX) number of transformation passes per episode
    patience=5,  # The (MAX) number of times to apply a series of transformations without observable change
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
)


class CompilerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.patience = 5
        self.reward_counter = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if reward <= 0:
            self.reward_counter += 1
        else:
            self.reward_counter = 0

        if self.reward_counter > self.patience:
            done = True
        return next_state, reward, done, info


if __name__ == "__main__":
    run = wandb.init(
        project="rl-compilers-experiments",
        config=config,
    )
    env = gym.make(
        wandb.config["compiler_gym_env"],
        observation_space=wandb.config["observation_space"],
        reward_space=wandb.config["reward_space"],
    )
    train_benchmarks = list(
        islice(env.datasets[wandb.config["train_benchmarks"]].benchmarks(), 1000)
    )
    env = RandomOrderBenchmarks(
        env=env,
        benchmarks=train_benchmarks,
    )
    wandb.config["train_benchmarks_size"] = len(train_benchmarks)

    env = CompilerWrapper(env)
    wandb.config["observation_space_shape"] = env.observation_space.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(
        wandb.config,
        input_dims=env.observation_space.shape,
        n_actions=len(wandb.config["actions"]),
        device=device,
    )

    train(run, agent, env)
    # TODO валидация
