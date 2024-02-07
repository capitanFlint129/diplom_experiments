import random

# noinspection PyUnresolvedReferences
import compiler_gym
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import wandb
from dqn import train, Agent, validate

config = dict(
    # Algorithm section
    algorithm="DQN",
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
    episode_length=10,  # The (MAX) number of transformation passes per episode
    patience=4,  # The (MAX) number of times to apply a series of transformations without observable change
    learn_memory_threshold=32,  # The number of fully exploratory episodes to run before starting learning
    # General section
    datasets=["benchmark://cbench-v1"],
    compiler_gym_env="llvm-v0",
    observation_space="Autophase",
    observation_space_shape=[56],
    reward_space="IrInstructionCountOz",
    logging_history_size=100,
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
    return compiler_gym.make(
        config["compiler_gym_env"],
        reward_space=config["reward_space"],
    )


def fix_seed():
    np.random.seed(config["random_state"])
    torch.manual_seed(config["random_state"])
    random.seed(config["random_state"])


def prepare_datasets(env, no_split=False):
    if no_split:
        benchmarks = []
        for dataset_name in config["datasets"]:
            benchmarks.extend(list(env.datasets[dataset_name].benchmarks()))
        random.shuffle(benchmarks)
        return benchmarks, benchmarks, benchmarks
    train_benchmarks = []
    val_benchmarks = []
    test_benchmarks = []
    for dataset_name in config["datasets"]:
        benchmarks = list(env.datasets[dataset_name].benchmarks())
        train, test = train_test_split(
            benchmarks, test_size=0.2, random_state=config["random_state"]
        )
        train, val = train_test_split(
            benchmarks, test_size=0.125, random_state=config["random_state"]
        )
        train_benchmarks.extend(train)
        val_benchmarks.extend(val)
        test_benchmarks.extend(test)
    return train_benchmarks, val_benchmarks, test_benchmarks


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        project="rl-compilers-experiments",
        config=config,
    )
    env = make_env()
    fix_seed()
    train_benchmarks, val_benchmarks, test_benchmarks = prepare_datasets(
        env, no_split=True
    )
    agent = Agent(
        input_dims=config["observation_space_shape"],
        n_actions=len(config["actions"]),
        config=config,
        device=device,
    )
    train(run, agent, env, config, train_benchmarks, val_benchmarks)
    env.close()

    env = make_env()
    agent = Agent(
        input_dims=config["observation_space_shape"],
        n_actions=len(config["actions"]),
        config=config,
        device=device,
    )
    agent.Q_eval.load_state_dict(torch.load(f"models/{run.name}.pth"))
    agent.eval()
    with torch.no_grad():
        test_geomean, test_walltime = validate(agent, env, config, test_benchmarks)
    print(f"Test geomean: {test_geomean}")
    run.summary["test_geomean_reward"] = test_geomean
    run.summary["test_mean_walltime"] = test_walltime
    env.close()


if __name__ == "__main__":
    main()
