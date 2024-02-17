# noinspection PyUnresolvedReferences
import compiler_gym
import torch

import wandb
from dqn.dqn import Agent
from dqn.train import train, validate
from utils import prepare_datasets, make_env, fix_seed

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
    lstm_hidden_size=128,  # The dimension of a fully connected layer
    action_embedding_size=50,
    episodes=4000,  # The number of episodes used to learn
    episode_length=10,  # The (MAX) number of transformation passes per episode
    patience=4,  # The (MAX) number of times to apply a series of transformations without observable change
    learn_memory_threshold=32,  # The number of fully exploratory episodes to run before starting learning
    # General section
    datasets=[
        "benchmark://cbench-v1",
        # "benchmark://mibench-v1",
        # "benchmark://opencv-v0",
    ],
    no_split=True,
    compiler_gym_env="llvm-v0",
    observation_space="InstCountNorm",
    observation_size=69,
    reward_space="IrInstructionCountOz",
    logging_history_size=100,
    actions=[
        "noop",
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        project="rl-compilers-experiments",
        config=config,
        # mode="disabled",
    )
    env = make_env(config)
    fix_seed(config["random_state"])
    train_benchmarks, val_benchmarks, test_benchmarks = prepare_datasets(
        env, config["datasets"], no_split=config["no_split"]
    )
    agent = Agent(
        observation_size=config["observation_size"],
        n_actions=len(config["actions"]),
        config=config,
        device=device,
    )
    train(
        run,
        agent,
        env,
        config,
        train_benchmarks,
        val_benchmarks,
        # enable_validation=False,
    )
    env.close()

    # final test
    env = make_env(config)
    agent = Agent(
        observation_size=config["observation_size"],
        n_actions=len(config["actions"]),
        config=config,
        device=device,
    )
    agent.Q_eval.load_state_dict(torch.load(f"models/{run.name}.pth"))
    agent.eval()
    with torch.no_grad():
        test_result = validate(agent, env, config, test_benchmarks)
    print(f"Test geomean: {test_result.geomean_reward}")
    run.summary["test_geomean_reward"] = test_result.geomean_reward
    run.summary["test_mean_walltime"] = test_result.mean_walltime
    for dataset_name, geomean_reward in test_result.geomean_reward_per_dataset.items():
        run.summary[f"test_geomean_reward_{dataset_name}"] = geomean_reward
    env.close()


if __name__ == "__main__":
    main()
