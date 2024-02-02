import os

# noinspection PyUnresolvedReferences
import compiler_gym
import numpy as np
import torch
import wandb
from compiler_gym.wrappers import (
    CycleOverBenchmarks,
)

from agent import Agent
from utils import make_env, prepare_datasets

MODELS_DIR = "models"

config = dict(
    compiler_gym_env="llvm-v0",
    observation_spaces=["InstCountNorm"],
    reward_space="IrInstructionCountOz",
    datasets=[
        # "benchmark://mibench-v1",
        "benchmark://github-v0",
        # "benchmark://cbench-v1",
    ],
    dataset_limit=None,
    algorithm="DQN",
    # monitoring_baseline_observation_name="IrInstructionCountOz",
    # monitoring_observation_space="IrInstructionCount",
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
    learn=32,
    actions=[
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


def train(run, agent, env, config):
    history_size = 100
    mem_cntr = 0
    history = np.zeros(history_size)

    for i in range(1, config["episodes"] + 1):
        env.reset()
        observation = get_observation(
            [env.observation[space_name] for space_name in config["observation_spaces"]]
        )
        done = False
        total = 0
        actions_taken = 0
        agent.actions_taken = []
        change_count = 0

        while (
            not done
            and actions_taken < config["episode_length"]
            and change_count < config["patience"]
        ):
            action = agent.choose_action(observation)
            flag = config["actions"][action]
            new_observation, reward, done, info = env.step(
                env.action_space.flags.index(flag),
                observation_spaces=config["observation_spaces"],
            )
            new_observation = get_observation(new_observation)
            actions_taken += 1
            total += reward

            if reward == 0:
                change_count += 1
            else:
                change_count = 0

            agent.store_transition(action, observation, reward, new_observation, done)
            agent.learn()
            observation = new_observation

            print(
                "Step: "
                + str(i)
                + " Episode Total: "
                + "{:.4f}".format(total)
                + " Epsilon: "
                + "{:.4f}".format(agent.epsilon)
                + " Action: "
                + flag
            )
            if len(agent.actions_taken) == len(config["actions"]):
                done = True

        index = mem_cntr % history_size
        history[index] = total
        mem_cntr += 1

        print("Average sum of rewards is " + str(np.mean(history)))
        run.log(
            {
                "average_rewards_sum_last_100": np.mean(history),
                "total_episode_reward": total,
            }
        )
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    torch.save(agent.policy_net.state_dict(), f"./{MODELS_DIR}/{run.name}.pth")


def get_observation(observations):
    return np.stack(observations).squeeze()


if __name__ == "__main__":
    run = wandb.init(
        project="rl-compilers-experiments",
        config=config,
    )
    env = make_env(config)
    train_benchmarks, val_benchmarks, test_benchmarks = prepare_datasets(env, config)
    env = CycleOverBenchmarks(env, train_benchmarks)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_actions = env.action_space.n
    env.reset()
    observation = get_observation(
        [env.observation[space_name] for space_name in config["observation_spaces"]]
    )
    input_dims = observation.shape

    agent = Agent(
        input_dims=input_dims, n_actions=n_actions, config=config, device=device
    )
    train(run, agent, env, config)
    env.close()
