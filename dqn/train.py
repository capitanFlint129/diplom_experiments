import math
import random
from itertools import count
from itertools import islice

# noinspection PyUnresolvedReferences
import compiler_gym
import compiler_gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from compiler_gym.wrappers import (
    ConstrainedCommandline,
    TimeLimit,
    CycleOverBenchmarks,
)
from sklearn.model_selection import train_test_split

import wandb
from config import config
from dqn import DQN, ReplayMemory, Transition
from wrapper import PatienceWrapper

MODELS_DIR = "models"


def make_env() -> compiler_gym.envs.CompilerEnv:
    env = compiler_gym.make(
        config["compiler_gym_env"],
        observation_space=config["observation_space"],
        reward_space=config["reward_space"],
    )
    env = ConstrainedCommandline(
        env,
        flags=config["actions"],
    )
    env = TimeLimit(env, max_episode_steps=config["episode_length"])
    return env


def prepare_datasets(env: compiler_gym.envs.CompilerEnv) -> tuple:
    trains_size = 10000
    if config["train_benchmarks"].startswith("generator:"):
        train_benchmarks = []
        for i in range(trains_size):
            benchmark = env.datasets[config["train_benchmarks"]].random_benchmark()
            print(benchmark)
            train_benchmarks.append(benchmark)
            # state = env.reset(
            #     benchmark=env.datasets[config["train_benchmarks"]].random_benchmark()
            # )
    else:
        train_benchmarks = list(
            islice(env.datasets[config["train_benchmarks"]].benchmarks(), trains_size)
        )
    train_benchmarks, val_benchmarks = train_test_split(
        train_benchmarks, test_size=0.15, random_state=config["random_state"]
    )
    test_benchmarks = list(env.datasets[config["test_benchmarks"]].benchmarks())
    return train_benchmarks, val_benchmarks, test_benchmarks


def make_training_env():
    return PatienceWrapper(
        CycleOverBenchmarks(make_env(), train_benchmarks), patience=config["patience"]
    )


def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def evaluate(run, val_benchmarks, model):
    env = PatienceWrapper(CycleOverBenchmarks(make_env(), val_benchmarks))
    reward_sums = []
    for i in range(len(val_benchmarks)):
        state = env.reset()
        done = False
        flags = []
        reward_sum = 0
        while not done:
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = model(state).max(1).indices.view(1, 1)
            flag = config["actions"][action]
            flags.append(flag)
            state, reward, done, info = env.step(env.action_space.flags.index(flag))
            reward_sum += reward
        reward_sums.append(reward_sum)
    run.log({"val_reward_sum": np.mean(reward_sums)})
    env.close()
    return sum(reward_sums)


if __name__ == "__main__":
    with make_env() as env:
        train_benchmarks, val_benchmarks, test_benchmarks = prepare_datasets(env)

    with make_training_env() as env:
        run = wandb.init(
            project="rl-compilers-experiments",
            config=config,
        )
        wandb.config["train_benchmarks_size"] = len(train_benchmarks)
        wandb.config["observation_space_shape"] = env.observation_space.shape

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        BATCH_SIZE = 128
        GAMMA = 0.99
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        TAU = 0.005
        LR = 1e-4
        n_actions = env.action_space.n
        state = env.reset()
        input_dims = env.observation_space.shape

        policy_net = DQN(
            input_dims,
            n_actions,
            fc1_dims=config["fc_dim"],
            fc2_dims=config["fc_dim"],
            fc3_dims=config["fc_dim"],
        ).to(device)
        target_net = DQN(
            input_dims,
            n_actions,
            fc1_dims=config["fc_dim"],
            fc2_dims=config["fc_dim"],
            fc3_dims=config["fc_dim"],
        ).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)
        episode_lengths = []

        sum_rewards_history = []
        steps_done = 0
        best_val_rewards_sum = float("-inf")
        for i_episode in range(config["episodes"]):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            total_episode_reward = 0

            for t in count():
                action = select_action(state, steps_done)
                steps_done += 1
                observation, reward, terminated, _ = env.step(action.item())
                total_episode_reward += reward
                reward = torch.tensor([reward], device=device)
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

                memory.push(state, action, next_state, reward)
                state = next_state
                optimize_model()
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

                if terminated:
                    break

            sum_rewards_history.append(total_episode_reward)
            run.log(
                {
                    "average_rewards_sum_last_100": np.mean(sum_rewards_history[-100:]),
                    "total_episode_reward": total_episode_reward,
                    # "model_instruction_count": env.observation["IrInstructionCount"],
                    # "oz_instruction_count": env.observation["IrInstructionCountOz"],
                    # "instruction_count_baseline_diff": env.observation[
                    #     "IrInstructionCountOz"
                    # ]
                    # - env.observation["IrInstructionCount"],
                }
            )

            if i_episode % 200 == 0:
                path = f"./{MODELS_DIR}/{run.name}.pth"
                torch.save(policy_net.state_dict(), path)
                eval_net = DQN(
                    input_dims,
                    n_actions,
                    fc1_dims=config["fc_dim"],
                    fc2_dims=config["fc_dim"],
                    fc3_dims=config["fc_dim"],
                ).to(device)
                result = evaluate(run, val_benchmarks, eval_net)
                if result > best_val_rewards_sum:
                    path = f"./{MODELS_DIR}/{run.name}.pth"
                    torch.save(policy_net.state_dict(), path)
