from itertools import islice

# noinspection PyUnresolvedReferences
import compiler_gym
import ray
from compiler_gym.wrappers import (
    ConstrainedCommandline,
    TimeLimit,
    CycleOverBenchmarks,
)
from matplotlib import pyplot as plt
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from sklearn.model_selection import train_test_split

# import wandb
from train import config


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
    train_benchmarks = list(
        islice(env.datasets[config["train_benchmarks"]].benchmarks(), 10000)
    )
    train_benchmarks, val_benchmarks = train_test_split(
        train_benchmarks, test_size=0.15, random_state=config["random_state"]
    )
    test_benchmarks = list(env.datasets[config["test_benchmarks"]].benchmarks())
    return train_benchmarks, val_benchmarks, test_benchmarks


def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
    del args
    return CycleOverBenchmarks(make_env(), train_benchmarks)


def run_agent_on_benchmarks(benchmarks):
    with make_env() as env:
        rewards = []
        for i, benchmark in enumerate(benchmarks, start=1):
            observation, done = env.reset(benchmark=benchmark), False
            while not done:
                action = agent.compute_action(observation)
                observation, _, done, _ = env.step(action)
            rewards.append(env.episode_reward)
            print(f"[{i}/{len(benchmarks)}] {env.state}")

    return rewards


def plot_results(x, y, name, ax):
    plt.sca(ax)
    plt.bar(range(len(y)), y)
    plt.ylabel("Reward (higher is better)")
    plt.xticks(range(len(x)), x, rotation=90)
    plt.title(f"Performance on {name} set")


if __name__ == "__main__":
    with make_env() as env:
        train_benchmarks, val_benchmarks, test_benchmarks = prepare_datasets(env)

    if ray.is_initialized():
        ray.shutdown()
    ray.init(include_dashboard=False, ignore_reinit_error=True)
    tune.register_env("compiler_gym", make_training_env)
    analysis = tune.run(
        "PPO",
        checkpoint_at_end=True,
        stop={
            "episodes_total": 500,
        },
        config={
            "seed": 0xCC,
            "num_workers": 1,
            "env": "compiler_gym",
            "rollout_fragment_length": 5,
            "train_batch_size": 5,
            "sgd_minibatch_size": 5,
        },
    )

    # validation
    agent = PPO(
        env="compiler_gym",
        config={
            "num_workers": 1,
            "seed": 0xCC,
            "explore": False,
        },
    )

    checkpoint = analysis.get_best_checkpoint(
        metric="episode_reward_mean", mode="max", trial=analysis.trials[0]
    )

    agent.restore(checkpoint)

    val_rewards = run_agent_on_benchmarks(val_benchmarks)
    test_rewards = run_agent_on_benchmarks(test_benchmarks)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(13, 3)
    plot_results(val_benchmarks, val_rewards, "val", ax1)
    plot_results(test_benchmarks, test_rewards, "test", ax2)
    plt.show()
