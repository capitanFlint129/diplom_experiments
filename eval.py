import torch
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer

from dqn import rollout, Agent
from train import make_env, config

if __name__ == "__main__":
    env = make_env()
    train_benchmarks = env.datasets[config["eval_benchmarks"]]

    agent = Agent(input_dims=[125], n_actions=len(config["actions"]))
    agent.Q_eval.load_state_dict(torch.load("models/mild-shadow-94.pth"))

    rewards = []
    times = []
    for benchmark in train_benchmarks:
        env.reset(benchmark=benchmark)
        with Timer() as timer:
            reward = rollout(agent, env)
        print(f"{benchmark} - {reward} - {timer.time}")
        rewards.append(reward)
        times.append(timer.time)
    print(f"Geomean reward: {geometric_mean(rewards)}")
    print(f"Mean walltime: {arithmetic_mean(times)}")


# Current mean walltime: 0.707s / benchmark.
# Current geomean reward: 1.0146.
