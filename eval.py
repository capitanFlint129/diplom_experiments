import torch
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer

from dqn import rollout, Agent
from train import make_env, config, fix_seed
from utils import prepare_datasets

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(config)
    fix_seed(config["random_state"])
    _, _, test_benchmarks = prepare_datasets(
        env, config["datasets"], no_split=config["no_split"]
    )

    agent = Agent(
        input_dims=config["observation_space_shape"],
        n_actions=len(config["actions"]),
        config=config,
        device=device,
    )
    agent.Q_eval.load_state_dict(torch.load("models/wandering-bee-129.pth"))

    rewards = []
    times = []
    for benchmark in test_benchmarks:
        env.reset(benchmark=benchmark)
        with Timer() as timer:
            reward = rollout(agent, env, config)
        print(f"benchmark: {benchmark} - reward: {reward} - time: {timer.time}")
        rewards.append(reward)
        times.append(timer.time)
    env.close()
    print(f"Geomean reward: {geometric_mean(rewards)}")
    print(f"Mean walltime: {arithmetic_mean(times)}")
