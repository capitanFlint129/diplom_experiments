import pandas as pd
import torch
from tqdm import tqdm

from config.config import TrainConfig, TEST_BENCHMARKS
from utils import (
    get_agent,
    get_model_path,
    optimize_with_model,
)
from utils import make_env

DATASET_URI = "benchmark://jotaibench-v0"
RUN_NAME = "peach-hill-96"


def main():
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(config)
    with open(TEST_BENCHMARKS + ".txt", "r") as inf:
        benchmarks_names = inf.readlines()
    optimizations = []
    final_benchmarks_names = []

    agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path(RUN_NAME),
    )

    for name in tqdm(benchmarks_names):
        name = name.strip()
        try:
            env.reset(benchmark=f"{DATASET_URI}/{name}")
        except ValueError as e:
            print(e)
            continue
        flags = optimize_with_model(config, agent, env, print_debug=False)
        optimizations.append(flags)
        final_benchmarks_names.append(name)
    result = pd.DataFrame(
        data={
            "benchmark": final_benchmarks_names,
            "optimizations": optimizations,
        }
    )
    result.to_csv("data/optimizations.csv")


if __name__ == "__main__":
    main()
