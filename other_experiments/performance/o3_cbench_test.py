import gym
import pandas as pd
import numpy as np
from compiler_gym import CompilerEnv
from compiler_gym.errors import BenchmarkInitError
from tqdm import tqdm
from compiler_gym.wrappers import RuntimePointEstimateReward

from config.action_config import O3_SEQ


def apply_pass_sequence(env: CompilerEnv, pass_sequence):
    action_space_passes_set = set(env.action_space.flags)
    for pass_el in pass_sequence:
        if pass_el in action_space_passes_set:
            observation, reward, done, info = env.step(
                env.action_space.flags.index(pass_el)
            )
            # print(reward)


def main():
    env: CompilerEnv = gym.make("llvm-v0")
    # env = RuntimePointEstimateReward(
    #     env=env,
    # )
    benchmarks = list(env.datasets["benchmark://cbench-v1"].benchmarks())
    results = {
        "benchmark": [],
        "base_runtime": [],
        "o3_runtime": [],
    }

    for benchmark in tqdm(benchmarks):
        try:
            env.reset(benchmark=benchmark)
        except BenchmarkInitError:
            print(f"Benchmark {benchmark} not runnable, skip it")
            continue
        if not env.observation["IsRunnable"]:
            print(f"Benchmark {benchmark} not runnable, skip it")
            continue
        results["benchmark"].append(str(benchmark))
        results["base_runtime"].append(env.observation["Runtime"])
        apply_pass_sequence(env, O3_SEQ)
        results["o3_runtime"].append(env.observation["Runtime"])

    pd_results = pd.DataFrame(data=results)
    pd_results.to_csv("o3_cbench_test_results.csv")
    print(f'base_runtime mean: {np.mean(results["base_runtime"])}')
    print(f'o3_runtime mean: {np.mean(results["o3_runtime"])}')


if __name__ == "__main__":
    main()
