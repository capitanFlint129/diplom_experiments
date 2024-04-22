import gym
import pandas as pd
import numpy as np
from compiler_gym import CompilerEnv
from compiler_gym.errors import BenchmarkInitError
from tqdm import tqdm
from compiler_gym.wrappers import RuntimePointEstimateReward
import compiler_gym

from config.action_config import O3_SEQ, O2_SEQ, O1_SEQ


def apply_pass_sequence(env: CompilerEnv, pass_sequence):
    action_space_passes_set = set(env.action_space.flags)
    for pass_el in pass_sequence:
        if pass_el in action_space_passes_set:
            observation, reward, done, info = env.step(
                env.action_space.flags.index(pass_el)
            )
            # print(reward)


# def get_runtime(env: CompilerEnv, n=10):
#     runtimes = []
#     for i in range(n):
#         runtimes.append(env.observation["Runtime"][0])
#     print(np.mean(runtimes), np.std(runtimes))
#     return np.mean(runtimes)


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
        "o3_speedup": [],
    }
    
    runtime_count = 30
    for benchmark in tqdm(benchmarks):
        with compiler_gym.make("llvm-v0", benchmark=benchmark) as new_env:
        # try:
            # env.reset(benchmark=benchmark)
        # except BenchmarkInitError:
            # print(f"Benchmark {benchmark} not runnable, skip it")
            # continue
        # results["benchmark"].append(str(benchmark))
        # results["base_runtime"].append(get_runtime(env))
            new_env.reset()

            if not new_env.observation["IsRunnable"]:
                print(f"Benchmark {benchmark} not runnable, skip it")
                continue

            results["benchmark"].append(str(benchmark))


            new_env.runtime_observation_count = runtime_count
            new_env.runtime_warmup_count = 0
            # new_env.apply(env.state)
            final_runtimes = new_env.observation.Runtime()
            assert len(final_runtimes) == runtime_count

            new_env.reset()
            # apply_pass_sequence(new_env, O1_SEQ)
            # new_env.send_param("llvm.apply_baseline_optimizations", "-O1")
            o3_runtimes = new_env.observation.Runtime()
            assert len(o3_runtimes) == runtime_count
            
            speedup = np.median(o3_runtimes) / max(np.median(final_runtimes), 1e-12)

            results["base_runtime"].append(np.median(final_runtimes))
            results["o3_runtime"].append(np.median(o3_runtimes))
            results["o3_speedup"].append(speedup)

    pd_results = pd.DataFrame(data=results)
    pd_results.to_csv("o3_cbench_test_results.csv")
    print(f'base_runtime mean: {np.mean(results["base_runtime"])}')
    print(f'o3_runtime mean: {np.mean(results["o3_runtime"])}')
    print(f'o3_speedup mean: {np.mean(results["o3_speedup"])}')


if __name__ == "__main__":
    main()
