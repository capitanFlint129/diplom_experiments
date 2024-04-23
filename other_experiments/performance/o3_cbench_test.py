import compiler_gym
import gym
import numpy as np
import pandas as pd
import torch
from compiler_gym import CompilerEnv
from tabulate import tabulate
from tqdm import tqdm

from config.action_config import COMPLETE_ACTION_SET
from config.config import TrainConfig
from dqn.dqn import DQNAgent
from env.performance_optimization import get_rblock_throughput_ir
from utils import (
    get_agent,
    get_model_path,
)


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


def optimize_with_model(config, agent: DQNAgent, env: CompilerEnv):
    prev_obs = np.zeros_like(env.observation["InstCountNorm"])
    for i in range(10):
        obs = env.observation["InstCountNorm"]
        # assert np.any(prev_obs != obs)
        action, value = agent.choose_action(
            obs,
            enable_epsilon_greedy=False,
            forbidden_actions=set(),
            eval_mode=True,
        )
        if value <= 0:
            break
        print(COMPLETE_ACTION_SET[action], end=" ")
        env.step(env.action_space.flags.index(COMPLETE_ACTION_SET[action]))
        prev_obs = obs
    print()


def main():
    env: CompilerEnv = gym.make("llvm-v0")
    # env = RuntimePointEstimateReward(
    #     env=env,
    # )
    benchmarks = list(env.datasets["benchmark://cbench-v1"].benchmarks())
    results = {
        "benchmark": [],
        #
        "base_runtime": [],
        "o3_runtime": [],
        "model_runtime": [],
        #
        "base_thr": [],
        "o3_thr": [],
        "model_thr": [],
        #
        "base_speedup": [],
        "o3_speedup": [],
        #
        "base_thr_imp": [],
        "o3_thr_imp": [],
    }

    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path("lively-snowflake-49"),
    )

    pd_results = pd.DataFrame(columns=list(results.keys()))

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

            results["benchmark"].append(str(benchmark).rsplit("/", maxsplit=1)[-1])

            new_env.runtime_observation_count = runtime_count
            new_env.runtime_warmup_count = 0
            # new_env.apply(env.state)
            new_env.reset()
            base_runtimes = new_env.observation.Runtime()
            assert len(base_runtimes) == runtime_count
            results["base_runtime"].append(np.median(base_runtimes))
            results["base_thr"].append(
                get_rblock_throughput_ir(new_env.observation["Ir"])
            )

            new_env.reset()
            # apply_pass_sequence(new_env, O1_SEQ)
            new_env.send_param("llvm.apply_baseline_optimizations", "-O3")
            o3_runtimes = new_env.observation.Runtime()
            assert len(o3_runtimes) == runtime_count
            results["o3_runtime"].append(np.median(o3_runtimes))
            results["o3_thr"].append(
                get_rblock_throughput_ir(new_env.observation["Ir"])
            )

            new_env.reset()
            optimize_with_model(config, agent, new_env)
            model_runtimes = new_env.observation.Runtime()
            assert len(model_runtimes) == runtime_count
            results["model_runtime"].append(np.median(model_runtimes))
            results["model_thr"].append(
                get_rblock_throughput_ir(new_env.observation["Ir"])
            )

            base_speedup = np.median(base_runtimes) / max(
                np.median(model_runtimes), 1e-12
            )
            o3_speedup = np.median(o3_runtimes) / max(np.median(model_runtimes), 1e-12)

            base_rblock_throughput_imp = (
                results["base_thr"][-1] / results["model_thr"][-1]
            )
            o3_rblock_throughput_imp = results["o3_thr"][-1] / results["model_thr"][-1]

            results["base_speedup"].append(base_speedup)
            results["o3_speedup"].append(o3_speedup)

            results["base_thr_imp"].append(base_rblock_throughput_imp)
            results["o3_thr_imp"].append(o3_rblock_throughput_imp)

            pd_results.loc[len(pd_results)] = [results[key][-1] for key in results]

            print(
                tabulate(
                    pd_results.iloc[[len(pd_results) - 1]],
                    headers="keys",
                    tablefmt="psql",
                )
            )

    pd_results.to_csv("o3_cbench_test_results.csv")
    print(
        tabulate(
            pd_results,
            headers="keys",
            tablefmt="psql",
        )
    )
    print(pd_results.drop(columns=["benchmark"]).mean())


if __name__ == "__main__":
    main()
