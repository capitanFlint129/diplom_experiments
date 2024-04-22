import gym
import pandas as pd
import numpy as np
from compiler_gym import CompilerEnv
from compiler_gym.errors import BenchmarkInitError
from tqdm import tqdm
from compiler_gym.wrappers import RuntimePointEstimateReward
import compiler_gym

from config.action_config import O3_SEQ, O2_SEQ, O1_SEQ, COMPLETE_ACTION_SET
from config.config import TrainConfig
from env.performance_optimization import LlvmMcaEnv, MyEnv
from dataclasses import asdict

import torch

from dqn.train import validate, rollout
from utils import (
    get_agent,
    get_model_path,
)

from dqn.dqn import DQNAgent


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
        action, value = agent.choose_action(obs, enable_epsilon_greedy=False, forbidden_actions=set(), eval_mode=True)
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
        "base_runtime": [],
        "o3_runtime": [],
        "model_runtime": [],
        "o3_speedup": [],
        "base_speedup": [],
    }

    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path("kind-serenity-37"),
    )
    
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
            new_env.reset()
            base_runtimes = new_env.observation.Runtime()
            assert len(base_runtimes) == runtime_count

            new_env.reset()
            optimize_with_model(config, agent, new_env)
            model_runtimes = new_env.observation.Runtime()
            assert len(model_runtimes) == runtime_count
            
            
            new_env.reset()
            # apply_pass_sequence(new_env, O1_SEQ)
            new_env.send_param("llvm.apply_baseline_optimizations", "-O3")

            optimize_with_model(config, agent, new_env)

            o3_runtimes = new_env.observation.Runtime()
            assert len(o3_runtimes) == runtime_count
            
            base_speedup = np.median(base_runtimes) / max(np.median(model_runtimes), 1e-12)
            o3_speedup = np.median(o3_runtimes) / max(np.median(model_runtimes), 1e-12)

            print(f"base_runtimes: {np.median(base_runtimes)}")
            print(f"o3_runtimes: {np.median(o3_runtimes)}")
            print(f"model_runtimes: {np.median(model_runtimes)}")
            print(f"base_speedup: {base_speedup}")
            print(f"o3_speedup: {o3_speedup}")

            results["base_runtime"].append(np.median(base_runtimes))
            results["o3_runtime"].append(np.median(o3_runtimes))
            results["model_runtime"].append(np.median(model_runtimes))
            results["base_speedup"].append(base_speedup)
            results["o3_speedup"].append(o3_speedup)

    pd_results = pd.DataFrame(data=results)
    pd_results.to_csv("o3_cbench_test_results.csv")
    print(f'base_runtime mean: {np.mean(results["base_runtime"])}')
    print(f'o3_runtime mean: {np.mean(results["o3_runtime"])}')
    print(f'model_runtime mean: {np.mean(results["model_runtime"])}')
    print(f'base_speedup mean: {np.mean(results["base_speedup"])}')
    print(f'o3_speedup mean: {np.mean(results["o3_speedup"])}')


if __name__ == "__main__":
    main()
