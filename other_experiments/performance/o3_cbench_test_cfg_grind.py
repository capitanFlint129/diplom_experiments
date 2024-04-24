import compiler_gym
import gym
import numpy as np
import pandas as pd
import torch
from compiler_gym import CompilerEnv
from tabulate import tabulate
from tqdm import tqdm

from config.config import TrainConfig
from env.cfg_grind import compile_and_get_instructions
from utils import (
    get_agent,
    get_model_path,
    optimize_with_model,
)

MODEL_ITERS = 10
RUNTIME_COUNT = 30


# WORKAROUND_CBENCH_COMMAND_ARGS = None


def apply_pass_sequence(env: CompilerEnv, pass_sequence):
    action_space_passes_set = set(env.action_space.flags)
    for pass_el in pass_sequence:
        if pass_el in action_space_passes_set:
            observation, reward, done, info = env.step(
                env.action_space.flags.index(pass_el)
            )
            # print(reward)


def get_speedup(compare_runtimes, model_runtimes) -> float:
    return np.median(compare_runtimes) / max(np.median(model_runtimes), 1e-12)


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
        #
        "base_runtime": [],
        "o3_runtime": [],
        "model_runtime": [],
        #
        "base_inst": [],
        "o3_inst": [],
        "model_inst": [],
        #
        "base_speedup": [],
        "o3_speedup": [],
        #
        "base_inst_imp": [],
        "o3_inst_imp": [],
    }

    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path("zesty-morning-52"),
    )

    pd_results = pd.DataFrame(columns=list(results.keys()))

    math_benchs = {
        "qsort",
        "stringsearch",
        "susan",
        "tiff2bw",
        "tiff2rgba",
        "tiffdither",
        "tiffmedian",
    }
    for benchmark in tqdm(benchmarks):
        with compiler_gym.make("llvm-v0", benchmark=benchmark) as new_env:
            new_env.reset()
            if not new_env.observation["IsRunnable"]:
                print(f"Benchmark {benchmark} not runnable, skip it")
                continue

            try:
                benchmark.validate(new_env)
            except Exception as e:
                benchmark_args = e.msg.split(".bc")[-1].strip()
                cg_working_dir = e.dir

            linkopts = (
                ["-lm"]
                if str(benchmark).rsplit("/", maxsplit=1)[-1] in math_benchs
                else []
            )
            results["benchmark"].append(str(benchmark).rsplit("/", maxsplit=1)[-1])

            new_env.runtime_observation_count = RUNTIME_COUNT
            new_env.runtime_warmup_count = 0
            new_env.reset()
            base_runtimes = new_env.observation.Runtime()
            assert len(base_runtimes) == RUNTIME_COUNT
            results["base_runtime"].append(np.median(base_runtimes))
            results["base_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=[],
                    result_path="tmp_o3_cbench_test_cfg_grind_bin",
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )

            new_env.reset()
            new_env.send_param("llvm.apply_baseline_optimizations", "-O3")
            o3_runtimes = new_env.observation.Runtime()
            assert len(o3_runtimes) == RUNTIME_COUNT
            results["o3_runtime"].append(np.median(o3_runtimes))
            results["o3_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=[],
                    result_path="tmp_o3_cbench_test_cfg_grind_bin",
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )

            new_env.reset()
            optimize_with_model(config, agent, new_env, iters=MODEL_ITERS)
            model_runtimes = new_env.observation.Runtime()
            assert len(model_runtimes) == RUNTIME_COUNT
            results["model_runtime"].append(np.median(model_runtimes))
            results["model_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=[],
                    result_path="tmp_o3_cbench_test_cfg_grind_bin",
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )

            base_speedup = get_speedup(base_runtimes, model_runtimes)
            o3_speedup = get_speedup(o3_runtimes, model_runtimes)

            base_rblock_inst_imp = results["base_inst"][-1] / results["model_inst"][-1]
            o3_rblock_inst_imp = results["o3_inst"][-1] / results["model_inst"][-1]

            results["base_speedup"].append(base_speedup)
            results["o3_speedup"].append(o3_speedup)

            results["base_inst_imp"].append(base_rblock_inst_imp)
            results["o3_inst_imp"].append(o3_rblock_inst_imp)

            pd_results.loc[len(pd_results)] = [results[key][-1] for key in results]

            print(
                tabulate(
                    pd_results.iloc[[len(pd_results) - 1]],
                    headers="keys",
                    tablefmt="psql",
                )
            )

    pd_results.to_csv("o3_cbench_test_cfg_grind.csv")
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
