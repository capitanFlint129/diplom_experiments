from subprocess import TimeoutExpired

import compiler_gym
import gym
import pandas as pd
import torch
from compiler_gym import CompilerEnv
from tabulate import tabulate
from tqdm import tqdm

from config.config import TrainConfig
from env.cfg_grind import compile_and_get_instructions
from runtime_eval.jotai.eval import measure_execution_mean_and_std
from utils import (
    get_agent,
    get_model_path,
    optimize_with_model,
)

MODEL_ITERS = 25
# RUNTIME_COUNT = 30
BIN_NAME = "tmp_o3_cbench_test_cfg_grind_bin"
RUN_TIME = "misunderstood-sunset-112"

# WORKAROUND_CBENCH_COMMAND_ARGS = None


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
        #
        "base_runtime": [],
        "o3_runtime": [],
        "o2_runtime": [],
        "model_runtime": [],
        #
        "base_inst": [],
        "o3_inst": [],
        "o2_inst": [],
        "model_inst": [],
        #
        "base_speedup": [],
        "o3_speedup": [],
        "o2_speedup": [],
        #
        "base_inst_imp": [],
        "o3_inst_imp": [],
        "o2_inst_imp": [],
    }

    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path(RUN_TIME),
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

    skipped_benchmarks = {
        # "bzip2",
    }
    for benchmark in tqdm(benchmarks):
        benchmark_name = str(benchmark).rsplit("/", maxsplit=1)[-1]
        if benchmark_name in skipped_benchmarks:
            continue
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

            linkopts = ["-lm"] if benchmark_name in math_benchs else []

            new_env.reset()
            try:
                optimize_with_model(config, agent, new_env, iters=MODEL_ITERS)
            except TimeoutExpired as e:
                print(f"IR2vec timeout skip benchmark: {e}")
            results["model_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=[],
                    result_path=BIN_NAME,
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )
            model_mean, model_std = measure_execution_mean_and_std(
                f"./{BIN_NAME}", benchmark_args
            )
            results["model_runtime"].append(model_mean)

            new_env.reset()
            results["base_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=[],
                    result_path=BIN_NAME,
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )
            base_mean, base_std = measure_execution_mean_and_std(
                f"./{BIN_NAME}", benchmark_args
            )
            results["base_runtime"].append(base_mean)

            new_env.reset()
            # new_env.send_param("llvm.apply_baseline_optimizations", "-O3")
            results["o3_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=["-O3"],
                    result_path=BIN_NAME,
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )
            o3_mean, o3_std = measure_execution_mean_and_std(
                f"./{BIN_NAME}", benchmark_args
            )
            results["o3_runtime"].append(o3_mean)

            new_env.reset()
            results["o2_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=["-O2"],
                    result_path=BIN_NAME,
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )
            o2_mean, o2_std = measure_execution_mean_and_std(
                f"./{BIN_NAME}", benchmark_args
            )
            results["o2_runtime"].append(o2_mean)

            base_speedup = (base_mean - model_mean) / base_mean
            o3_speedup = (o3_mean - model_mean) / base_mean
            o2_speedup = (o2_mean - model_mean) / base_mean

            base_inst_imp = (
                results["base_inst"][-1] - results["model_inst"][-1]
            ) / results["base_inst"][-1]
            o3_inst_imp = (
                results["o3_inst"][-1] - results["model_inst"][-1]
            ) / results["base_inst"][-1]
            o2_inst_imp = (
                results["o2_inst"][-1] - results["model_inst"][-1]
            ) / results["base_inst"][-1]

            results["base_speedup"].append(base_speedup)
            results["o3_speedup"].append(o3_speedup)
            results["o2_speedup"].append(o2_speedup)

            results["base_inst_imp"].append(base_inst_imp)
            results["o3_inst_imp"].append(o3_inst_imp)
            results["o2_inst_imp"].append(o2_inst_imp)

            results["benchmark"].append(benchmark_name)

            pd_results.loc[len(pd_results)] = [results[key][-1] for key in results]

            print(
                tabulate(
                    pd_results.iloc[[len(pd_results) - 1]],
                    headers="keys",
                    tablefmt="psql",
                )
            )

    pd_results.to_csv("o3_cbench_test_cfg_grind_custom_runtime_measure.csv")
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
