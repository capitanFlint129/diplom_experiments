import argparse
import dataclasses
import os
from subprocess import TimeoutExpired
import random

import compiler_gym
import gym
import numpy as np
import pandas as pd
import torch
from compiler_gym import CompilerEnv
from scipy import stats
from tabulate import tabulate
from tqdm import tqdm

from config.config import TrainConfig
from env.performance_optimization.llvm import compile_lm_safely
from runtime_eval.hyperfine_utils import save_whisker_plot
from runtime_eval.jotai.eval import measure_execution_mean_and_std
from utils import (
    get_agent,
    get_model_path,
    optimize_with_model,
)

# MODEL_ITERS = 25
# RUNTIME_COUNT = 30
BENCHMARKS_LIMIT = None
CBENCH_EVAL_DIR = "cbench_eval_dir"
BIN_NAME = "tmp_o3_cbench_test_cfg_grind_bin"

# WORKAROUND_CBENCH_COMMAND_ARGS = None


@dataclasses.dataclass
class OptimizatorData:
    name: str
    sequence: list
    # env: CompilerEnv


def apply_pass_sequence(env: CompilerEnv, pass_sequence):
    action_space_passes_set = set(env.action_space.flags)
    for pass_el in pass_sequence:
        if pass_el in action_space_passes_set:
            observation, reward, done, info = env.step(
                env.action_space.flags.index(pass_el)
            )
            # print(reward)


def compare_optimizators(
    model_opt: OptimizatorData,
    baseline_opt: OptimizatorData,
    env: CompilerEnv,
    linkopts,
    benchmark_args,
    prepare_command,
    n=30,
) -> list[float]:
    model_bin = os.path.join(
        CBENCH_EVAL_DIR, args.run_name, f"{BIN_NAME}-{model_opt.name}"
    )
    baseline_bin = os.path.join(
        CBENCH_EVAL_DIR, args.run_name, f"{BIN_NAME}-{baseline_opt.name}"
    )

    compile_lm_safely(
        ir=env.observation["Ir"],
        sequence=model_opt.sequence,
        result_path=model_bin,
        linkopts=linkopts,
    )

    compile_lm_safely(
        ir=env.observation["Ir"],
        sequence=baseline_opt.sequence,
        result_path=baseline_bin,
        linkopts=linkopts,
    )
    speedups = []
    for i in range(n):
        if random.randint(0, 1):
            model_runtime, _, _ = measure_execution_mean_and_std(
                f"./{model_bin}",
                benchmark_args,
                prepare_command=prepare_command,
                specific_name=model_opt.name,
                runs=1,
                warmup=0,
            )
            baseline_runtime, _, _ = measure_execution_mean_and_std(
                f"./{baseline_bin}",
                benchmark_args,
                prepare_command=prepare_command,
                specific_name=baseline_opt.name,
                runs=1,
                warmup=0,
            )
        else:
            baseline_runtime, _, _ = measure_execution_mean_and_std(
                f"./{baseline_bin}",
                benchmark_args,
                prepare_command=prepare_command,
                specific_name=baseline_opt.name,
                runs=1,
                warmup=0,
            )
            model_runtime, _, _ = measure_execution_mean_and_std(
                f"./{model_bin}",
                benchmark_args,
                prepare_command=prepare_command,
                specific_name=model_opt.name,
                runs=1,
                warmup=0,
            )

        speedups.append(
            (baseline_runtime - model_runtime) / max(baseline_runtime, 1e-12)
        )
    return speedups


def print_df_last_row(df):
    print(
        tabulate(
            df.iloc[[len(df) - 1]],
            headers="keys",
            tablefmt="psql",
        )
    )


def print_df(df):
    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="psql",
        )
    )


def save_hyperfine_whisker_plots(
    hypefine_results: dict[str, dict[str, dict]], split_by_optimizator=True
):
    plots_dir = os.path.join(CBENCH_EVAL_DIR, args.run_name, "whisker_plots")
    os.makedirs(plots_dir, exist_ok=True)
    if split_by_optimizator:
        benchmarks_names = list(
            hypefine_results[next(iter(hypefine_results.keys()))].keys()
        )
        for benchmark_name in benchmarks_names:
            hyperfine_run_data = {
                optimizator_name: hypefine_results[optimizator_name][benchmark_name]
                for optimizator_name in hypefine_results
            }
            save_whisker_plot(
                hyperfine_run_data,
                result_filename=os.path.join(
                    plots_dir, f"{benchmark_name}_whisker_plot.svg"
                ),
                title=benchmark_name,
            )
    else:
        for optimizator_name, hyperfine_run_data in hypefine_results.items():
            save_whisker_plot(
                hyperfine_run_data,
                result_filename=os.path.join(
                    plots_dir, f"{optimizator_name}_whisker_plot.svg"
                ),
                title=os.path.join(CBENCH_EVAL_DIR, args.run_name, optimizator_name),
            )


def main():
    N = 100 if args.n == -1 else args.n
    print(f"Runs for each benchmark: {N}")
    os.makedirs(os.path.join(CBENCH_EVAL_DIR, args.run_name), exist_ok=True)

    env: CompilerEnv = gym.make("llvm-v0")
    benchmarks = list(env.datasets["benchmark://cbench-v1"].benchmarks())
    results = {}

    # config = TrainConfig()
    config = TrainConfig.load_config(args.run_name)
    config.save(args.run_name, replace=False)
    episode_len = args.iters if args.iters > -1 else config.episode_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path(args.run_name),
    )

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
        "bzip2",
    }

    for benchmark in tqdm(benchmarks[:BENCHMARKS_LIMIT]):
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
            # prepare_command = 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'
            prepare_command = ""

            new_env.reset()
            try:
                flags = optimize_with_model(
                    config,
                    agent,
                    new_env,
                    iters=episode_len,
                    eval_mode=not args.disable_eval_mode,
                    hack=args.hack,
                )
                tmp = []
                for seq in flags:
                    if seq != "noop":
                        tmp += [f"-{f}" for f in seq.split()]
                flags = tmp
            except TimeoutExpired as e:
                print(f"IR2vec timeout skip benchmark: {e}")
            except Exception as e:
                print(f"Exception. Skip benchmark: {e}")
                continue

            new_env.reset()
            speedups = compare_optimizators(
                OptimizatorData("model", flags),
                OptimizatorData("o3", ["-O3"]),
                # OptimizatorData("o3", ["-O3"]),
                # OptimizatorData("o0", []),
                new_env,
                linkopts,
                benchmark_args,
                prepare_command,
                n=N,
            )
            results[benchmark_name] = speedups
            print(
                f"{benchmark_name}: {np.mean(speedups)} - {stats.norm.interval(0.95, loc=np.mean(speedups), scale=np.std(speedups) / np.sqrt(N))}"
            )

    df = pd.DataFrame(data=results)
    df.to_csv(f"{args.run_name}_speedup_with_confidence.csv")
    all_speedups = np.concatenate(list(results.values()))
    print(
        f"all: {np.mean(all_speedups)} - {stats.norm.interval(0.95, loc=np.mean(all_speedups), scale=np.std(all_speedups) / np.sqrt(len(all_speedups)))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable_eval_mode",
        help="disable_eval_mode",
        action="store_true",
    )
    parser.add_argument(
        "--hack",
        help="eval_mode",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="debug",
        action="store_true",
    )
    parser.add_argument(
        "--iters",
        type=int,
        # nargs=1,
        # action="store",
        # choices=range(0, 100),
        default=-1,
    )
    parser.add_argument(
        "--n",
        type=int,
        # nargs=1,
        # action="store",
        # choices=range(0, 100),
        default=-1,
    )
    parser.add_argument("run_name", help="run name")

    args = parser.parse_args()
    if args.hack and args.disable_eval_mode:
        raise Exception("args.hack == True and args.disable_eval_mode == False failed")

    main()
