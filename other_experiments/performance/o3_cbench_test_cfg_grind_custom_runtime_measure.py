import argparse
import os
from subprocess import TimeoutExpired

import compiler_gym
import gym
import pandas as pd
import torch
from compiler_gym import CompilerEnv
from tabulate import tabulate
from tqdm import tqdm

from env.cfg_grind import compile_and_get_instructions
from runtime_eval.hyperfine_utils import save_whisker_plot
from runtime_eval.jotai.eval import measure_execution_mean_and_std
from utils import (
    get_agent,
    get_model_path,
    optimize_with_model,
    load_config,
)

# MODEL_ITERS = 25
# RUNTIME_COUNT = 30
BENCHMARKS_LIMIT = None
CBENCH_EVAL_DIR = "cbench_eval_dir"
BIN_NAME = "tmp_o3_cbench_test_cfg_grind_bin"

# WORKAROUND_CBENCH_COMMAND_ARGS = None


def apply_pass_sequence(env: CompilerEnv, pass_sequence):
    action_space_passes_set = set(env.action_space.flags)
    for pass_el in pass_sequence:
        if pass_el in action_space_passes_set:
            observation, reward, done, info = env.step(
                env.action_space.flags.index(pass_el)
            )
            # print(reward)


def process_optimizator(
    optimizator_name,
    env,
    results: dict,
    hypefine_results: dict,
    linkopts,
    benchmark_args,
    prepare_command,
    sequence,
) -> tuple[float, float]:
    bin_path = os.path.join(CBENCH_EVAL_DIR, args.run_name, BIN_NAME)
    results[f"{optimizator_name}_inst"].append(
        compile_and_get_instructions(
            ir=env.observation["Ir"],
            sequence=sequence,
            result_path=bin_path,
            execution_args=benchmark_args,
            linkopts=linkopts,
        )
    )
    mean, std, hyperfine_result = measure_execution_mean_and_std(
        f"./{bin_path}",
        benchmark_args,
        prepare_command=prepare_command,
        specific_name=optimizator_name,
        warmup=10,
    )
    hypefine_results[optimizator_name][results["benchmark"][-1]] = hyperfine_result
    results[f"{optimizator_name}_runtime"].append(mean)
    return mean, std


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
    os.makedirs(os.path.join(CBENCH_EVAL_DIR, args.run_name), exist_ok=True)

    env: CompilerEnv = gym.make("llvm-v0")
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

    # config = TrainConfig()
    config = load_config(args.run_name)
    config.save(args.run_name, replace=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path(args.run_name),
    )

    pd_results = pd.DataFrame(columns=list(results.keys()))
    pd_results_std = pd.DataFrame(
        columns=[
            "benchmark",
            "base_runtime",
            "o3_runtime",
            "o2_runtime",
            "model_runtime",
        ]
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
    hypefine_results = {
        "base": {},
        "o3": {},
        "o2": {},
        "model": {},
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
                optimize_with_model(
                    config, agent, new_env, iters=config.episode_length, eval_mode=True
                )
            except TimeoutExpired as e:
                print(f"IR2vec timeout skip benchmark: {e}")
            except Exception as e:
                print(f"Exception. Skip benchmark: {e}")
                continue

            results["benchmark"].append(benchmark_name)

            model_mean, model_std = process_optimizator(
                "model",
                new_env,
                results,
                hypefine_results,
                linkopts,
                benchmark_args,
                prepare_command,
                sequence=[],
            )

            new_env.reset()
            base_mean, base_std = process_optimizator(
                "base",
                new_env,
                results,
                hypefine_results,
                linkopts,
                benchmark_args,
                prepare_command,
                sequence=[],
            )

            new_env.reset()
            o3_mean, o3_std = process_optimizator(
                "o3",
                new_env,
                results,
                hypefine_results,
                linkopts,
                benchmark_args,
                prepare_command,
                sequence=["-O3"],
            )

            new_env.reset()
            o2_mean, o2_std = process_optimizator(
                "o2",
                new_env,
                results,
                hypefine_results,
                linkopts,
                benchmark_args,
                prepare_command,
                sequence=["-O2"],
            )

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

            pd_results.loc[len(pd_results)] = [results[key][-1] for key in results]
            pd_results_std.loc[len(pd_results_std)] = [
                benchmark_name,
                base_std,
                o3_std,
                o2_std,
                model_std,
            ]

            print_df_last_row(pd_results)
            print_df_last_row(pd_results_std)

    pd_results.to_csv(
        os.path.join(CBENCH_EVAL_DIR, args.run_name, "runtime_measure.csv")
    )
    pd_results_std.to_csv(
        os.path.join(CBENCH_EVAL_DIR, args.run_name, "runtime_measure_std.csv")
    )
    save_hyperfine_whisker_plots(hypefine_results)
    print_df(pd_results)
    print_df(pd_results_std)
    print(pd_results.drop(columns=["benchmark"]).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name")
    parser.add_argument(
        "--debug",
        help="debug",
        action="store_true",
    )
    args = parser.parse_args()

    main()
