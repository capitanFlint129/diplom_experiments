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

MODEL_ITERS = 10
RUNTIME_COUNT = 30
BIN_NAME = "tmp_o3_cbench_test_cfg_grind_bin"
RUN_TIME = "daily-capybara-88"


def get_speedup(compare_mean, model_mean) -> float:
    return compare_mean / max(model_mean, 1e-12)


def main():
    env: CompilerEnv = gym.make("llvm-v0")
    benchmarks = list(env.datasets["benchmark://cbench-v1"].benchmarks())
    results = {
        "benchmark": [],
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
            results["base_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=[],
                    result_path=BIN_NAME,
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )
            base_mean, base_std, _ = measure_execution_mean_and_std(
                f"./{BIN_NAME}", benchmark_args
            )
            results["base_runtime"].append(base_mean)

            new_env.reset()
            new_env.send_param("llvm.apply_baseline_optimizations", "-O3")
            results["o3_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=[],
                    result_path=BIN_NAME,
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )
            o3_mean, o3_std, _ = measure_execution_mean_and_std(
                f"./{BIN_NAME}", benchmark_args
            )
            results["o3_runtime"].append(o3_mean)

            new_env.reset()
            optimize_with_model(config, agent, new_env, iters=MODEL_ITERS)
            results["model_inst"].append(
                compile_and_get_instructions(
                    ir=new_env.observation["Ir"],
                    sequence=[],
                    result_path=BIN_NAME,
                    execution_args=benchmark_args,
                    linkopts=linkopts,
                )
            )
            model_mean, model_std, _ = measure_execution_mean_and_std(
                f"./{BIN_NAME}", benchmark_args
            )
            results["model_runtime"].append(model_mean)

            base_speedup = get_speedup(base_mean, model_mean)
            o3_speedup = get_speedup(o3_mean, model_mean)

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
