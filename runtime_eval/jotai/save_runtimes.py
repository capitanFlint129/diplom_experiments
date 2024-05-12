import argparse
import json
import os
import subprocess
import random

import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import stats

from env.performance_optimization.cfg_grind import get_executed_instructions
from runtime_eval.jotai.eval import measure_execution_mean_and_std
from runtime_eval.jotai.consts import TMP_DATA_DIR


WARMUP = 0
DATASET_URI = "benchmark://jotaibench-v0"
MEASURE_EXECUTED_INSTRUCTIONS = True


def compare_optimizators(
    model_bin: str,
    benchmark_args,
    prepare_command,
    n=30,
) -> list[float]:
    # speedups = []
    # for i in range(n):
    #     if random.randint(0, 1):
    runtimes = []
    for i in range(n):
        model_runtime, _, _ = measure_execution_mean_and_std(
            f"./{model_bin}",
            benchmark_args,
            prepare_command=prepare_command,
            specific_name=f"{args.run_name}_model",
            runs=1,
            warmup=0,
        )
        runtimes.append(model_runtime)
        #     baseline_runtime, _, _ = measure_execution_mean_and_std(
        #         f"./{baseline_bin}",
        #         benchmark_args,
        #         prepare_command=prepare_command,
        #         specific_name=f"{args.run_name}_o3",
        #         runs=1,
        #         warmup=0,
        #     )
        # else:
        #     baseline_runtime, _, _ = measure_execution_mean_and_std(
        #         f"./{baseline_bin}",
        #         benchmark_args,
        #         prepare_command=prepare_command,
        #         specific_name=f"{args.run_name}_o3",
        #         runs=1,
        #         warmup=0,
        #     )
        #     model_runtime, _, _ = measure_execution_mean_and_std(
        #         f"./{model_bin}",
        #         benchmark_args,
        #         prepare_command=prepare_command,
        #         specific_name=f"{args.run_name}_model",
        #         runs=1,
        #         warmup=0,
        #     )
        #
        # speedups.append(
        #     (baseline_runtime - model_runtime) / max(baseline_runtime, 1e-12)
        # )
    return runtimes


def main():
    os.makedirs(RUN_DIR_PATH, exist_ok=True)

    results = {
        "benchmark": [],
        "runtimes": [],
    }

    execution_args = "0"
    benchmarks = list(sorted(os.listdir(f"{RUN_DIR_PATH}/bin")))
    if args.debug:
        benchmarks = benchmarks[:10]
    for benchmark in tqdm(benchmarks):
        runtimes = compare_optimizators(
            f"{RUN_DIR_PATH}/bin/{benchmark}",
            benchmark_args=execution_args,
            prepare_command="",
            n=args.runs,
        )
        results["benchmark"].append(benchmark)
        results["runtimes"].append(runtimes)
        # print(
        #     f"{benchmark}: {np.mean(speedups)} - {stats.norm.interval(0.95, loc=np.mean(speedups), scale=np.std(speedups) / np.sqrt(args.runs))}"
        # )

    # all_speedups = np.concatenate(results["speedups"])
    # print(all_speedups)

    # results["benchmark"].append("all")
    # results["speedups"].append(all_speedups)

    df = pd.DataFrame(data=results)
    df.to_csv(f"{TMP_DATA_DIR}/{args.run_name}/{args.run_name}_runtimes.csv")
    # print(
    #     f"all: {np.mean(all_speedups)} - {stats.norm.interval(0.95, loc=np.mean(all_speedups), scale=np.std(all_speedups) / np.sqrt(len(all_speedups)))}"
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name")
    parser.add_argument(
        "--debug",
        help="debug",
        action="store_true",
    )
    parser.add_argument(
        "--runs",
        type=int,
        # nargs=1,
        # action="store",
        # choices=range(0, 100),
        default=300,
    )
    args = parser.parse_args()

    RUN_DIR_PATH = f"{TMP_DATA_DIR}/{args.run_name}"

    main()
