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
from runtime_eval.llvm_test_suite.consts import DATA_DIR


WARMUP = 0
DATASET_URI = "benchmark://jotaibench-v0"
MEASURE_EXECUTED_INSTRUCTIONS = True


def compare_optimizators(
    model_bin: str,
    baseline_bin: str,
    benchmark_args,
    prepare_command,
    n=30,
) -> list[float]:
    speedups = []
    for i in range(n):
        if random.randint(0, 1):
            model_runtime, _, _ = measure_execution_mean_and_std(
                f"./{model_bin}",
                benchmark_args,
                prepare_command=prepare_command,
                specific_name=f"{args.run_name}_model",
                runs=1,
                warmup=0,
            )
            baseline_runtime, _, _ = measure_execution_mean_and_std(
                f"./{baseline_bin}",
                benchmark_args,
                prepare_command=prepare_command,
                specific_name=f"{args.run_name}_o3",
                runs=1,
                warmup=0,
            )
        else:
            baseline_runtime, _, _ = measure_execution_mean_and_std(
                f"./{baseline_bin}",
                benchmark_args,
                prepare_command=prepare_command,
                specific_name=f"{args.run_name}_o3",
                runs=1,
                warmup=0,
            )
            model_runtime, _, _ = measure_execution_mean_and_std(
                f"./{model_bin}",
                benchmark_args,
                prepare_command=prepare_command,
                specific_name=f"{args.run_name}_model",
                runs=1,
                warmup=0,
            )

        speedups.append(
            (baseline_runtime - model_runtime) / max(baseline_runtime, 1e-12)
        )
    return speedups


def main():
    results = {
        "benchmark": [],
        "speedups": [],
    }

    execution_args = "0"
    benchmarks = list(sorted(os.listdir(f"{RUN_DIR_PATH}/model")))
    if args.debug:
        benchmarks = benchmarks[:10]
    for benchmark in tqdm(benchmarks):
        speedups = compare_optimizators(
            f"{RUN_DIR_PATH}/model/{benchmark}",
            f"{RUN_DIR_PATH}/O3/{benchmark}",
            execution_args,
            prepare_command="",
            n=args.n,
        )
        results["benchmark"].append(benchmark)
        results["speedups"].append(speedups)
        print(
            f"{benchmark}: {np.mean(speedups)} - {stats.norm.interval(0.95, loc=np.mean(speedups), scale=np.std(speedups) / np.sqrt(args.n))}"
        )

    all_speedups = np.concatenate(results["speedups"])
    print(all_speedups)

    results["benchmark"].append("all")
    results["speedups"].append(all_speedups)

    df = pd.DataFrame(data=results)
    df.to_csv(f"{args.run_name}_speedup_with_confidence.csv")
    print(
        f"all: {np.mean(all_speedups)} - {stats.norm.interval(0.95, loc=np.mean(all_speedups), scale=np.std(all_speedups) / np.sqrt(len(all_speedups)))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name")
    parser.add_argument(
        "--debug",
        help="debug",
        action="store_true",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=300,
    )
    args = parser.parse_args()

    RUN_DIR_PATH = f"{DATA_DIR}/{args.run_name}"

    main()
