import json
import os
import subprocess

import pandas as pd
from tqdm import tqdm

DATASET_URI = "benchmark://jotaibench-v0"
RUN_NAME = "lively-snowflake-49"
RUNS = 30


def parse_exec_time(time_stdout) -> float:
    time_str = time_stdout.split("\n", maxsplit=1)[0]
    minutes, time_str = time_str.split("m")
    seconds = time_str.split("s")[0]
    minutes = float(minutes.replace(",", "."))
    seconds = float(seconds.replace(",", "."))
    return minutes * 60 + seconds


def parse_exec_time_hyperfine(time_stdout) -> float:
    time_str = time_stdout.split("\n", maxsplit=1)[0]
    minutes, time_str = time_str.split("m")
    seconds = time_str.split("s")[0]
    minutes = float(minutes.replace(",", "."))
    seconds = float(seconds.replace(",", "."))
    return minutes * 60 + seconds


def measure_execution_mean_and_std(
    bin_path, execution_args: str = "", runs=RUNS
) -> tuple[float, float]:
    # with Timer() as timer:
    #     proc = subprocess.run(
    #         [bin_path] + execution_args.split(),
    #         capture_output=True,
    #         # shell=True,
    #     )
    # if proc.returncode != 0:
    #     print(proc.stderr)
    #     raise Exception(f"run failed: {proc.stderr}")
    # runtimes.append(timer.time)
    proc = subprocess.run(
        f"hyperfine '{bin_path} {execution_args}' --export-json hyperfine_result.json --show-output",
        shell=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"run failed: {proc.stderr}")
    with open("hyperfine_result.json", "r") as hyperfine_result:
        result = json.load(hyperfine_result)
    return result["results"][0]["mean"], result["results"][0]["stddev"]


def main():
    result = {
        "benchmark": [],
        "O3 runtime, mean": [],
        "O3 runtime, std": [],
        "O0 runtime, mean": [],
        "O0 runtime, std": [],
        "model runtime, mean": [],
        "model runtime, std": [],
        "O3 speedup": [],
        "O0 speedup": [],
    }
    execution_args = "0"
    benchmarks = sorted(os.listdir("data/model"))
    for benchmark in tqdm(benchmarks):
        o3_runtimes_mean, o3_runtimes_std = measure_execution_mean_and_std(
            f"data/O3/{benchmark}", execution_args=execution_args
        )

        o0_runtimes_mean, o0_runtimes_std = measure_execution_mean_and_std(
            f"data/O0/{benchmark}", execution_args=execution_args
        )

        model_runtimes_mean, model_runtimes_std = measure_execution_mean_and_std(
            f"data/model/{benchmark}", execution_args=execution_args
        )

        result["benchmark"].append(str(benchmark).split("/")[-1])
        result["O3 runtime, mean"].append(o3_runtimes_mean)
        result["O3 runtime, std"].append(o3_runtimes_std)
        result["O0 runtime, mean"].append(o0_runtimes_mean)
        result["O0 runtime, std"].append(o0_runtimes_std)
        result["model runtime, mean"].append(model_runtimes_mean)
        result["model runtime, std"].append(model_runtimes_std)
        result["O3 speedup"].append(o3_runtimes_mean / model_runtimes_mean)
        result["O0 speedup"].append(o0_runtimes_mean / model_runtimes_mean)

    result_df = pd.DataFrame(data=result)
    print(result_df.drop(columns=["benchmark"]).mean())


if __name__ == "__main__":
    main()
