import json
import os
import random
import subprocess

import pandas as pd
from tqdm import tqdm

from env.cfg_grind import get_executed_instructions

DATASET_URI = "benchmark://jotaibench-v0"
RUN_NAME = "lively-snowflake-49"
RESULT_DIR = "_eval_results"
RUNS = 30
MEASURE_EXECUTED_INSTRUCTIONS = True


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
    os.makedirs(RESULT_DIR, exist_ok=True)

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

    exec_inst = {
        "O3 exec_inst": [],
        "O0 exec_inst": [],
        "model exec_inst": [],
        "O3 exec_inst imp": [],
        "O0 exec_inst imp": [],
    }

    execution_args = "0"
    benchmarks = list(sorted(os.listdir("data/model")))
    random.seed(0)
    random.shuffle(benchmarks)
    for benchmark in tqdm(benchmarks[:100]):
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

        if MEASURE_EXECUTED_INSTRUCTIONS:
            o3_exec_inst = get_executed_instructions(
                f"data/O3/{benchmark}", execution_args
            )
            o0_exec_inst = get_executed_instructions(
                f"data/O0/{benchmark}", execution_args
            )
            model_exec_inst = get_executed_instructions(
                f"data/model/{benchmark}", execution_args
            )
            exec_inst["O3 exec_inst"].append(o3_exec_inst)
            exec_inst["O0 exec_inst"].append(o0_exec_inst)
            exec_inst["model exec_inst"].append(model_exec_inst)
            exec_inst["O3 exec_inst imp"].append(o3_exec_inst / model_exec_inst)
            exec_inst["O0 exec_inst imp"].append(o0_exec_inst / model_exec_inst)

    if MEASURE_EXECUTED_INSTRUCTIONS:
        result.update(exec_inst)
    result_df = pd.DataFrame(data=result)
    result_df.to_csv(os.path.join(RESULT_DIR, f"{RUN_NAME}.csv"))
    print(result_df.drop(columns=["benchmark"]).mean())


if __name__ == "__main__":
    main()
