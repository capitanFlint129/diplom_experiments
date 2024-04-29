import json
import os
import random
import subprocess

import pandas as pd
from tqdm import tqdm

from env.cfg_grind import get_executed_instructions
from runtime_eval.jotai.generate_optimizations_for_jotai import RUN_NAME

DATASET_URI = "benchmark://jotaibench-v0"
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
    bin_path, execution_args: str = "", prepare_command="", runs=RUNS, specific_name="", warmup=0,
) -> tuple[float, float, dict]:
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
    filename = f"{specific_name}_hyperfine_result.json"
    if len(prepare_command) != 0:
        command = f"hyperfine --prepare '{prepare_command}' --warmup {0} '{bin_path} {execution_args}' --export-json {filename} --show-output"
    else:
        command = f"hyperfine --warmup {0} '{bin_path} {execution_args}' --export-json {filename} --show-output"
    proc = subprocess.run(
        command,
        shell=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"run failed: {proc.stderr}")
    with open(filename, "r") as hyperfine_result:
        result = json.load(hyperfine_result)
    return result["results"][0]["mean"], result["results"][0]["stddev"], result["results"][0]


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    result = {
        "benchmark": [],
        "O3 runtime, mean": [],
        "O3 runtime, std": [],
        "O2 runtime, mean": [],
        "O2 runtime, std": [],
        "O0 runtime, mean": [],
        "O0 runtime, std": [],
        "model runtime, mean": [],
        "model runtime, std": [],
        "O3 speedup": [],
        "O2 speedup": [],
        "O0 speedup": [],
    }

    exec_inst = {
        "O3 exec_inst": [],
        "O2 exec_inst": [],
        "O0 exec_inst": [],
        "model exec_inst": [],
        "O3 exec_inst imp": [],
        "O2 exec_inst imp": [],
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

        o2_runtimes_mean, o2_runtimes_std = measure_execution_mean_and_std(
            f"data/O2/{benchmark}", execution_args=execution_args
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
        result["O2 runtime, mean"].append(o2_runtimes_mean)
        result["O2 runtime, std"].append(o2_runtimes_std)
        result["O0 runtime, mean"].append(o0_runtimes_mean)
        result["O0 runtime, std"].append(o0_runtimes_std)
        result["model runtime, mean"].append(model_runtimes_mean)
        result["model runtime, std"].append(model_runtimes_std)
        result["O3 speedup"].append(
            (o3_runtimes_mean - model_runtimes_mean) / o0_runtimes_mean
        )
        result["O2 speedup"].append(
            (o2_runtimes_mean - model_runtimes_mean) / o0_runtimes_mean
        )
        result["O0 speedup"].append(
            (o0_runtimes_mean - model_runtimes_mean) / o0_runtimes_mean
        )

        if MEASURE_EXECUTED_INSTRUCTIONS:
            o3_exec_inst = get_executed_instructions(
                f"data/O3/{benchmark}", execution_args
            )
            o2_exec_inst = get_executed_instructions(
                f"data/O2/{benchmark}", execution_args
            )
            o0_exec_inst = get_executed_instructions(
                f"data/O0/{benchmark}", execution_args
            )
            model_exec_inst = get_executed_instructions(
                f"data/model/{benchmark}", execution_args
            )
            exec_inst["O3 exec_inst"].append(o3_exec_inst)
            exec_inst["O2 exec_inst"].append(o2_exec_inst)
            exec_inst["O0 exec_inst"].append(o0_exec_inst)
            exec_inst["model exec_inst"].append(model_exec_inst)
            exec_inst["O3 exec_inst imp"].append(o3_exec_inst / model_exec_inst)
            exec_inst["O2 exec_inst imp"].append(o2_exec_inst / model_exec_inst)
            exec_inst["O0 exec_inst imp"].append(o0_exec_inst / model_exec_inst)

    if MEASURE_EXECUTED_INSTRUCTIONS:
        result.update(exec_inst)
    result_df = pd.DataFrame(data=result)
    result_df.to_csv(os.path.join(RESULT_DIR, f"{RUN_NAME}.csv"))
    print(result_df.drop(columns=["benchmark"]).mean())


if __name__ == "__main__":
    main()
