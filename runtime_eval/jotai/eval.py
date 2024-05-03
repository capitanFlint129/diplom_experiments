import argparse
import json
import os
import subprocess

import pandas as pd
from tqdm import tqdm

from env.performance_optimization.cfg_grind import get_executed_instructions


WARMUP = 0
DATASET_URI = "benchmark://jotaibench-v0"
MEASURE_EXECUTED_INSTRUCTIONS = True


# def parse_exec_time(time_stdout) -> float:
#     time_str = time_stdout.split("\n", maxsplit=1)[0]
#     minutes, time_str = time_str.split("m")
#     seconds = time_str.split("s")[0]
#     minutes = float(minutes.replace(",", "."))
#     seconds = float(seconds.replace(",", "."))
#     return minutes * 60 + seconds
#
#
# def parse_exec_time_hyperfine(time_stdout) -> float:
#     time_str = time_stdout.split("\n", maxsplit=1)[0]
#     minutes, time_str = time_str.split("m")
#     seconds = time_str.split("s")[0]
#     minutes = float(minutes.replace(",", "."))
#     seconds = float(seconds.replace(",", "."))
#     return minutes * 60 + seconds


def measure_execution_mean_and_std(
    bin_path,
    execution_args: str = "",
    prepare_command="",
    specific_name="",
    warmup=0,
    runs=-1,
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
        command = f"hyperfine --prepare '{prepare_command}' --warmup {warmup} '{bin_path} {execution_args}' --export-json {filename} --show-output"
    else:
        command = f"hyperfine --warmup {warmup} '{bin_path} {execution_args}' --export-json {filename} --show-output"
    if runs > -1:
        command += f" --runs {runs}"
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
    return (
        result["results"][0]["mean"],
        result["results"][0]["stddev"],
        result["results"][0],
    )


def main():
    os.makedirs(RUN_DIR_PATH, exist_ok=True)

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
    benchmarks = list(sorted(os.listdir(f"{RUN_DIR_PATH}/model")))
    if args.debug:
        benchmarks = benchmarks[:100]
    # random.seed(0)
    # random.shuffle(benchmarks)
    for benchmark in tqdm(benchmarks):
        o3_runtimes_mean, o3_runtimes_std, _ = measure_execution_mean_and_std(
            f"{RUN_DIR_PATH}/O3/{benchmark}",
            execution_args=execution_args,
            warmup=WARMUP,
        )

        o2_runtimes_mean, o2_runtimes_std, _ = measure_execution_mean_and_std(
            f"{RUN_DIR_PATH}/O2/{benchmark}",
            execution_args=execution_args,
            warmup=WARMUP,
        )

        o0_runtimes_mean, o0_runtimes_std, _ = measure_execution_mean_and_std(
            f"{RUN_DIR_PATH}/O0/{benchmark}",
            execution_args=execution_args,
            warmup=WARMUP,
        )

        model_runtimes_mean, model_runtimes_std, _ = measure_execution_mean_and_std(
            f"{RUN_DIR_PATH}/model/{benchmark}",
            execution_args=execution_args,
            warmup=WARMUP,
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
                f"{RUN_DIR_PATH}/O3/{benchmark}", execution_args
            )
            o2_exec_inst = get_executed_instructions(
                f"{RUN_DIR_PATH}/O2/{benchmark}", execution_args
            )
            o0_exec_inst = get_executed_instructions(
                f"{RUN_DIR_PATH}/O0/{benchmark}", execution_args
            )
            model_exec_inst = get_executed_instructions(
                f"{RUN_DIR_PATH}/model/{benchmark}", execution_args
            )
            exec_inst["O3 exec_inst"].append(o3_exec_inst)
            exec_inst["O2 exec_inst"].append(o2_exec_inst)
            exec_inst["O0 exec_inst"].append(o0_exec_inst)
            exec_inst["model exec_inst"].append(model_exec_inst)
            exec_inst["O3 exec_inst imp"].append(
                (o3_exec_inst - model_exec_inst) / o0_exec_inst
            )
            exec_inst["O2 exec_inst imp"].append(
                (o2_exec_inst - model_exec_inst) / o0_exec_inst
            )
            exec_inst["O0 exec_inst imp"].append(
                (o0_exec_inst - model_exec_inst) / o0_exec_inst
            )

    if MEASURE_EXECUTED_INSTRUCTIONS:
        result.update(exec_inst)
    result_df = pd.DataFrame(data=result)
    result_df.to_csv(os.path.join(RUN_DIR_PATH, f"{args.run_name}.csv"))
    print(result_df.drop(columns=["benchmark"]).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name")
    parser.add_argument(
        "--debug",
        help="debug",
        action="store_true",
    )
    # parser.add_argument(
    #     "--runs",
    #     type=int,
    #     default=-1,
    # )
    args = parser.parse_args()

    RUN_DIR_PATH = f"_runtime_eval/{args.run_name}"

    main()
