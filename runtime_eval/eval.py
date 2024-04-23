import os
import numpy as np
import subprocess

import pandas as pd

DATASET_URI = "benchmark://jotaibench-v0"
RUN_NAME = "lively-snowflake-49"


RUNS = 30


def parse_exec_time(time_stdout) -> float:
    # real	0m0,001s
    pass


def get_runtimes(bin_path, runs=RUNS) -> list[float]:
    runtimes = []
    for i in range(runs):
        proc = subprocess.run(["time", bin_path], capture_output=True)
        runtimes.append(parse_exec_time(proc.stdout.decode()))
        if proc.returncode != 0:
            raise Exception("run failed")
    return runtimes


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
    benchmarks = sorted(os.listdir("data/model"))
    for benchmark in benchmarks:
        o3_runtimes = get_runtimes(f"time data/O3/{benchmark}")
        o3_runtimes_mean = np.mean(o3_runtimes)
        o3_runtimes_std = np.std(o3_runtimes)

        o0_runtimes = get_runtimes(f"time data/O0/{benchmark}")
        o0_runtimes_mean = np.mean(o0_runtimes)
        o0_runtimes_std = np.std(o0_runtimes)

        model_runtimes = get_runtimes(f"time data/model/{benchmark}")
        model_runtimes_mean = np.mean(model_runtimes)
        model_runtimes_std = np.std(model_runtimes)

    result_df = pd.DataFrame(data=result)
    print(result_df.drop(columns=["benchmark"]).mean())


if __name__ == "__main__":
    main()
