import argparse
import multiprocessing
import os

import pandas as pd
from tqdm import tqdm

from env.cfg_grind import get_executed_instructions

DATASET_URI = "benchmark://jotaibench-v0"
RESULT_DIR = "_eval_results"


parser = argparse.ArgumentParser()
parser.add_argument("run_name", help="run name")
parser.add_argument(
    "--debug",
    help="debug",
    action="store_true",
)
args = parser.parse_args()


def process_benchmark(benchmark_name: str):
    execution_args = "0"
    try:
        o3_exec_inst = get_executed_instructions(
            f"data/O3/{benchmark_name}", execution_args
        )
        o2_exec_inst = get_executed_instructions(
            f"data/O2/{benchmark_name}", execution_args
        )
        o0_exec_inst = get_executed_instructions(
            f"data/O0/{benchmark_name}", execution_args
        )
        model_exec_inst = get_executed_instructions(
            f"data/model/{benchmark_name}", execution_args
        )
    except Exception as e:
        print(f"Failed to process {benchmark_name}: {e}")
        return benchmark_name, None, None, None, None
    return benchmark_name, o3_exec_inst, o2_exec_inst, o0_exec_inst, model_exec_inst


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    result = {
        "benchmark": [],
        "O3 exec_inst": [],
        "O2 exec_inst": [],
        "O0 exec_inst": [],
        "model exec_inst": [],
        "O3 exec_inst imp": [],
        "O2 exec_inst imp": [],
        "O0 exec_inst imp": [],
    }

    benchmarks = list(sorted(os.listdir("data/model")))
    # random.seed(0)
    # random.shuffle(benchmarks)

    if args.debug:
        benchmarks = benchmarks[:100]

    pool = multiprocessing.Pool(processes=1)
    results = list(
        tqdm(
            pool.imap(
                process_benchmark,
                benchmarks,
            ),
            total=len(benchmarks),
        )
    )
    pool.close()

    for (
        benchmark_name,
        o3_exec_inst,
        o2_exec_inst,
        o0_exec_inst,
        model_exec_inst,
    ) in results:
        if (
            o3_exec_inst is None
            or o2_exec_inst is None
            or o0_exec_inst is None
            or model_exec_inst is None
        ):
            continue
        result["benchmark"].append(benchmark_name)
        result["O3 exec_inst"].append(o3_exec_inst)
        result["O2 exec_inst"].append(o2_exec_inst)
        result["O0 exec_inst"].append(o0_exec_inst)
        result["model exec_inst"].append(model_exec_inst)
        result["O3 exec_inst imp"].append(
            (o3_exec_inst - model_exec_inst) / o0_exec_inst
        )
        result["O2 exec_inst imp"].append(
            (o2_exec_inst - model_exec_inst) / o0_exec_inst
        )
        result["O0 exec_inst imp"].append(
            (o0_exec_inst - model_exec_inst) / o0_exec_inst
        )

    result_df = pd.DataFrame(data=result)
    result_df.to_csv(os.path.join(RESULT_DIR, f"{args.run_name}.csv"))
    print(result_df.drop(columns=["benchmark"]).mean())


if __name__ == "__main__":
    main()
