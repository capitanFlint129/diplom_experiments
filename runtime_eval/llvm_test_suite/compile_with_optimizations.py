import argparse
import multiprocessing
import os.path

import pandas as pd
from tqdm import tqdm

from env.performance_optimization.llvm import (
    compile_bc_source_with_opt_sequence,
    linkopts_safe_compile,
)
from runtime_eval.llvm_test_suite.consts import TMP_DATA_DIR
from runtime_eval.llvm_test_suite.generate_optimizations import BenchmarkData


def process_benchmark(benchmark_data: BenchmarkData):
    p = multiprocessing.current_process()
    tmpfilename = f"{p.name}.o"

    def _compile(linkopts):
        if not isinstance(benchmark_data.optimizations, str):
            benchmark_data.optimizations = ""
            print(f"model_sequence is not str: {benchmark_data.optimizations}")
        result_path = f"{RUN_DIR_PATH}/bin/{benchmark_data.name}"
        if "Shootout-C++" in benchmark_data.original_bc_path:
            result_path = f"{RUN_DIR_PATH}/bin/Shootout-C++-{benchmark_data.name}"
        elif "Shootout" in benchmark_data.original_bc_path:
            result_path = f"{RUN_DIR_PATH}/bin/Shootout-{benchmark_data.name}"
        compile_bc_source_with_opt_sequence(
            source_path=benchmark_data.original_bc_path,
            result_path=result_path,
            sequence=benchmark_data.optimizations.split(),
            linkopts=linkopts,
            tmpfilename=tmpfilename,
        )

    try:
        linkopts_safe_compile(
            _compile, [[], ["-lm"], ["-lstdc++"], ["-lm", "-lstdc++"]]
        )
    except Exception as e:
        raise Exception(f"Failed to compile {benchmark_data.original_bc_path}: {e}")


def main():
    optimizations_filename = f"{RUN_DIR_PATH}/optimizations.csv"
    print(optimizations_filename)
    os.makedirs(f"{RUN_DIR_PATH}/bin", exist_ok=True)
    model_optimizations = pd.read_csv(optimizations_filename)
    benchmarks = [
        BenchmarkData(benchmark, path, optimizations)
        for (benchmark, path, optimizations) in zip(
            model_optimizations.benchmark,
            model_optimizations.path,
            model_optimizations.optimizations,
        )
    ]

    if args.debug:
        benchmarks = benchmarks[:10]

    if args.no_parallel:
        for benchmark in tqdm(benchmarks):
            process_benchmark(benchmark)
    else:
        pool = multiprocessing.Pool()
        list(
            tqdm(
                pool.imap(
                    process_benchmark,
                    benchmarks,
                ),
                total=len(model_optimizations),
            )
        )
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name")
    parser.add_argument(
        "--debug",
        help="debug",
        action="store_true",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
    )
    args = parser.parse_args()

    RUN_DIR_PATH = f"{TMP_DATA_DIR}/{args.run_name}"

    main()
