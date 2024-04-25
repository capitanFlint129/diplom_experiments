import ast
import os.path

import pandas as pd
from tqdm import tqdm

from env.llvm import compile_one_source_with_opt_sequence

BASELINE_SEQUENCES = {
    "O3": ["-O3"],
    "O2": ["-O2"],
    "O1": ["-O1"],
}

JOTAI_SRC_PATH = [
    "/home/flint/diplom/jotai-benchmarks/benchmarks/anghaLeaves",
    "/home/flint/diplom/jotai-benchmarks/benchmarks/anghaMath",
]


def get_benchmark_source_path(benchmark_name):
    for path in JOTAI_SRC_PATH:
        source_path = os.path.join(path, f"{benchmark_name}.c")
        if os.path.isfile(source_path):
            return source_path


def main():
    os.makedirs("data/O0", exist_ok=True)
    os.makedirs("data/O2", exist_ok=True)
    os.makedirs("data/O3", exist_ok=True)
    os.makedirs("data/model", exist_ok=True)
    model_optimizations = pd.read_csv("data/optimizations.csv")
    for row in tqdm(model_optimizations.iterrows(), total=model_optimizations.shape[0]):
        benchmark_name = row[1].benchmark.strip()
        model_sequence = ast.literal_eval(row[1].optimizations)

        source_path = get_benchmark_source_path(benchmark_name)

        try:
            compile_one_source_with_opt_sequence(
                source_path=source_path,
                result_path=f"data/O0/{benchmark_name}",
                sequence=["-O0"],
            )

            compile_one_source_with_opt_sequence(
                source_path=source_path,
                result_path=f"data/O2/{benchmark_name}",
                sequence=["-O2"],
            )

            compile_one_source_with_opt_sequence(
                source_path=source_path,
                result_path=f"data/O3/{benchmark_name}",
                sequence=["-O3"],
            )

            compile_one_source_with_opt_sequence(
                source_path=source_path,
                result_path=f"data/model/{benchmark_name}",
                sequence=model_sequence,
            )
        except Exception as e:
            print(f"Failed to compile {benchmark_name}: {e}")


if __name__ == "__main__":
    main()
