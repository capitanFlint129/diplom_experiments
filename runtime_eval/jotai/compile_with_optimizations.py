import argparse
import multiprocessing
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

parser = argparse.ArgumentParser()
parser.add_argument("run_name", help="run name")
parser.add_argument(
    "--debug",
    help="debug",
    action="store_true",
)
args = parser.parse_args()


def get_benchmark_source_path(benchmark_name):
    for path in JOTAI_SRC_PATH:
        source_path = os.path.join(path, f"{benchmark_name}.c")
        if os.path.isfile(source_path):
            return source_path


def process_benchmark(data: tuple[str, str]):
    benchmark_name, model_sequence = data
    p = multiprocessing.current_process()
    tmpfilename = f"{p.name}.o"

    def _compile(linkopts):
        compile_one_source_with_opt_sequence(
            source_path=source_path,
            result_path=f"data/O0/{benchmark_name}",
            sequence=[],
            linkopts=linkopts,
            tmpfilename=tmpfilename,
        )
        compile_one_source_with_opt_sequence(
            source_path=source_path,
            result_path=f"data/O2/{benchmark_name}",
            sequence=["-O2"],
            linkopts=linkopts,
            tmpfilename=tmpfilename,
        )
        compile_one_source_with_opt_sequence(
            source_path=source_path,
            result_path=f"data/O3/{benchmark_name}",
            sequence=["-O3"],
            linkopts=linkopts,
            tmpfilename=tmpfilename,
        )
        compile_one_source_with_opt_sequence(
            source_path=source_path,
            result_path=f"data/model/{benchmark_name}",
            sequence=model_sequence.split(),
            linkopts=linkopts,
            tmpfilename=tmpfilename,
        )

    source_path = get_benchmark_source_path(benchmark_name)

    try:
        _compile([])
    except Exception as e:
        print(f"Failed to compile {benchmark_name} try to use -lm: {e}")
        try:
            _compile(["-lm"])
        except Exception as e:
            print(f"Failed to compile {benchmark_name}: {e}")


def main():
    filename = f"data/optimizations_{args.run_name}.csv"
    print(filename)
    os.makedirs("data/O0", exist_ok=True)
    os.makedirs("data/O2", exist_ok=True)
    os.makedirs("data/O3", exist_ok=True)
    os.makedirs("data/model", exist_ok=True)
    model_optimizations = pd.read_csv(filename)

    pool = multiprocessing.Pool()
    list(
        tqdm(
            pool.imap(
                process_benchmark,
                zip(model_optimizations.benchmark, model_optimizations.optimizations),
            ),
            total=len(model_optimizations),
        )
    )
    pool.close()


if __name__ == "__main__":
    main()
