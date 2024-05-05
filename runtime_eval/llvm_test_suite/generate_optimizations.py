import argparse
import dataclasses
import os
import os.path
from multiprocessing import Pool
from subprocess import TimeoutExpired

import pandas as pd
import torch
from tqdm import tqdm

from config.config import TrainConfig
from runtime_eval.llvm_test_suite.consts import LLVM_TEST_SUITE_PATH, TMP_DATA_DIR
from utils import (
    get_agent,
    get_model_path,
    optimize_with_model,
    make_env,
)


def _find_bc_files(directory: str) -> list:
    """
    Recursively traverses a directory and its subdirectories, and returns a list of all files with the .bc extension.

    Args:
        directory (str): The path to the directory to traverse.

    Returns:
        list: A list of file paths for all .bc files found in the directory and its subdirectories.
    """
    bc_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bc"):
                bc_files.append(os.path.join(root, file))
    return bc_files


@dataclasses.dataclass
class BenchmarkData:
    name: str
    original_bc_path: str
    optimizations: str = ""


def init_worker(function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    function.env = make_env(config)
    function.agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path(args.run_name),
    )


def process_benchmark(benchmark_data: BenchmarkData, env=None, agent=None):
    if env is None:
        env = process_benchmark.env
    if agent is None:
        agent = process_benchmark.agent

    try:
        benchmark = env.make_benchmark(benchmark_data.original_bc_path)
        env.reset(benchmark=benchmark)
    except ValueError as e:
        print(e)
        return None

    try:
        flags = optimize_with_model(
            config,
            agent,
            env,
            eval_mode=True,
            iters=config.episode_length,
            print_debug=False,
            hack=args.hack,
        )
    except TimeoutExpired as e:
        print(f"IR2vec timeout skip benchmark: {e}")
        return None, None
    except Exception as e:
        print(f"Exception. Skip benchmark: {e}")
        return None, None

    optimizations = []
    for f in flags:
        if f != "noop":
            optimizations += f.split()
    benchmark_data.optimizations = " ".join(optimizations)
    return benchmark_data


def main():

    bc_files = _find_bc_files(LLVM_TEST_SUITE_PATH)

    benchmarks = [
        BenchmarkData(os.path.basename(el).split(".")[0], el) for el in bc_files
    ]

    if args.debug:
        benchmarks = benchmarks[:10]
    elif args.n > -1:
        benchmarks = benchmarks[: args.n]

    if args.no_parallel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = make_env(config)
        agent = get_agent(
            config,
            device,
            policy_net_path=get_model_path(args.run_name),
        )
        results = []
        for benchmark in tqdm(benchmarks):
            results.append(process_benchmark(benchmark, env, agent))
    else:
        pool = Pool(
            initializer=init_worker, initargs=(process_benchmark,), processes=None
        )
        results = list(
            tqdm(pool.imap(process_benchmark, benchmarks), total=len(benchmarks))
        )
        pool.close()

    final_benchmarks: list[BenchmarkData] = [el for el in results if el is not None]

    result = pd.DataFrame(
        data={
            "benchmark": [el.name for el in final_benchmarks],
            "path": [el.original_bc_path for el in final_benchmarks],
            "optimizations": [el.optimizations for el in final_benchmarks],
        }
    )
    os.makedirs(f"{TMP_DATA_DIR}/{args.run_name}", exist_ok=True)
    result.to_csv(f"{TMP_DATA_DIR}/{args.run_name}/optimizations.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name")
    parser.add_argument(
        "--n",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--debug",
        help="debug",
        action="store_true",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
    )
    parser.add_argument(
        "--hack",
        help="eval_mode",
        action="store_true",
    )
    args = parser.parse_args()
    config = TrainConfig.load_config(args.run_name)

    main()
