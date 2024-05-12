import argparse
import dataclasses
import os.path
import random
from multiprocessing import Pool
from subprocess import TimeoutExpired
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from config.config import TrainConfig, TEST_BENCHMARKS_DIR
from runtime_eval.jotai.consts import TMP_DATA_DIR
from utils import (
    get_agent,
    get_model_path,
    optimize_with_model,
)
from utils import make_env

DATASET_URI = "benchmark://jotaibench-v0"


@dataclasses.dataclass
class BenchmarkData:
    name: str
    # original_bc_path: str
    optimizations: str = ""


def init_worker(function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    function.env = make_env(config)
    if args.o3 or args.o0:
        function.agent = None
    else:
        function.agent = get_agent(
            config,
            device,
            policy_net_path=get_model_path(
                args.run_name, last_checkpoint=args.last_iter
            ),
        )


def process_benchmark(
    benchmark_data: BenchmarkData, env=None, agent=None
) -> Optional[BenchmarkData]:
    if env is None:
        env = process_benchmark.env
    if agent is None:
        agent = process_benchmark.agent
    try:
        env.reset(benchmark=f"{DATASET_URI}/{benchmark_data.name}")
    except ValueError as e:
        print(e)
        return None

    if args.o3:
        flags = ["-O3"]
    elif args.o0:
        flags = [""]
    else:
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
            return None
        except Exception as e:
            print(f"Exception. Skip benchmark: {e}")
            return None

    optimizations = []
    for f in flags:
        if f != "noop":
            optimizations += f.split()
    benchmark_data.optimizations = " ".join(optimizations)
    return benchmark_data


def main():
    with open(
        os.path.join(TEST_BENCHMARKS_DIR, f"test_benchmarks_{args.run_name}.txt"), "r"
    ) as inf:
        benchmarks_names = inf.readlines()
        benchmarks_names = [name.strip() for name in benchmarks_names]
        benchmarks = [BenchmarkData(name) for name in benchmarks_names]

    random.seed(129)
    random.shuffle(benchmarks)

    if args.debug:
        benchmarks = benchmarks[:20]
    elif args.n > -1:
        benchmarks = benchmarks[: args.n]

    if args.no_parallel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = make_env(config)
        if args.o3 or args.o0:
            agent = None
        else:
            agent = get_agent(
                config,
                device,
                policy_net_path=get_model_path(
                    args.run_name, last_checkpoint=args.last_iter
                ),
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
    parser.add_argument(
        "--last_iter",
        help="last_iter",
        action="store_true",
    )
    parser.add_argument(
        "--o3",
        help="ignore model and compile with -O3",
        action="store_true",
    )
    parser.add_argument(
        "--o0",
        action="store_true",
    )
    args = parser.parse_args()
    if args.o3 or args.o0:
        config = TrainConfig()
    else:
        config = TrainConfig.load_config(args.run_name)

    main()


# import argparse
# import os.path
# from subprocess import TimeoutExpired
#
# import pandas as pd
# import torch
# from tqdm import tqdm
#
# from config.config import TrainConfig, TEST_BENCHMARKS_DIR
# from utils import (
#     get_agent,
#     get_model_path,
#     optimize_with_model,
# )
# from utils import make_env
#
# DATASET_URI = "benchmark://jotaibench-v0"
# # RUN_NAME = "peach-hill-96"
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument("run_name", help="run name")
# args = parser.parse_args()
#
#
# def main():
#     config = TrainConfig()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     env = make_env(config)
#     with open(
#         os.path.join(TEST_BENCHMARKS_DIR, f"test_benchmarks_{args.run_name}.txt"), "r"
#     ) as inf:
#         benchmarks_names = inf.readlines()
#     optimizations = []
#     final_benchmarks_names = []
#
#     agent = get_agent(
#         config,
#         device,
#         policy_net_path=get_model_path(args.run_name),
#     )
#
#     for name in tqdm(benchmarks_names):
#         name = name.strip()
#         try:
#             env.reset(benchmark=f"{DATASET_URI}/{name}")
#         except ValueError as e:
#             print(e)
#             continue
#         try:
#             flags = optimize_with_model(
#                 config,
#                 agent,
#                 env,
#                 eval_mode=True,
#                 iters=config.episode_length,
#                 print_debug=False,
#             )
#         except TimeoutExpired as e:
#             print(f"IR2vec timeout skip benchmark: {e}")
#             continue
#         except Exception as e:
#             print(f"Exception. Skip benchmark: {e}")
#             continue
#         optimizations.append(flags)
#         final_benchmarks_names.append(name)
#     result = pd.DataFrame(
#         data={
#             "benchmark": final_benchmarks_names,
#             "optimizations": optimizations,
#         }
#     )
#     result.to_csv(f"data/optimizations_{args.run_name}.csv")
#
#
# if __name__ == "__main__":
#     main()
