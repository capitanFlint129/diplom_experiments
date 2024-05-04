import argparse
import os.path
import random
from multiprocessing import Pool
from subprocess import TimeoutExpired

import pandas as pd
import torch
from tqdm import tqdm

from config.config import TrainConfig, TEST_BENCHMARKS_DIR
from utils import (
    get_agent,
    get_model_path,
    optimize_with_model,
)
from utils import make_env

DATASET_URI = "benchmark://jotaibench-v0"


def init_worker(function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    function.env = make_env(config)
    function.agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path(args.run_name),
    )


def process_benchmark(name):
    env = process_benchmark.env
    agent = process_benchmark.agent
    try:
        env.reset(benchmark=f"{DATASET_URI}/{name}")
    except ValueError as e:
        print(e)
        return None, None

    try:
        flags = optimize_with_model(
            config,
            agent,
            env,
            eval_mode=True,
            iters=config.episode_length,
            print_debug=False,
        )
    except TimeoutExpired as e:
        print(f"IR2vec timeout skip benchmark: {e}")
        return None, None
    except Exception as e:
        print(f"Exception. Skip benchmark: {e}")
        return None, None

    return name, flags


def main():
    pool = Pool(initializer=init_worker, initargs=(process_benchmark,), processes=None)

    with open(
        os.path.join(TEST_BENCHMARKS_DIR, f"test_benchmarks_{args.run_name}.txt"), "r"
    ) as inf:
        benchmarks_names = inf.readlines()
        benchmarks_names = [name.strip() for name in benchmarks_names]

    random.seed(129)
    random.shuffle(benchmarks_names)

    if args.debug:
        benchmarks_names = benchmarks_names[:20]
    elif args.n > -1:
        benchmarks_names = benchmarks_names[: args.n]

    results = list(
        tqdm(
            pool.imap(process_benchmark, benchmarks_names), total=len(benchmarks_names)
        )
    )
    pool.close()

    final_benchmarks_names, optimizations = zip(*results)
    final_benchmarks_names = [el for el in final_benchmarks_names if el is not None]
    optimizations = [el for el in optimizations if el is not None]
    optimizations = [
        opt if isinstance(opt, str) else [el for el in opt if el != "noop"]
        for opt in optimizations
    ]
    optimizations = [
        opt if isinstance(opt, str) else " ".join(opt) for opt in optimizations
    ]

    result = pd.DataFrame(
        data={
            "benchmark": final_benchmarks_names,
            "optimizations": optimizations,
        }
    )
    os.makedirs(f"_runtime_eval/{args.run_name}", exist_ok=True)
    result.to_csv(f"_runtime_eval/{args.run_name}/optimizations.csv")


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
    args = parser.parse_args()
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
