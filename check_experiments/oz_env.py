import os
import subprocess
import tempfile
from typing import Any, Union
from compiler_gym.util.timer import Timer

import pandas as pd

from check_experiments.actions import OZ_FLAGS_SEQUENCE
from config import LLVM_BINS_PATH

OPT_UTIL_NAME = "opt"
OZ_FLAG = "-Oz"


def apply_oz(ir_source: str) -> str:
    return apply_passes(ir_source, [OZ_FLAG])


def apply_passes(ir_source: str, passes_list: list[str], use_bc: bool = False) -> str:
    command = [get_full_util_path(OPT_UTIL_NAME)] + passes_list
    if not use_bc:
        command.append("-S")
    kwargs = {}
    if not use_bc:
        kwargs["encoding"] = "utf-8"
    proc = subprocess.run(
        command,
        input=ir_source,
        capture_output=True,
        **kwargs,
    )
    return proc.stdout


def env_action(
    env, ir_source: str, passes_list: list[str], observations: list[str]
) -> tuple[str, dict[str, Any]]:
    new_ir_source = apply_passes(ir_source, passes_list)
    observations_values = get_observations_by_raw_ir(env, new_ir_source, observations)
    return new_ir_source, observations_values


def get_observations_by_raw_ir(
    env, ir_source: Union[str, bytes], observations: list[str], use_bc: bool = False
) -> dict[str, Any]:
    if use_bc:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, "file.bc")
            with open(file_path, "wb") as file:
                file.write(ir_source)
            benchmark = env.make_benchmark(file_path)
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, "file.ll")
            with open(file_path, "w") as file:
                file.write(ir_source)
            benchmark = env.make_benchmark(file_path)
    env.reset(benchmark=benchmark)
    observations_values = {
        observation: env.observation[observation] for observation in observations
    }
    return observations_values


def get_full_util_path(util_name: str) -> str:
    return os.path.join(LLVM_BINS_PATH, util_name)


# if __name__ == "__main__":
#     from config import TrainConfig
#     from utils import make_env, prepare_datasets
#
#     config = TrainConfig()
#     observations_names = ["Ir", "IrInstructionCountOz", "IrInstructionCount"]
#     with make_env(config) as env:
#         env_clone = env.fork()
#         benchmarks, _, _ = prepare_datasets(
#             env,
#             config.datasets,
#             train_val_test_split=False,
#             skipped=set(config.skipped_benchmarks),
#         )
#         data = {
#             "benchmark": [],
#             "ir_lines_number": [],
#             "IrInstructionCountOz": [],
#             "IrInstructionCount": [],
#             "opt_ir_lines_number": [],
#             "opt_IrInstructionCountOz": [],
#             "opt_IrInstructionCount": [],
#             "time": [],
#         }
#         for benchmark in benchmarks:
#             print(benchmark)
#             env.reset(benchmark=benchmark)
#             ir_source = str(env.observation["Ir"])
#
#             data["benchmark"].append(str(benchmark))
#             data["ir_lines_number"].append(ir_source.count("\n"))
#             observations = get_observations_by_raw_ir(
#                 env, ir_source, observations_names
#             )
#             data["IrInstructionCountOz"].append(observations["IrInstructionCountOz"])
#             data["IrInstructionCount"].append(observations["IrInstructionCount"])
#
#             with Timer() as timer:
#                 compressed_ir_source, observations = env_action(
#                     env,
#                     ir_source,
#                     OZ_FLAGS_SEQUENCE,
#                     observations=observations_names,
#                 )
#             data["opt_IrInstructionCountOz"].append(
#                 observations["IrInstructionCountOz"]
#             )
#             data["opt_IrInstructionCount"].append(observations["IrInstructionCount"])
#             data["time"].append(timer.time)
#         pd_data = pd.DataFrame(data=data)
#         pd_data["opt_to_cgOz_comparise"] = (
#             pd_data["opt_IrInstructionCount"] / pd_data["IrInstructionCountOz"]
#         )
#         pd_data.to_csv("oz_results.csv")
#         env_clone.close()


"""
oz_compressed_ir = apply_oz(ir_source)
instruction_count = oz_compressed_ir.count("\n")
with open("file.ll", "w") as ouf:
    ouf.write(oz_compressed_ir)
compressed_benchmark = env.make_benchmark("file.ll")
env_clone.reset(benchmark=compressed_benchmark)
assert (
    env_clone.observation["IrInstructionCountOz"]
    == env.observation["IrInstructionCountOz"]
)
assert (
    env.observation["IrInstructionCountOz"]
    == env_clone.observation["IrInstructionCount"]
), f"-Oz diff: {env.observation['IrInstructionCountOz'] - env_clone.observation['IrInstructionCount']}"
"""
