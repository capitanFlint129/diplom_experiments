import os
import subprocess
from pathlib import Path

from compiler_gym import CompilerEnv
from compiler_gym.envs import llvm

# noinspection PyUnresolvedReferences
from compiler_gym.wrappers import RuntimePointEstimateReward

from config.config import TrainConfig

LLVM_BINS = "/home/flint/.local/share/compiler_gym/llvm-v0/bin"
MCA_BIN = os.path.join(LLVM_BINS, "llvm-mca")
# CLANG_BIN = os.path.join(LLVM_BINS, 'clang')
CLANG_BIN = "clang"
OPT_BIN = os.path.join(LLVM_BINS, "opt")
LLC_BIN = os.path.join(LLVM_BINS, "llc")
JOTAI_BENCHMARKS_DIRS = [
    "/home/flint/diplom/jotai-benchmarks/benchmarks/anghaLeaves/",
    "/home/flint/diplom/jotai-benchmarks/benchmarks/anghaMath/",
]
TMP_DIR = "tmp"

O0 = "-O0"
O1 = "-O1"
O2 = "-O2"
O3 = "-O3"
OZ = "-Oz"


class LlvmMcaEnv:
    def __init__(self, config: TrainConfig, env: CompilerEnv, observation_name: str):
        self._cg_env = env
        self._observation_name = observation_name
        self._tmp_dir = f"{self.__class__}_env_tmp"
        os.makedirs(self._tmp_dir)
        self._filename = "tmpfile.bc"
        self._filepath = os.path.join(self._tmp_dir, self._filename)
        self._config = config

    def reset(self):
        attempts = 100
        for i in range(attempts):
            self._cg_env.reset()
            proc = subprocess.run(
                [
                    CLANG_BIN,
                    O0,
                    "--target=x86_64",
                    "-emit-llvm",
                    "-o",
                    self._filepath,
                ],
                input=self._cg_env.benchmark.source,
                capture_output=True,
            )
            if proc.returncode != 0:
                print(proc.stderr)
            else:
                self._rblock_throughput_initial = get_rblock_throughput(self._filepath)
                return
        raise Exception(f"Failed to reset after {attempts} attempts")

    def step(self, action):
        rblock_throughput_before = get_rblock_throughput(self._filepath)
        flag = self._config.actions[action]
        proc = subprocess.run(
            [
                OPT_BIN,
                flag,
                self._filepath,
                "-o",
                self._filepath,
            ],
        )
        if proc.returncode != 0:
            print(proc.stderr)
            raise Exception(f"Opt step failed {self._cg_env.benchmark}")
        rblock_throughput_after = get_rblock_throughput(self._filepath)
        return (
            rblock_throughput_before - rblock_throughput_after
        ) / self._rblock_throughput_initial

    def multistep(self):
        pass

    def get_observation(self, obs_name):
        space = self._cg_env.observation.spaces[obs_name]
        bitcode = Path(self._filepath)
        return llvm.compute_observation(space, bitcode)


def get_mca_result_from_ir(bc_path):
    proc = subprocess.run(
        f"{LLC_BIN} {bc_path} | llvm-mca",
        capture_output=True,
        shell=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def parse_rblock_throughput(mca_output):
    line = mca_output.split("\n", maxsplit=9)[8]
    # print(line)
    return float(line.split()[2])


def get_rblock_throughput(bc_path):
    return parse_rblock_throughput(get_mca_result_from_ir(bc_path))


# def get_mca_result(source_path, optimization):
#     proc = subprocess.run(
#         f"{CLANG_BIN} {source_path} {optimization} --target=x86_64 -S -fsanitize=address,undefined,signed-integer-overflow -fno-sanitize-recover=all -o - | llvm-mca",
#         capture_output=True,
#         shell=True,
#         encoding="utf-8",
#     )
#     assert proc.returncode == 0, proc.stderr
#     return proc.stdout


# def parse_rblock_throughput(mca_output):
#     line = mca_output.split('\n', maxsplit=9)[8]
#     # print(line)
#     return float(line.split()[2])

# def get_rblock_throughput(source_path, optimization):
#     return parse_rblock_throughput(get_mca_result(source_path, optimization))
