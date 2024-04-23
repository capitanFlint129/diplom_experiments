import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from compiler_gym import CompilerEnv
from compiler_gym.envs import llvm

# noinspection PyUnresolvedReferences
from compiler_gym.wrappers import RuntimePointEstimateReward

from config.config import TrainConfig

LLVM_BINS = "/home/flint/.local/share/compiler_gym/llvm-v0/bin"
MCA_BIN = os.path.join(LLVM_BINS, "llvm-mca")
LLVM_AS_BIN = os.path.join(LLVM_BINS, "llvm-as")
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


class MyEnv:
    def reset(self, benchmark=None) -> None:
        pass

    def step(self, action) -> float:
        pass

    # def multistep(self):
    #     pass

    def get_observation(self, obs_name):
        pass


class CgLlvmMcaEnv(MyEnv):
    def __init__(self, config: TrainConfig, env: CompilerEnv):
        self._cg_env = env
        self._config = config

    def reset(self, benchmark=None):
        attempts = 100
        self._opts = []
        for i in range(attempts):
            try:
                self._cg_env.reset(benchmark=benchmark)
                self._rblock_throughput_initial = get_rblock_throughput_ir(
                    self._cg_env.observation["Ir"]
                )
                return
            except ValueError as e:
                print(e, file=sys.stderr)
            except Exception as e:
                print(e, file=sys.stderr)
        raise Exception(f"Failed to reset after {attempts} attempts")

    def step(self, action):
        rblock_throughput_before = get_rblock_throughput_ir(
            self._cg_env.observation["Ir"]
        )
        self._cg_env.step(action)
        rblock_throughput_after = get_rblock_throughput_ir(
            self._cg_env.observation["Ir"]
        )
        return (
            rblock_throughput_before - rblock_throughput_after
        ) / self._rblock_throughput_initial

    def get_observation(self, obs_name):
        if obs_name == "IR2Vec":
            return get_ir2vec(self._cg_env.observation["Ir"])
        else:
            return self._cg_env.observation[obs_name]


class LlvmMcaEnv(MyEnv):
    def __init__(self, config: TrainConfig, env: CompilerEnv):
        self._cg_env = env
        self._tmp_dir = f"mca_env_tmp"
        os.makedirs(self._tmp_dir, exist_ok=True)
        self._filename = "tmpfile.bc"
        self._filename_o3 = "tmpfile_o3.bc"
        self._filepath = os.path.join(self._tmp_dir, self._filename)
        self._filepath_o3 = os.path.join(self._tmp_dir, self._filename_o3)
        self._config = config
        self._opts = []

    def reset(self, benchmark=None):
        attempts = 100
        self._opts = []
        for i in range(attempts):
            try:
                self._cg_env.reset(benchmark=benchmark)
                # o3_rb = self._get_03_rb()
                # self._o3_rb = o3_rb
                # -O3 -Xclang -disable-llvm-passes
                proc = subprocess.run(
                    [
                        CLANG_BIN,
                        O0,
                        "--target=x86_64",
                        "-Xclang",
                        # "-disable-llvm-passes",
                        "-disable-O0-optnone",
                        "-S",
                        "-emit-llvm",
                        "-x",
                        "c",
                        "-",
                        "-o",
                        self._filepath,
                    ],
                    input=self._cg_env.benchmark.source.encode(),
                    capture_output=True,
                )
                if proc.returncode != 0:
                    print(proc.stderr)
                proc = subprocess.run(
                    [
                        LLVM_AS_BIN,
                        self._filepath,
                        "-o",
                        self._filepath,
                    ],
                    capture_output=True,
                )
                if proc.returncode != 0:
                    print(proc.stderr)
                else:
                    self._rblock_throughput_initial = get_rblock_throughput_bc(
                        self._filepath
                    )
                    # print(f"O3 - {o3_rb} | O0 - {self._rblock_throughput_initial}")
                    return
            except ValueError as e:
                print(e, file=sys.stderr)
            except Exception as e:
                print(e, file=sys.stderr)
        raise Exception(f"Failed to reset after {attempts} attempts")

    def _get_03_rb(self):
        proc = subprocess.run(
            [
                CLANG_BIN,
                O3,
                "--target=x86_64",
                "-S",
                "-emit-llvm",
                "-x",
                "c",
                "-",
                "-o",
                self._filepath_o3,
            ],
            input=self._cg_env.benchmark.source.encode(),
            capture_output=True,
        )
        if proc.returncode != 0:
            print(proc.stderr)
            return None
        else:
            return get_rblock_throughput_bc(self._filepath_o3)

    def step(self, action):
        rblock_throughput_before = get_rblock_throughput_bc(self._filepath)
        if isinstance(action, str):
            flag = action
        else:
            flag = self._config.actions[action]
        self._opts.append(flag)
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
        rblock_throughput_after = get_rblock_throughput_bc(self._filepath)
        return (
            rblock_throughput_before - rblock_throughput_after
        ) / self._rblock_throughput_initial

    def multistep(self):
        pass

    def get_observation(self, obs_name):
        if obs_name == "InstCountNorm":
            space = self._cg_env.observation.spaces["InstCount"]
            bitcode = Path(self._filepath)
            obs = llvm.compute_observation(space, bitcode)
            return obs[1:] / obs[0]
        elif obs_name == "IR2Vec":
            return get_ir2vec(self._cg_env.observation["Ir"])
        else:
            space = self._cg_env.observation.spaces[obs_name]
            bitcode = Path(self._filepath)
            return llvm.compute_observation(space, bitcode)


def get_ir2vec(ir_text: str) -> np.ndarray:
    ir2vec_bin = "/home/flint/diplom/IR2Vec/build/bin/ir2vec"
    seed_emb_path = (
        "/home/flint/diplom/IR2Vec/vocabulary/seedEmbeddingVocab-300-llvm10.txt"
    )
    with tempfile.NamedTemporaryFile("w") as ll_file:
        with tempfile.NamedTemporaryFile("r") as result_file:
            ll_file.write(ir_text)
            ll_file.flush()
            proc = subprocess.run(
                [
                    ir2vec_bin,
                    "-fa",
                    "-vocab",
                    seed_emb_path,
                    "-o",
                    result_file.name,
                    "-level",
                    "p",
                    ll_file.name,
                ],
                capture_output=True,
            )
            if proc.returncode != 0:
                raise Exception("IR2Vec failed")
            observation = np.loadtxt(result_file.name)
    return observation


def get_mca_result_from_ir(bc_path):
    proc = subprocess.run(
        f"{LLC_BIN} {bc_path} -o - | llvm-mca",
        capture_output=True,
        shell=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def get_mca_result_from_ir_str(ir):
    proc = subprocess.run(
        f"{LLC_BIN} -o - | llvm-mca",
        input=ir,
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


def get_rblock_throughput_bc(bc_path):
    return parse_rblock_throughput(get_mca_result_from_ir(bc_path))


def get_rblock_throughput_ir(ir):
    return parse_rblock_throughput(get_mca_result_from_ir_str(ir))


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
