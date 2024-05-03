import os
import subprocess
import sys
from pathlib import Path

from compiler_gym import CompilerEnv
from compiler_gym.envs import llvm

from config.config import TrainConfig, LLVM_AS_BIN, OPT_BIN
from env.my_env import MyEnv
from env.performance_optimization.llvm import (
    clang_compile_to_ir,
    clang_compile_to_ir_o0,
)
from observation.utils import get_rblock_throughput_bc, get_rblock_throughput_ir
from utils import get_observation_from_cg, get_ir2vec

O0 = "-O0"
O1 = "-O1"
O2 = "-O2"
O3 = "-O3"
OZ = "-Oz"


class CgLlvmMcaEnv(MyEnv):
    def __init__(self, config: TrainConfig, env: CompilerEnv, debug: bool = False):
        self._cg_env = env
        self._config = config
        self._debug = debug
        self._rblock_throughput_prev = None

    def get_cur_ir(self) -> CompilerEnv:
        return self._cg_env.observation["Ir"]

    def reset(self, benchmark=None, val=False):
        attempts = 100
        self._opts = []
        for i in range(attempts):
            try:
                self._cg_env.reset(benchmark=benchmark)
                self._rblock_throughput_initial = get_rblock_throughput_ir(
                    self._cg_env.observation["Ir"]
                )
                self._rblock_throughput_prev = self._rblock_throughput_initial
                return
            except ValueError as e:
                print(e, file=sys.stderr)
            except Exception as e:
                print(e, file=sys.stderr)
        raise Exception(f"Failed to reset after {attempts} attempts")

    def step(self, flags: list[str]):
        if flags[0] == "noop":
            if self._debug:
                print(
                    f"reward: {0} - throughput: {self._rblock_throughput_prev} - throughput_prev: {self._rblock_throughput_prev} - throughput_initial: {self._rblock_throughput_initial}"
                )
            return 0
        if len(flags) > 1:
            self._cg_env.multistep(
                [self._cg_env.action_space.flags.index(f) for f in flags]
            )
        else:
            self._cg_env.step(self._cg_env.action_space.flags.index(flags[0]))

        rblock_throughput = get_rblock_throughput_ir(self._cg_env.observation["Ir"])
        reward = (
            self._rblock_throughput_prev - rblock_throughput
        ) / self._rblock_throughput_initial
        if self._debug:
            print(
                f"reward: {reward} - throughput: {rblock_throughput} - throughput_prev: {self._rblock_throughput_prev} - throughput_initial: {self._rblock_throughput_initial}"
            )
        self._rblock_throughput_prev = rblock_throughput
        return reward

    def get_observation(self, obs_name):
        return get_observation_from_cg(self._cg_env, obs_name)


class LlvmMcaEnv(MyEnv):
    def __init__(self, config: TrainConfig, env: CompilerEnv):
        self._cg_env = env
        self._tmp_dir = f"_mca_env_tmp"
        os.makedirs(self._tmp_dir, exist_ok=True)
        self._filename = "tmpfile.bc"
        self._filename_o3 = "tmpfile_o3.bc"
        self._filepath = os.path.join(self._tmp_dir, self._filename)
        self._filepath_o3 = os.path.join(self._tmp_dir, self._filename_o3)
        self._config = config
        self._opts = []

    def get_cur_ir(self) -> CompilerEnv:
        return self._cg_env.observation["Ir"]

    def reset(self, benchmark=None, val=False):
        attempts = 100
        self._opts = []
        for i in range(attempts):
            try:
                self._cg_env.reset(benchmark=benchmark)
                # o3_rb = self._get_03_rb()
                # self._o3_rb = o3_rb
                # -O3 -Xclang -disable-llvm-passes
                clang_compile_to_ir_o0(
                    source=self._cg_env.benchmark.source, result_path=self._filepath
                )
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
        clang_compile_to_ir(
            source=self._cg_env.benchmark.source,
            level=O3,
            result_path=self._filepath_o3,
        )
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
