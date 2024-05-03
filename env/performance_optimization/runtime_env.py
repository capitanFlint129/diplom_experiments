import json
import os
import subprocess

import numpy as np
from compiler_gym import CompilerEnv

from config.config import TrainConfig
from env.my_env import MyEnv
from env.performance_optimization.cfg_grind import (
    compile_and_get_instructions,
    compile_and_get_instructions_no_sequence,
)
from env.performance_optimization.llvm import compile_lm_safely
from utils import get_observation_from_cg

O0 = "-O0"
O1 = "-O1"
O2 = "-O2"
O3 = "-O3"
OZ = "-Oz"


class RuntimeEnv(MyEnv):
    def __init__(self, config: TrainConfig, env: CompilerEnv, debug=False):
        self._cg_env = env
        self._tmp_dir = f"_cfg_grind_env_tmp"
        os.makedirs(self._tmp_dir, exist_ok=True)
        self._filename = "tmpfile"
        self._bin_filepath = os.path.join(self._tmp_dir, self._filename)
        self._config = config
        self._runtime_o3 = None

        self._runtime_initial = None
        self._runtime_prev = None
        self._runtime_baseline = None

        self._debug = debug

    def get_cur_ir(self) -> CompilerEnv:
        return self._cg_env.observation["Ir"]

    def reset(self, benchmark=None, val=False):
        attempts = 100
        for i in range(attempts):
            self._cg_env.reset(benchmark=benchmark)
            # O3
            if self._debug:
                self._runtime_o3 = self._compile_and_get_runtime_seq(sequence=[O3])
            self._runtime_baseline = self._compile_and_get_runtime_seq(sequence=[O3])
            # Initial
            self._runtime_initial = self._compile_and_get_runtime()
            if self._runtime_initial < 1e-6 or self._runtime_baseline < 1e-6:
                raise Exception("Runtime of benchmark too small")
            if self._debug:
                print(f"O3: {self._runtime_o3} - O0: {self._runtime_initial}")
            self._runtime_prev = self._runtime_initial
            return
        raise Exception(f"Failed to reset after {attempts} attempts")

    def step(self, flags):
        if flags[0] == "noop":
            if self._debug:
                print(
                    f"reward: {0} - runtime: {self._runtime_prev} - runtime_prev: {self._runtime_prev} - runtime_initial: {self._runtime_initial}"
                )
            return 0
        if len(flags) > 1:
            self._cg_env.multistep(
                [self._cg_env.action_space.flags.index(f) for f in flags]
            )
        else:
            self._cg_env.step(self._cg_env.action_space.flags.index(flags[0]))
        runtime = self._compile_and_get_runtime()
        reward = (self._runtime_prev - runtime) / max(
            max(
                self._runtime_initial - self._runtime_baseline,
                0.01 * self._runtime_initial,
            ),
            1e-12,
        )
        if self._debug:
            print(
                f"reward: {reward} - runtime: {runtime} - runtime_prev: {self._runtime_prev} - runtime_initial: {self._runtime_initial}"
            )
        reward = np.clip(reward, -3, 3)
        self._runtime_prev = runtime
        return reward

    def get_observation(self, obs_name):
        return get_observation_from_cg(self._cg_env, obs_name)

    def _compile_and_get_runtime(self) -> float:
        compile_lm_safely(
            ir=self._cg_env.observation["Ir"],
            sequence=[],
            result_path=self._bin_filepath,
            linkopts=[],
        )
        runtime = _measure_runtime(
            self._bin_filepath,
            execution_args="0",
        )
        return runtime

    def _compile_and_get_runtime_seq(self, sequence) -> float:
        compile_lm_safely(
            ir=self._cg_env.observation["Ir"],
            sequence=sequence,
            result_path=self._bin_filepath,
            linkopts=[],
        )
        runtime = _measure_runtime(
            self._bin_filepath,
            execution_args="0",
        )
        return runtime


def _measure_runtime(
    bin_path,
    execution_args: str = "",
    warmup=0,
) -> float:
    filename = f"_runtime_env_hyperfine_result.json"
    command = f"hyperfine --warmup {warmup} '{bin_path} {execution_args}' --export-json {filename} --show-output"
    proc = subprocess.run(
        command,
        shell=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"run failed: {proc.stderr}")
    with open(filename, "r") as hyperfine_result:
        result = json.load(hyperfine_result)
    return result["results"][0]["mean"]


def _compile_and_get_instructions(cg_env, bin_filepath) -> int:
    attempts = 10
    for i in range(attempts):
        try:
            return compile_and_get_instructions_no_sequence(
                ir=cg_env.observation["Ir"],
                result_path=bin_filepath,
                execution_args="0",
                linkopts=[],
            )
        except Exception as e:
            print(f"compile_and_get_instructions_no_sequence failed: {e}")
    raise Exception(
        f"compile_and_get_instructions_no_sequence failed after {attempts} attempts"
    )


def _compile_and_get_instructions_seq(cg_env, bin_filepath, sequence) -> int:
    return compile_and_get_instructions(
        ir=cg_env.observation["Ir"],
        sequence=sequence,
        result_path=bin_filepath,
        execution_args="0",
        linkopts=[],
    )


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