import os
import sys

from compiler_gym import CompilerEnv

from config.config import TrainConfig
from env.my_env import MyEnv
from env.performance_optimization.cfg_grind import (
    compile_and_get_instructions,
    compile_and_get_instructions_no_sequence,
)
from utils import get_observation_from_cg

O0 = "-O0"
O1 = "-O1"
O2 = "-O2"
O3 = "-O3"
OZ = "-Oz"


class CfgGridSubsetEnv(MyEnv):
    def __init__(
        self,
        config: TrainConfig,
        env: CompilerEnv,
        debug=False,
    ):
        assert (
            len(config.actions_sequence) == config.episode_length
        ), f"Actions: {len(config.actions_sequence)}, episode length: {config.episode_length}"
        self._prepare_actions = config.prepare_actions.split()
        self._cg_env = env
        self._tmp_dir = f"_cfg_grind_env_tmp"
        os.makedirs(self._tmp_dir, exist_ok=True)
        self._filename = "tmpfile"
        self._bin_filepath = os.path.join(self._tmp_dir, self._filename)
        self._config = config
        self._executed_insts_o3 = None
        self._executed_insts_initial = None
        self._executed_insts_prev = None
        self._executed_insts_baseline = None
        self._debug = debug
        self._actions_sequence = config.actions_sequence
        self._cur_index = 0
        # self._initial_ir = str(self._cg_env.observation["Ir"])
        self._cur_seq = []

    def is_runtime(self) -> bool:
        return False

    def get_cur_ir(self) -> CompilerEnv:
        return self._cg_env.observation["Ir"]

    def reset(self, benchmark=None, val=False):
        self._cur_seq = []
        self._cur_index = 0
        attempts = 100
        self._opts = []
        for i in range(attempts):
            try:
                self._cg_env.reset(benchmark=benchmark)

                # O3
                if self._debug:
                    self._executed_insts_o3 = self._compile_and_get_instructions_seq(
                        sequence=[O3]
                    )

                self._executed_insts_baseline = self._compile_and_get_instructions_seq(
                    sequence=[O3]
                )
                # Initial
                self._executed_insts_initial = self._compile_and_get_instructions_seq(
                    self._cur_seq
                )

                self._cg_env.multistep(
                    [
                        self._cg_env.action_space.flags.index(f)
                        for f in self._prepare_actions
                    ]
                )

                if val:
                    self._executed_insts_prev = self._executed_insts_initial
                else:
                    self._executed_insts_prev = self._compile_and_get_instructions_seq(
                        self._cur_seq
                    )
                # if self._debug:
                print(
                    f"O3: {self._executed_insts_baseline} - O0: {self._executed_insts_initial} - diff {self._executed_insts_initial - self._executed_insts_baseline}"
                )
                prepare_improve = (
                    self._executed_insts_initial - self._executed_insts_prev
                ) / max(
                    self._executed_insts_initial - self._executed_insts_baseline,
                    int(0.01 * self._executed_insts_initial),
                )
                print(f"prepare actions improve: {prepare_improve}")
                return True
            except ValueError as e:
                print(e, file=sys.stderr)
            except Exception as e:
                print(e, file=sys.stderr)
        raise Exception(f"Failed to reset after {attempts} attempts")

    def step(self, action: list[str]):
        action = action[0]
        if action == "apply":
            self._cg_env.multistep(
                [
                    self._cg_env.action_space.flags.index(f)
                    for f in self._actions_sequence[self._cur_index].split()
                ]
            )
            self._cur_index += 1
        elif action == "skip":
            self._cur_index += 1
            if self._debug:
                print(
                    f"reward: {0} - executed_insts: {self._executed_insts_prev} - executed_insts_prev: {self._executed_insts_prev} - executed_insts_initial: {self._executed_insts_initial}"
                )
            return 0
        else:
            raise Exception("O3 subset env: unknown action")
        executed_insts = self._compile_and_get_instructions()
        reward = (self._executed_insts_prev - executed_insts) / (
            max(
                self._executed_insts_initial - self._executed_insts_baseline,
                int(0.01 * self._executed_insts_initial),
            )
        )
        if self._debug:
            print(
                f"reward: {reward} - executed_insts: {executed_insts} - executed_insts_prev: {self._executed_insts_prev} - executed_insts_initial: {self._executed_insts_initial}"
            )
        self._executed_insts_prev = executed_insts
        return reward

    def get_observation(self, obs_name):
        return get_observation_from_cg(self._cg_env, obs_name)

    def _compile_and_get_instructions_seq(self, sequence) -> int:
        return compile_and_get_instructions(
            ir=self._cg_env.observation["Ir"],
            sequence=sequence,
            result_path=self._bin_filepath,
            execution_args="0",
            linkopts=[],
        )

    def _compile_and_get_instructions(self) -> int:
        attempts = 10
        for i in range(attempts):
            try:
                return compile_and_get_instructions_no_sequence(
                    ir=self._cg_env.observation["Ir"],
                    result_path=self._bin_filepath,
                    execution_args="0",
                    linkopts=[],
                )
            except Exception as e:
                print(f"compile_and_get_instructions_no_sequence failed: {e}")
        raise Exception(
            f"compile_and_get_instructions_no_sequence failed after {attempts} attempts"
        )


class CfgGridEnv(MyEnv):
    def __init__(self, config: TrainConfig, env: CompilerEnv, debug=False):
        self._cg_env = env
        self._tmp_dir = f"_cfg_grind_env_tmp"
        os.makedirs(self._tmp_dir, exist_ok=True)
        self._filename = "tmpfile"
        self._bin_filepath = os.path.join(self._tmp_dir, self._filename)
        self._config = config
        self._executed_insts_o3 = None
        self._executed_insts_initial = None
        self._executed_insts_prev = None
        self._executed_insts_baseline = None
        self._executed_insts_o2 = None
        self._debug = debug

    def is_runtime(self) -> bool:
        return False

    def get_cur_ir(self) -> CompilerEnv:
        return self._cg_env.observation["Ir"]

    def reset(self, benchmark=None, val=False):
        attempts = 100
        self._opts = []
        for i in range(attempts):
            try:
                self._cg_env.reset(benchmark=benchmark)

                # O3
                if self._debug:
                    self._executed_insts_o3 = self._compile_and_get_instructions_seq(
                        sequence=[O3]
                    )

                self._executed_insts_baseline = self._compile_and_get_instructions_seq(
                    sequence=[O3]
                )
                self._executed_insts_o2 = (
                    self._executed_insts_baseline
                ) = self._compile_and_get_instructions_seq(sequence=[O2])

                # Initial
                self._executed_insts_initial = self._compile_and_get_instructions()
                if (
                    self._executed_insts_initial - self._executed_insts_baseline
                    < self._config.jotai_improve_threshold_insts
                ):
                    return False
                if self._debug:
                    print(
                        f"O3: {self._executed_insts_o3} - O0: {self._executed_insts_initial}"
                    )
                self._executed_insts_prev = self._executed_insts_initial
                return True
            except ValueError as e:
                print(e, file=sys.stderr)
            except Exception as e:
                print(e, file=sys.stderr)
        raise Exception(f"Failed to reset after {attempts} attempts")

    def step(self, flags):
        if flags[0] == "noop":
            if self._debug:
                print(
                    f"reward: {0} - executed_insts: {self._executed_insts_prev} - executed_insts_prev: {self._executed_insts_prev} - executed_insts_initial: {self._executed_insts_initial}"
                )
            return 0
        if len(flags) > 1:
            self._cg_env.multistep(
                [self._cg_env.action_space.flags.index(f) for f in flags]
            )
        else:
            self._cg_env.step(self._cg_env.action_space.flags.index(flags[0]))
        executed_insts = self._compile_and_get_instructions()
        reward = (self._executed_insts_prev - executed_insts) / (
            max(
                self._executed_insts_initial - self._executed_insts_baseline,
                int(0.01 * self._executed_insts_initial),
            )
        )
        if self._debug:
            print(
                f"reward: {reward} - executed_insts: {executed_insts} - executed_insts_prev: {self._executed_insts_prev} - executed_insts_initial: {self._executed_insts_initial}"
            )
        self._executed_insts_prev = executed_insts
        return reward

    def step_ignore_reward(self, flags):
        if flags[0] == "noop":
            if self._debug:
                print(
                    f"reward: {0} - executed_insts: {self._executed_insts_prev} - executed_insts_prev: {self._executed_insts_prev} - executed_insts_initial: {self._executed_insts_initial}"
                )
            return 0
        if len(flags) > 1:
            self._cg_env.multistep(
                [self._cg_env.action_space.flags.index(f) for f in flags]
            )
        else:
            self._cg_env.step(self._cg_env.action_space.flags.index(flags[0]))
        return 0
    
    def get_final_reward(self) -> float:
        model_result = self._compile_and_get_instructions()
        return (self._executed_insts_initial - model_result) / (
            max(
                self._executed_insts_initial - self._executed_insts_baseline,
                1,
            )
        )

    def get_observation(self, obs_name):
        return get_observation_from_cg(self._cg_env, obs_name)

    def gather_data(self, without_train=False) -> tuple[float, float, float, float]:
        model_result = self._executed_insts_prev
        if without_train:
            model_result = self._compile_and_get_instructions()
        return (
            self._executed_insts_initial,
            self._executed_insts_o2,
            self._executed_insts_baseline,
            model_result,
        )

    def _compile_and_get_instructions(self) -> int:
        attempts = 10
        for i in range(attempts):
            try:
                return compile_and_get_instructions_no_sequence(
                    ir=self._cg_env.observation["Ir"],
                    result_path=self._bin_filepath,
                    execution_args="0",
                    linkopts=[],
                )
            except Exception as e:
                print(f"compile_and_get_instructions_no_sequence failed: {e}")
        raise Exception(
            f"compile_and_get_instructions_no_sequence failed after {attempts} attempts"
        )

    def _compile_and_get_instructions_seq(self, sequence) -> int:
        return compile_and_get_instructions(
            ir=self._cg_env.observation["Ir"],
            sequence=sequence,
            result_path=self._bin_filepath,
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
