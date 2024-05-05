import os
from dataclasses import dataclass, field
from typing import Union

from dataclasses_json import dataclass_json

from config.action_config import *

WANDB_PROJECT_NAME = "trash2"
COMPILER_GYM_PATH = "~/.local/share/compiler_gym"
LLVM_BINS_PATH = os.path.join(COMPILER_GYM_PATH, "llvm-v0/bin")
MODELS_DIR = "_models"
TEST_BENCHMARKS_DIR = "data/test_benchmarks"
MODELS_DIR_PROJECT = os.path.join(MODELS_DIR, WANDB_PROJECT_NAME)
RAW_IR_OBSERVATION_NAME = "Ir"
LLVM_BINS = "/home/flint/.local/share/compiler_gym/llvm-v0/bin"
MCA_BIN = os.path.join(LLVM_BINS, "llvm-mca")
# # CLANG_BIN = os.path.join(LLVM_BINS, 'clang')
# CLANG_BIN = "clang"
LLC_BIN = os.path.join(LLVM_BINS, "llc")
# JOTAI_BENCHMARKS_DIRS = [
#     "/home/flint/diplom/jotai-benchmarks/benchmarks/anghaLeaves/",
#     "/home/flint/diplom/jotai-benchmarks/benchmarks/anghaMath/",
# ]
# TMP_DIR = "tmp"
LLVM_AS_BIN = os.path.join(LLVM_BINS, "llvm-as")
OPT_BIN = os.path.join(LLVM_BINS, "opt")
LLVM_TEST_SUITE_DATASET_PATH = "_my_data/llvm_test_suite"


@dataclass_json
@dataclass
class TrainConfig:
    # Algorithm section
    algorithm: str = "LstmDQN"
    task: str = "classic_phase_ordering"
    # task: str = "subset"
    actions_sequence: list[str] = field(
        default_factory=lambda: []
    )  # for subset task only
    prepare_actions: list[str] = field(default_factory=lambda: [])
    enable_dueling_dqn: bool = False
    gamma: float = 0.9
    epsilon: float = 1.0  # The starting value for epsilon
    epsilon_end: float = 0.05  # The ending value for epsilon
    epsilon_dec: float = 3e-5  # The decrement value for epsilon
    fc_dim: int = 256  # The dimension of a fully connected layer
    lstm_hidden_size: int = 512  # The dimension of a fully connected layer
    # Learning
    lr: float = 1e-4  # The learning rate
    tau: float = 0.99  # soft update coefficient
    batch_size: int = 256  # The batch size
    max_mem_size: int = 100000  # The maximum memory size
    prefill: int = 0
    episodes: int = 30000  # The number of episodes used to learn
    validation_interval: int = 500  # The number of episodes used to learn
    episode_length: int = 15  # The (MAX) number of transformation passes per episode
    patience: int = 5  # The (MAX) number of times to apply a series of transformations without observable change
    val_patience: int = 5
    eval_with_forbidden_actions: bool = True
    eval_with_bestsequence: bool = False
    learn_memory_threshold: int = max(batch_size, 32)
    enable_soft_update: bool = False
    replace_period: int = 500

    # General section
    dataset: str = "llvm_test_suite_benchmarks"
    # dataset: str = "benchmark://jotaibench-v0"
    # datasets: list = field(
    #     default_factory=lambda: [
    #         # "benchmark://cbench-v1",
    #         ("benchmark://anghabench-v1", 5000),
    #         "benchmark://mibench-v1",
    #         # ("/home/flint/diplom/datasets/bc/angha_kernels_largest_10k/", 2000),
    #         # "benchmark://opencv-v0",
    #     ]
    # )
    # test_datasets: list = field(
    #     default_factory=lambda: [
    #         "benchmark://cbench-v1",
    #         # ("benchmark://anghabench-v1", 2000),
    #         # ("/home/flint/diplom/datasets/bc/angha_kernels_largest_10k/", 2000),
    #         # "benchmark://mibench-v1",
    #         # "benchmark://opencv-v0",
    #     ]
    # )
    # train_val_test_split: bool = False
    # skipped_benchmarks: list = field(default_factory=lambda: [])
    compiler_gym_env: str = "llvm-v0"
    observation_space: str = "IR2Vec+InstCountNorm+AutophaseNorm"
    # observation_space: list = field(
    #     default_factory=lambda: [
    #         # "IR2Vec",
    #         "InstCountNorm",
    #         # "Autophase",
    #     ]
    # )
    observation_modifiers: list = field(
        default_factory=lambda: [
            # "start-IR2Vec",
            # "remains-counter",
            # "mca",
            "remains-counter-normalized",
            # "prev-2",
        ]
    )
    val_size: Union[int, float] = 0.2
    reward_space: str = "Runtime"
    # reward_space: str = "CfgInstructions"
    # reward_space: str = "MCA"
    observation_size: int = None
    actions: list = field(default_factory=lambda: POSET_RL_ODG_O3)
    reward_scale: float = 1
    special_actions: list = field(
        default_factory=lambda: [
            "noop",
        ]
    )
    # Experiment section (logging and reproduce)
    logging_history_size: int = 100
    random_state: int = 99
    codesize_bins_number: int = 23

    def __post_init__(self):
        if self.task == "subset":
            self.episode_length = len(self.actions_sequence)

        if len(self.special_actions) > 0 and self.special_actions[0] != self.actions[0]:
            self.actions = self.special_actions + self.actions

        if self.observation_space == "IR2Vec+InstCountNorm+AutophaseNorm":
            self.observation_size = 425
        elif self.observation_space.startswith("IR2Vec"):
            self.observation_size = 300
        elif self.observation_space == "InstCountNorm+AutophaseNorm":
            self.observation_size = 125
        elif self.observation_space.startswith("InstCount"):
            self.observation_size = 69
        elif self.observation_space.startswith("Autophase"):
            self.observation_size = 56

        if "mca" in self.observation_modifiers:
            self.observation_size += 3
        if "remains-counter" in self.observation_modifiers:
            self.observation_size += 1
        if "remains-counter-normalized" in self.observation_modifiers:
            self.observation_size += 1

    def save(self, run_name: str, replace=False):
        if not os.path.exists(MODELS_DIR_PROJECT):
            os.makedirs(MODELS_DIR_PROJECT)
        file = os.path.join(MODELS_DIR_PROJECT, f"{run_name}_config.json")
        if replace or not os.path.isfile(file):
            with open(file, "w") as ouf:
                ouf.write(self.to_json())

    @staticmethod
    def load_config(run_name: str) -> "TrainConfig":
        with open(
            os.path.join(MODELS_DIR_PROJECT, f"{run_name}_config.json"), "r"
        ) as inf:
            config = TrainConfig.from_json(inf.read())
            return config

    @staticmethod
    def load_config_from_path(path: str) -> "TrainConfig":
        with open(path, "r") as inf:
            config = TrainConfig.from_json(inf.read())
            return config


RUNS_NUMBER = {
    "stepanov_container": 1,
    "Shootout-C++-lists": 1,
    "Shootout-lists": 1,
    "lowercase": 1,
    "sphereflake": 1,
    "loop_unroll": 1,
    "smallpt": 1,
    "simple_types_constant_folding": 1,
    "huffbench": 1,
    "spirit": 1,
    "symm": 1,
    "stepanov_vector": 1,
    "stepanov_abstraction": 1,
    "stepanov_v1p2": 1,
    "almabench": 2,
    "flops": 2,
    "salsa20": 2,
    "syr2k": 2,
    "Shootout-C++-methcall": 2,
    "Shootout-C++-ary3": 2,
    "syrk": 2,
    "ray": 2,
    "whetstone": 3,
    "flops-3": 3,
    "Shootout-ary3": 3,
    "flops-6": 3,
    "flops-5": 3,
    "spectral-norm": 3,
    "flops-8": 3,
    "chomp": 3,
    "mandel-text": 3,
    "Shootout-C++-ackermann": 3,
    "adi": 3,
    "fp-convert": 3,
    "Shootout-C++-matrix": 3,
    "Shootout-objinst": 3,
    "correlation": 3,
    "Shootout-C++-fibo": 3,
    "queens": 3,
    "Shootout-fib2": 3,
    "dry": 3,
    "fftbench": 3,
    "Shootout-matrix": 3,
    "floyd-warshall": 3,
    "Shootout-C++-random": 3,
    "Shootout-C++-heapsort": 3,
    "fannkuch": 3,
    "fldry": 3,
    "Shootout-heapsort": 3,
    "trmm": 3,
    "Shootout-C++-objinst": 3,
    "fdtd-apml": 3,
    "covariance": 3,
    "linpack-pc": 3,
    "Shootout-random": 3,
    "oourafft": 3,
    "Shootout-hash": 3,
    "Shootout-methcall": 3,
    "gramschmidt": 3,
    "lpbench": 3,
    "Shootout-C++-hash2": 3,
    "ReedSolomon": 3,
    "Shootout-C++-sieve": 3,
    "functionobjects": 3,
    "Shootout-sieve": 3,
    "perlin": 3,
    "simple_types_loop_invariant": 3,
    "cholesky": 5,
    "ffbench": 5,
    "flops-4": 5,
    "mandel": 5,
    "matmul_f64_4x4": 5,
    "seidel-2d": 5,
    "flops-7": 5,
    "nsieve-bits": 5,
    "fdtd-2d": 5,
    "richards_benchmark": 5,
    "bigfib": 5,
    "recursive": 5,
    "pi": 5,
    "fasta": 5,
    "doitgen": 5,
    "fbench": 5,
    "flops-1": 5,
    "mandel-2": 5,
    "himenobmtxpa": 5,
    "mvt": 5,
    "dt": 5,
    "gesummv": 5,
    "revertBits": 5,
    "Shootout-C++-except": 5,
    "Shootout-C++-moments": 5,
    "dynprog": 5,
    "Shootout-C++-lists1": 5,
    "durbin": 5,
    "oopack_v1p8": 5,
    "evalloop": 5,
    "gemver": 5,
    "puzzle": 5,
    "n-body": 5,
    "FloatMM": 5,
    "flops-2": 5,
    "Shootout-C++-hash": 5,
}
