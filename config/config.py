import os
from dataclasses import dataclass, field

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
            "mca",
            "remains-counter-normalized",
            # "prev-2",
        ]
    )
    # reward_space: str = "Runtime"
    reward_space: str = "CfgInstructions"
    # reward_space: str = "MCA"
    observation_size: int = None
    actions: list = field(default_factory=lambda: O23_SUBSEQ_CBENCH_MINS)
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
