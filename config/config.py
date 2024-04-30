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


@dataclass_json
@dataclass
class TrainConfig:
    # Algorithm section
    algorithm: str = "LstmDQN"
    # reward_space: str = "CfgInstructions"
    reward_space: str = "MCA"
    enable_dueling_dqn: bool = False
    gamma: float = 0.9
    epsilon: float = 1.0  # The starting value for epsilon
    epsilon_end: float = 0.05  # The ending value for epsilon
    epsilon_dec: float = 3e-5  # The decrement value for epsilon
    fc_dim: int = 128  # The dimension of a fully connected layer
    lstm_hidden_size: int = 256  # The dimension of a fully connected layer
    # Learning
    lr: float = 1e-4  # The learning rate
    tau: float = 0.99  # soft update coefficient
    batch_size: int = 256  # The batch size
    max_mem_size: int = 100000  # The maximum memory size
    prefill: int = 0
    episodes: int = 30000  # The number of episodes used to learn
    validation_interval: int = 500  # The number of episodes used to learn
    episode_length: int = 20  # The (MAX) number of transformation passes per episode
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
    observation_space: str = "IR2Vec"
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
            "remains-counter-normalized",
            # "prev-2",
        ]
    )
    observation_size: int = 301
    # reward_space: str = "IrInstructionCountOz"
    # reward_space: str = "RuntimePointEstimateReward"
    # reward_space: str = "LlvmMca"
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
        if self.special_actions[0] != self.actions[0]:
            self.actions = self.special_actions + self.actions

    def save(self, run_name: str, replace=False):
        if not os.path.exists(MODELS_DIR_PROJECT):
            os.makedirs(MODELS_DIR_PROJECT)
        file = os.path.join(MODELS_DIR_PROJECT, f"{run_name}_config.json")
        if replace or not os.path.isfile(file):
            with open(file, "w") as ouf:
                ouf.write(self.to_json())
