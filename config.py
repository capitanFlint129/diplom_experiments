import os
from dataclasses import dataclass, field

from action_config import *

WANDB_PROJECT_NAME = "rl-compilers-experiments-DQN-fix-results-2"
COMPILER_GYM_PATH = "~/.local/share/compiler_gym"
LLVM_BINS_PATH = os.path.join(COMPILER_GYM_PATH, "llvm-v0/bin")
MODELS_DIR = "_models"


@dataclass
class TrainConfig:
    # Algorithm section
    algorithm: str = "LstmDQN"
    gamma: float = 0.9
    epsilon: float = 1.0  # The starting value for epsilon
    epsilon_end: float = 0.05  # The ending value for epsilon
    epsilon_dec: float = 5e-5  # The decrement value for epsilon
    fc_dim: int = 256  # The dimension of a fully connected layer
    lstm_hidden_size: int = 256  # The dimension of a fully connected layer
    # Learning
    lr: float = 5e-5  # The learning rate
    tau: float = 0.99  # soft update coefficient
    batch_size: int = 512  # The batch size
    max_mem_size: int = 100000  # The maximum memory size
    episodes: int = 4000  # The number of episodes used to learn
    validation_interval: int = 500  # The number of episodes used to learn
    episode_length: int = 25  # The (MAX) number of transformation passes per episode
    patience: int = 5  # The (MAX) number of times to apply a series of transformations without observable change
    val_patience: int = 5
    eval_with_forbidden_actions: bool = True
    learn_memory_threshold: int = max(batch_size, 32)
    enable_soft_update: bool = True
    replace_period: int = 500

    # General section
    datasets: list = field(
        default_factory=lambda: [
            "benchmark://cbench-v1",
            # ("benchmark://anghabench-v1", 2000),
            # ("/home/flint/diplom/datasets/bc/angha_kernels_largest_10k/", 2000),
            # "benchmark://mibench-v1",
            # "benchmark://opencv-v0",
        ]
    )
    train_val_test_split: bool = True
    skipped_benchmarks: list = field(default_factory=lambda: [])
    compiler_gym_env: str = "llvm-v0"
    observation_space: list = field(
        default_factory=lambda: [
            # "IR2Vec",
            "InstCountNorm",
            # "Autophase",
        ]
    )
    observation_modifiers: list = field(
        default_factory=lambda: [
            # "start-IR2Vec",
            # "remains-counter",
            "remains-counter-normalized",
            # "prev-2",
        ]
    )
    observation_size: int = 70
    reward_space: str = "IrInstructionCountOz"
    actions: list = field(
        default_factory=lambda: COMPILER_GYM_LEADERBOARD_DQN_ACTION_SET
    )
    # Experiment section (logging and reproduce)
    logging_history_size: int = 100
    random_state: int = 99
    codesize_bins_number: int = 23
