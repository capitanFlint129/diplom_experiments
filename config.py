import os
from dataclasses import dataclass

from action_config import COMPILER_GYM_LEADERBOARD_DQN_ACTION_SET

COMPILER_GYM_PATH = "~/.local/share/compiler_gym"
LLVM_BINS_PATH = os.path.join(COMPILER_GYM_PATH, "llvm-v0/bin")
MODELS_DIR = "_models"


@dataclass
class TrainConfig:
    # Algorithm section
    algorithm = "DoubleDQN"
    gamma = 0.99
    epsilon = 1.0  # The starting value for epsilon
    epsilon_end = 0.05  # The ending value for epsilon
    epsilon_dec = 5e-5  # The decrement value for epsilon
    fc_dim = 128  # The dimension of a fully connected layer
    lstm_hidden_size = 512  # The dimension of a fully connected layer
    action_embedding_size = 100
    # Learning
    lr = 1e-4  # The learning rate
    tau = 0.99  # soft update coefficient
    batch_size = 1024  # The batch size
    max_mem_size = 100000  # The maximum memory size
    replace = 500  # The number of iterations to run before replacing target network
    episodes = 4000  # The number of episodes used to learn
    validation_interval = 500  # The number of episodes used to learn
    episode_length = 50  # The (MAX) number of transformation passes per episode
    patience = 10  # The (MAX) number of times to apply a series of transformations without observable change
    val_patience = 5
    eval_with_forbidden_actions = True
    learn_memory_threshold = max(
        batch_size, 32
    )  # The number of fully exploratory episodes to run before starting learning

    # General section
    datasets = [
        # ("benchmark://anghabench-v1", 2000),
        ("/home/flint/diplom/datasets/bc/angha_kernels_largest_10k/", 2000),
        # "benchmark://cbench-v1",
        # "benchmark://mibench-v1",
        # "benchmark://opencv-v0",
    ]
    train_val_test_split = False
    # некоторые программы очень большие и вычисление observation на них может занимать много
    # времени и памяти, поэтому для начальных экспериментов удобно пропускать некоторые бенчмарки
    skipped_benchmarks = [
        # "benchmark://cbench-v1/bzip2",
        # "benchmark://cbench-v1/ghostscript",
        # "benchmark://cbench-v1/tiff2rgba",
        # "benchmark://cbench-v1/tiff2bw",
        # "benchmark://cbench-v1/tiffdither",
        # "benchmark://cbench-v1/tiffmedian",
        # "benchmark://cbench-v1/lame",
        # "benchmark://cbench-v1/jpeg-c",
        # "benchmark://cbench-v1/jpeg-d",
    ]
    compiler_gym_env = "llvm-v0"
    observation_space = [
        # "IR2VecNormalized",
        # "InstCountNorm",
        "AutophaseNorm",
    ]
    observation_modifiers = [
        "remains-counter",
        "prev-2",
    ]
    observation_size = 113
    reward_space = "IrInstructionCountOz"
    actions = COMPILER_GYM_LEADERBOARD_DQN_ACTION_SET
    # Experiment section (logging and reproduce)
    logging_history_size = 100
    random_state = 42
    codesize_bins_number = 23
