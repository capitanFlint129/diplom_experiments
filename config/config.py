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
    # RESUME set epsilon
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
    episode_length: int = 10  # The (MAX) number of transformation passes per episode
    patience: int = 5  # The (MAX) number of times to apply a series of transformations without observable change
    val_patience: int = 5
    eval_with_forbidden_actions: bool = True
    eval_with_bestsequence: bool = False
    learn_memory_threshold: int = max(batch_size, 32)
    enable_soft_update: bool = False
    replace_period: int = 500

    # General section
    dataset: str = "benchmark://jotaibench-v0"
    # dataset: str = "llvm_test_suite_benchmarks"
    jotai_improve_threshold_insts: int = 500
    llvm_test_suite_runtime_validation: bool = False
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
    skipped_benchmarks: list = field(
        default_factory=lambda: [
            "stepanov_vector",
            "stepanov_abstraction",
            "stepanov_v1p2",
        ]
    )
    compiler_gym_env: str = "llvm-v0"
    observation_space: str = "AutophaseNorm"
    # observation_space: str = "IR2Vec+InstCountNorm+AutophaseNorm"
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
    val_size: Union[int, float] = 0.01
    # reward_space: str = "Runtime"
    reward_space: str = "CfgInstructions"
    # reward_space: str = "MCA"
    observation_size: int = None
    actions: list = field(default_factory=lambda: O23_SUBSEQ_POSET_LIKE)
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
    "almabench": 1,  # 2,
    "flops": 1,  # 2,
    "salsa20": 1,  # 2,
    "syr2k": 1,  # 2,
    "Shootout-C++-methcall": 1,  # 2,
    "Shootout-C++-ary3": 1,  # 2,
    "syrk": 1,  # 2,
    "ray": 1,  # 2,
    "whetstone": 1,  # 3,
    "flops-3": 1,  # 3,
    "Shootout-ary3": 1,  # 3,
    "flops-6": 1,  # 3,
    "flops-5": 1,  # 3,
    "spectral-norm": 1,  # 3,
    "flops-8": 1,  # 3,
    "chomp": 1,  # 3,
    "mandel-text": 1,  # 3,
    "Shootout-C++-ackermann": 1,  # 3,
    "adi": 1,  # 3,
    "fp-convert": 1,  # 3,
    "Shootout-C++-matrix": 1,  # 3,
    "Shootout-objinst": 1,  # 3,
    "correlation": 1,  # 3,
    "Shootout-C++-fibo": 1,  # 3,
    "queens": 1,  # 3,
    "Shootout-fib2": 1,  # 3,
    "dry": 1,  # 3,
    "fftbench": 1,  # 3,
    "Shootout-matrix": 1,  # 3,
    "floyd-warshall": 1,  # 3,
    "Shootout-C++-random": 1,  # 3,
    "Shootout-C++-heapsort": 1,  # 3,
    "fannkuch": 1,  # 3,
    "fldry": 1,  # 3,
    "Shootout-heapsort": 1,  # 3,
    "trmm": 1,  # 3,
    "Shootout-C++-objinst": 1,  # 3,
    "fdtd-apml": 1,  # 3,
    "covariance": 1,  # 3,
    "linpack-pc": 1,  # 3,
    "Shootout-random": 1,  # 3,
    "oourafft": 1,  # 3,
    "Shootout-hash": 1,  # 3,
    "Shootout-methcall": 1,  # 3,
    "gramschmidt": 1,  # 3,
    "lpbench": 1,  # 3,
    "Shootout-C++-hash2": 1,  # 3,
    "ReedSolomon": 1,  # 3,
    "Shootout-C++-sieve": 1,  # 3,
    "functionobjects": 1,  # 3,
    "Shootout-sieve": 1,  # 3,
    "perlin": 1,  # 3,
    "simple_types_loop_invariant": 1,  # 3,
    "cholesky": 3,  # 5,
    "ffbench": 3,  # 5,
    "flops-4": 3,  # 5,
    "mandel": 3,  # 5,
    "matmul_f64_4x4": 3,  # 5,
    "seidel-2d": 3,  # 5,
    "flops-7": 3,  # 5,
    "nsieve-bits": 3,  # 5,
    "fdtd-2d": 3,  # 5,
    "richards_benchmark": 3,  # 5,
    "bigfib": 3,  # 5,
    "recursive": 3,  # 5,
    "pi": 3,  # 5,
    "fasta": 3,  # 5,
    "doitgen": 3,  # 5,
    "fbench": 3,  # 5,
    "flops-1": 3,  # 5,
    "mandel-2": 3,  # 5,
    "himenobmtxpa": 3,  # 5,
    "mvt": 3,  # 5,
    "dt": 3,  # 5,
    "gesummv": 3,  # 5,
    "revertBits": 3,  # 5,
    "Shootout-C++-except": 3,  # 5,
    "Shootout-C++-moments": 3,  # 5,
    "dynprog": 3,  # 5,
    "Shootout-C++-lists1": 3,  # 5,
    "durbin": 3,  # 5,
    "oopack_v1p8": 3,  # 5,
    "evalloop": 3,  # 5,
    "gemver": 3,  # 5,
    "puzzle": 3,  # 5,
    "n-body": 3,  # 5,
    "FloatMM": 3,  # 5,
    "flops-2": 3,  # 5,
    "Shootout-C++-hash": 3,  # 5,
}


O3_RUNTIMES = {
    "Shootout-objinst": 0.0004,
    "Shootout-nestedloop": 0.00041,
    "lowercase": 0.00042,
    "RealMM": 0.00092,
    "Shootout-C++-objinst": 0.0011,
    "Oscar": 0.00111,
    "Shootout-C++-nestedloop": 0.00127,
    "lu": 0.00141,
    "jacobi-1d-imper": 0.0023,
    "Shootout-ackermann": 0.004,
    "Queens": 0.00787,
    "Towers": 0.00846,
    "reg_detect": 0.01365,
    "Perm": 0.0164,
    "Bubblesort": 0.02184,
    "Quicksort": 0.03386,
    "Shootout-C++-ary2": 0.05234,
    "Shootout-C++-ary": 0.05303,
    "Treesort": 0.06163,
    "Shootout-C++-strcat": 0.06246,
    "Puzzle": 0.06847,
    "puzzle": 0.08843,
    "Shootout-C++-moments": 0.08854,
    "FloatMM": 0.10018,
    "trisolv": 0.10334,
    "misr": 0.10946,
    "Shootout-C++-except": 0.11746,
    "partialsums": 0.12887,
    "oopack_v1p8": 0.13302,
    "atax": 0.14053,
    "Shootout-C++-lists1": 0.14824,
    "Shootout-strcat": 0.1488,
    "jacobi-2d-imper": 0.16336,
    "dt": 0.17446,
    "bicg": 0.17718,
    "revertBits": 0.18049,
    "spectral-norm": 0.21382,
    "gesummv": 0.23964,
    "bigfib": 0.2573,
    "dynprog": 0.30136,
    "dry": 0.32441,
    "fldry": 0.32548,
    "mvt": 0.34927,
    "Shootout-C++-hash": 0.38128,
    "evalloop": 0.40567,
    "mandel": 0.4494,
    "gemver": 0.45621,
    "pi": 0.48116,
    "n-body": 0.48501,
    "seidel-2d": 0.51239,
    "cholesky": 0.51475,
    "flops-2": 0.52092,
    "ffbench": 0.52751,
    "matmul_f64_4x4": 0.53654,
    "flops-4": 0.5465,
    "durbin": 0.54936,
    "richards_benchmark": 0.56599,
    "himenobmtxpa": 0.57402,
    "recursive": 0.58484,
    "flops-7": 0.62282,
    "fdtd-2d": 0.65348,
    "nsieve-bits": 0.68212,
    "flops-1": 0.69906,
    "fasta": 0.70129,
    "loop_unroll": 0.70718,
    "mandel-2": 0.70863,
    "simple_types_constant_folding": 0.70926,
    "whetstone": 0.71391,
    "doitgen": 0.78614,
    "flops-6": 0.8192,
    "fbench": 0.82923,
    "fftbench": 0.84199,
    "flops-5": 0.84465,
    "Shootout-C++-ackermann": 0.86268,
    "flops-8": 1.0934,
    "Shootout-matrix": 1.13,
    "Shootout-C++-ary3": 1.14881,
    "Shootout-ary3": 1.16011,
    "flops-3": 1.17701,
    "Shootout-C++-matrix": 1.1992,
    "linpack-pc": 1.29683,
    "perlin": 1.37033,
    "chomp": 1.38913,
    "mandel-text": 1.39265,
    "Shootout-C++-sieve": 1.44016,
    "Shootout-C++-fibo": 1.49169,
    "fp-convert": 1.50005,
    "Shootout-fib2": 1.51509,
    "trmm": 1.57902,
    "adi": 1.65915,
    "simple_types_loop_invariant": 1.71299,
    "Shootout-C++-random": 1.85354,
    "queens": 1.87183,
    "Shootout-random": 1.87815,
    "Shootout-C++-hash2": 1.95104,
    "fannkuch": 1.96626,
    "stepanov_vector": 2.21538,
    "Shootout-C++-heapsort": 2.26614,
    "oourafft": 2.3183,
    "ray": 2.33063,
    "Shootout-heapsort": 2.35064,
    "fdtd-apml": 2.40065,
    "floyd-warshall": 2.40321,
    "Shootout-C++-lists": 2.66023,
    "functionobjects": 2.86602,
    "syrk": 2.94183,
    "Shootout-sieve": 3.089,
    "ReedSolomon": 3.31701,
    "Shootout-hash": 3.32685,
    "gramschmidt": 3.43861,
    "lpbench": 3.48094,
    "Shootout-methcall": 3.58161,
    "sphereflake": 3.63097,
    "Shootout-lists": 4.02499,
    "Shootout-C++-methcall": 4.13668,
    "stepanov_container": 4.20587,
    "stepanov_abstraction": 4.2305,
    "flops": 4.35059,
    "salsa20": 4.56787,
    "almabench": 4.93649,
    "correlation": 5.05741,
    "covariance": 5.13021,
    "syr2k": 5.7571,
    "spirit": 5.86565,
    "smallpt": 6.08772,
    "stepanov_v1p2": 8.21778,
    "huffbench": 11.1218,
    "symm": 34.2176,
}

O0_RUNTIMES = {
    "RealMM": 0.00276,
    "Oscar": 0.00464,
    "lu": 0.00647,
    "Shootout-ackermann": 0.0107,
    "jacobi-1d-imper": 0.012,
    "Queens": 0.01861,
    "Towers": 0.02624,
    "Perm": 0.03101,
    "Bubblesort": 0.04715,
    "Quicksort": 0.06027,
    "Shootout-C++-strcat": 0.08438,
    "Treesort": 0.1046,
    "partialsums": 0.13773,
    "Shootout-strcat": 0.14104,
    "Shootout-C++-ary2": 0.15385,
    "Shootout-C++-ary": 0.15752,
    "reg_detect": 0.18549,
    "Shootout-C++-except": 0.22524,
    "Puzzle": 0.23318,
    "trisolv": 0.24745,
    "misr": 0.28746,
    "bicg": 0.35639,
    "revertBits": 0.35818,
    "atax": 0.38457,
    "jacobi-2d-imper": 0.401,
    "Shootout-C++-moments": 0.48918,
    "gesummv": 0.5445,
    "Shootout-C++-lists1": 0.5579,
    "seidel-2d": 0.61825,
    "oopack_v1p8": 0.63265,
    "mvt": 0.65357,
    "gemver": 0.80912,
    "mandel": 0.83378,
    "Shootout-C++-hash": 1.00353,
    "spectral-norm": 1.03618,
    "dt": 1.10656,
    "bigfib": 1.11124,
    "pi": 1.13384,
    "FloatMM": 1.14495,
    "nsieve-bits": 1.18237,
    "puzzle": 1.21489,
    "evalloop": 1.30215,
    "n-body": 1.3114,
    "recursive": 1.31658,
    "durbin": 1.32312,
    "fasta": 1.34667,
    "cholesky": 1.46967,
    "richards_benchmark": 1.5741,
    "chomp": 1.58664,
    "fdtd-2d": 1.70716,
    "fbench": 1.79521,
    "flops-2": 2.11949,
    "mandel-2": 2.12365,
    "dynprog": 2.38503,
    "whetstone": 2.39068,
    "fdtd-apml": 2.44071,
    "ffbench": 2.44825,
    "mandel-text": 2.58123,
    "Shootout-objinst": 2.65195,
    "queens": 2.71779,
    "flops-7": 3.07039,
    "Shootout-C++-ackermann": 3.20339,
    "doitgen": 3.23637,
    "fftbench": 3.31467,
    "Shootout-fib2": 3.43211,
    "Shootout-ary3": 3.5125,
    "adi": 3.92523,
    "Shootout-C++-objinst": 4.05989,
    "trmm": 4.12851,
    "flops-1": 4.17779,
    "Shootout-hash": 4.3231,
    "Shootout-C++-fibo": 4.43291,
    "Shootout-C++-heapsort": 4.48122,
    "matmul_f64_4x4": 4.66332,
    "flops-4": 4.79805,
    "flops-3": 5.22597,
    "flops-6": 5.26236,
    "fannkuch": 5.33084,
    "gramschmidt": 5.38356,
    "Shootout-methcall": 5.44303,
    "almabench": 5.53802,
    "flops-8": 5.68613,
    "Shootout-heapsort": 5.91763,
    "dry": 5.93107,
    "Shootout-C++-sieve": 6.22304,
    "flops-5": 6.26221,
    "Shootout-random": 6.42834,
    "fldry": 6.43594,
    "himenobmtxpa": 6.44691,
    "Shootout-C++-hash2": 6.64699,
    "Shootout-C++-random": 6.65798,
    "functionobjects": 6.85954,
    "correlation": 6.93813,
    "lpbench": 7.21615,
    "syrk": 7.3722,
    "covariance": 7.38984,
    "Shootout-C++-matrix": 8.47549,
    "fp-convert": 8.81604,
    "simple_types_loop_invariant": 9.26083,
    "Shootout-matrix": 9.43305,
    "Shootout-C++-ary3": 9.54945,
    "perlin": 10.1473,
    "ray": 10.5286,
    "sphereflake": 11.3737,
    "floyd-warshall": 11.711,
    "stepanov_container": 11.727,
    "Shootout-C++-methcall": 11.8655,
    "linpack-pc": 11.9292,
    "oourafft": 13.533,
    "syr2k": 13.9077,
    "ReedSolomon": 14.6436,
    "loop_unroll": 15.0418,
    "smallpt": 15.7597,
    "Shootout-sieve": 15.9806,
    "simple_types_constant_folding": 16.1437,
    "Shootout-nestedloop": 17.3048,
    "flops": 17.4165,
    "Shootout-C++-nestedloop": 17.8144,
    "Shootout-C++-lists": 17.9243,
    "salsa20": 22.7238,
    "Shootout-lists": 25.3135,
    "spirit": 39.4506,
    "huffbench": 42.7719,
    "lowercase": 43.2052,
    "symm": 53.4342,
    "stepanov_vector": 54.6562,
    "stepanov_abstraction": 73.4996,
    "stepanov_v1p2": 125.414,
}
