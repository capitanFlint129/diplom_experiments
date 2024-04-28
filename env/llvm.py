import os
import subprocess
from typing import Optional

# noinspection PyUnresolvedReferences
from compiler_gym.wrappers import RuntimePointEstimateReward

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


def compile_one_source_with_opt_sequence(source_path, result_path, sequence):
    proc = subprocess.run(
        " | ".join(
            [
                f"{CLANG_BIN} {O0} --target=x86_64 -Xclang -disable-O0-optnone -S -emit-llvm -x c -o - {source_path}",
                f"opt {' '.join(sequence)} -S -o -",
                f"llc -filetype=obj -o tmp.o",
            ]
        ),
        shell=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"Compilation failed {proc.stderr}")
    proc = subprocess.run(
        [
            CLANG_BIN,
            "tmp.o",
            "-o",
            result_path,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"Compilation failed {proc.stderr}")


def compile_ll_with_opt_sequence(ir, result_path, sequence, linkopts):
    proc = subprocess.run(
        " | ".join(
            [
                f"opt {' '.join(sequence)} -S -o {result_path}.ll",
                # f"llc -filetype=obj -o tmp.o -lm",
            ]
        ),
        input=ir.encode(),
        shell=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"Compilation failed {proc.stderr}")
    compile_ll(f"{result_path}.ll", result_path, linkopts=linkopts)


def compile_ll(source_path, result_path, linkopts):
    proc = subprocess.run(
        [
            "llc",
            # "-O=3",
            "-filetype=obj",
            source_path,
            "-o",
            f"{result_path}.o",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"llc: Compilation failed {proc.stderr}")
    proc = subprocess.run(
        # clang hello-world.o -o hello-world
        [
            CLANG_BIN,
            f"{result_path}.o",
        ]
        + linkopts
        + [
            "-no-pie",
            "-o",
            result_path,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"Compilation failed {proc.stderr}")


def clang_compile_to_ir(
    source: str, level: str, result_path: str = "-"
) -> Optional[str]:
    proc = subprocess.run(
        [
            CLANG_BIN,
            level,
            "--target=x86_64",
            "-S",
            "-emit-llvm",
            "-x",
            "c",
            "-",
            "-o",
            result_path,
        ],
        input=source.encode(),
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"Clang compilation failed: {proc.stderr}")
    if result_path == "-":
        return proc.stdout.decode()
    return None


def clang_compile_to_ir_o0(source: str, result_path: str = "-") -> Optional[str]:
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
            result_path,
        ],
        input=source.encode(),
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"Clang compilation failed: {proc.stderr}")
    if result_path == "-":
        return proc.stdout.decode()
    return None


def opt(flags, input_file, result_file):
    proc = subprocess.run(
        [
            OPT_BIN,
        ]
        + flags
        + [
            input_file,
            "-o",
            result_file,
        ],
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"Opt step failed {proc.stderr}")
