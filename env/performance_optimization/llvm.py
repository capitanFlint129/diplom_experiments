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


def compile_one_source_with_opt_sequence(
    source_path: str,
    result_path: str,
    sequence: list[str],
    linkopts: list[str],
    tmpfilename: str = "tmp.o",
):
    if len(sequence) == 0:
        proc = subprocess.run(
            " | ".join(
                [
                    f"{CLANG_BIN} {O0} --target=x86_64 -Xclang -disable-O0-optnone -S -emit-llvm -x c -o - {source_path}",
                    f"{LLC_BIN} -filetype=obj -o {tmpfilename}",
                ]
            ),
            shell=True,
            capture_output=True,
        )
    else:
        proc = subprocess.run(
            " | ".join(
                [
                    f"{CLANG_BIN} {O0} --target=x86_64 -Xclang -disable-O0-optnone -S -emit-llvm -x c -o - {source_path}",
                    f"{OPT_BIN} {' '.join(sequence)} -S -o -",
                    f"{LLC_BIN} -filetype=obj -o {tmpfilename}",
                ]
            ),
            shell=True,
            capture_output=True,
        )

    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"\nCompilation failed - {' '.join(sequence)}:\n {proc.stderr}")
    proc = subprocess.run(
        [
            CLANG_BIN,
            tmpfilename,
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
        raise Exception(
            f"\nCompilation failed {CLANG_BIN} - {' '.join(sequence)}:\n{proc.stderr}"
        )


def linkopts_safe_compile(compile, linkopts: list[list[str]]):
    for opts in linkopts:
        try:
            compile(opts)
            return
        except Exception:
            # print(f"Failed to compile with {opts} try to use next linkopts")
            pass
    raise Exception(f"failed to compile with {linkopts}")


def compile_bc_source_with_opt_sequence(
    source_path: str,
    result_path: str,
    sequence: list[str],
    linkopts: list[str],
    tmpfilename: str = "tmp.o",
):
    if len(sequence) == 0:
        proc = subprocess.run(
            f"{LLC_BIN} -filetype=obj -o {tmpfilename} {source_path}",
            shell=True,
            capture_output=True,
        )
    else:
        proc = subprocess.run(
            " | ".join(
                [
                    f"{OPT_BIN} {' '.join(sequence)} -S -o - {source_path}",
                    f"{LLC_BIN} -O=3 -filetype=obj -o {tmpfilename}",
                ]
            ),
            shell=True,
            capture_output=True,
        )

    if proc.returncode != 0:
        # print(proc.stderr)
        raise Exception(f"\nCompilation failed - {' '.join(sequence)}:\n {proc.stderr}")
    proc = subprocess.run(
        [
            CLANG_BIN,
            tmpfilename,
        ]
        + linkopts
        + [
            # "-nostartfiles",
            # "-shared",
            "-no-pie",
            "-o",
            result_path,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        # print(proc.stderr)
        raise Exception(
            f"\nCompilation failed {CLANG_BIN} - {' '.join(sequence)}:\n{proc.stderr}"
        )


def compile_ll_with_opt_sequence(ir, result_path, sequence, linkopts):
    proc = subprocess.run(
        " | ".join(
            [
                f"{OPT_BIN} {' '.join(sequence)} -S -o {result_path}.ll",
                # f"{LLC_BIN} -filetype=obj -o tmp.o -lm",
            ]
        ),
        input=ir.encode(),
        shell=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        # print(proc.stderr)
        raise Exception(f"Compilation failed {proc.stderr}")
    compile_ll(f"{result_path}.ll", result_path, linkopts=linkopts)


# def compile_bc_source_with_opt_sequence(source_path, result_path, sequence, linkopts):
#     proc = subprocess.run(
#         f"{OPT_BIN} {' '.join(sequence)} -S -o {result_path}.ll {source_path}",
#         shell=True,
#         capture_output=True,
#     )
#     if proc.returncode != 0:
#         print(proc.stderr)
#         raise Exception(f"Compilation failed {proc.stderr}")
#     compile_ll(f"{result_path}.ll", result_path, linkopts=linkopts)


def compile_ll(source_path, result_path, linkopts):
    proc = subprocess.run(
        [
            LLC_BIN,
            "-O=3",
            "-filetype=obj",
            source_path,
            "-o",
            f"{result_path}.o",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        # print(proc.stderr)
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
        # print(proc.stderr)
        raise Exception(f"Compilation failed {proc.stderr}")


def compile_lm_safely(ir: str, sequence: list[str], result_path: str, linkopts):
    try:
        compile_ll_with_opt_sequence(
            ir,
            result_path=result_path,
            sequence=sequence,
            linkopts=linkopts,
        )
    except Exception as e:
        print(f"Compilation failed, recompile with -lm: {e}")
        compile_ll_with_opt_sequence(
            ir,
            result_path=result_path,
            sequence=sequence,
            linkopts=linkopts + ["-lm"],
        )


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
