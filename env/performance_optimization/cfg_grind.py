import json
import os.path
import subprocess

from env.performance_optimization.llvm import (
    compile_ll,
    compile_lm_safely,
)

CFG_GRIND_ASMMAP_BIN = "cfggrind_asmmap"
CFG_GRIND_INFO_BIN = "cfggrind_info"
VALGRIND_BIN = "valgrind"
CFG_GRIND_TMP_PATH = "_tmp_cfg_grind"
MAP_FILE = os.path.join(CFG_GRIND_TMP_PATH, "tmp.map")
CFG_FILE = os.path.join(CFG_GRIND_TMP_PATH, "tmp.cfg")


def get_executed_instructions(bin_path: str, execution_args: str) -> int:
    os.makedirs(CFG_GRIND_TMP_PATH, exist_ok=True)
    # proc = subprocess.run(
    #     [
    #         CFG_GRIND_ASMMAP_BIN,
    #         bin_path,
    #         ">",
    #         MAP_FILE,
    #     ],
    # )
    # proc = subprocess.run(
    #     f"{CFG_GRIND_ASMMAP_BIN} {bin_path} > {MAP_FILE}",
    #     shell=True,
    #     capture_output=True,
    # )
    #
    # if proc.returncode != 0:
    #     print(proc.stderr)
    #     raise Exception(f"cfggrind_asmmap failed {bin_path}: {proc.stderr}")

    proc = subprocess.run(
        [
            VALGRIND_BIN,
            "-q",
            "--tool=cfggrind",
            f"--cfg-outfile={CFG_FILE}",
            # f"--instrs-map={MAP_FILE}",
            "--cfg-dump=bubble",
            f"./{bin_path}",
        ]
        + execution_args.split(),
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"valgrind failed {bin_path}: {proc.stderr}")

    proc = subprocess.run(
        [
            CFG_GRIND_INFO_BIN,
            "-s",
            "program",
            # "-i",
            # MAP_FILE,
            "-m",
            "json",
            CFG_FILE,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"cfggrind failed {bin_path}: {proc.stderr}")
    cfg_grind_result = json.loads(proc.stdout.decode())
    return cfg_grind_result["dynamic"]["instructions"]["count"]


def compile_and_get_instructions(
    ir: str, sequence: list[str], result_path: str, execution_args: str, linkopts
) -> int:
    compile_lm_safely(ir, sequence, result_path, linkopts)
    return get_executed_instructions(result_path, execution_args)


def compile_and_get_instructions_no_sequence(
    ir: str, result_path: str, execution_args: str, linkopts
) -> int:
    with open(f"{result_path}.ll", "w") as ouf:
        ouf.write(ir)
    try:
        compile_ll(f"{result_path}.ll", result_path, linkopts=linkopts)
    except Exception as e:
        print(f"Compilation failed, recompile with -lm: {e}")
        compile_ll(f"{result_path}.ll", result_path, linkopts=linkopts + ["-lm"])
    return get_executed_instructions(result_path, execution_args)
