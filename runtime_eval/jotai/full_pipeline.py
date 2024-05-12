import argparse
import os
import shutil
import subprocess

from runtime_eval.llvm_test_suite.consts import (
    LLVM_TEST_SUITE_TEST_PATH,
    TMP_DATA_DIR,
    LLVM_TEST_SUITE_TEST_PATH_O3_COMPILED,
)

MODELS = [
    # награда
    # "best_model_mca-228",
    # "best_model_insts-229",
    # "best_model_runtime-230",
    # # представления
    # "autophase-264",
    # "ir2vec-265",
    # "instcount-271",
    # # пространства действий
    # "poset_odg-259",
    # "poset_manual_o3-260",
    # "micomp_actions-270",
    # "poset_o23_manual_with_o23-272",
    # "o3_with_full_action_set-275",
    # # особые случаи
    # # "__o3",
    # "__o3_opt",
    "__o0_opt",
]


def generate_optimizations(model_name: str, o3: bool = False, o0: bool = False):
    assert not (o3 and o0)
    command = [
        "python",
        "runtime_eval/jotai/generate_optimizations.py",
        "--last_iter",
        "--n",
        f"{args.n}",
        model_name,
    ]
    if args.hack:
        command.append("--hack")
    if o3:
        command.append("--o3")
    if o0:
        command.append("--o0")
    # if args.last_iter:
    #     command.append("--last_iter")
    proc = subprocess.run(command)
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"generate_optimizations: error for model {model_name}")


def compile_with_optimizations(model_name: str):
    command = [
        "python",
        "runtime_eval/jotai/compile_with_optimizations.py",
        model_name,
    ]
    proc = subprocess.run(command)
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"compile_with_optimizations: error for model {model_name}")


def save_runtimes(model_name: str):
    command = [
        "python",
        "runtime_eval/jotai/save_runtimes.py",
        model_name,
        "--runs",
        f"{args.runs}",
    ]
    proc = subprocess.run(command)
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"replace_with_generated_bins: error for model {model_name}")


def lit_run_o3(model_name: str, runs: int = 1):
    run_dir_path = f"{TMP_DATA_DIR}/{model_name}"
    results_path = f"{run_dir_path}/results"
    os.makedirs(results_path, exist_ok=True)
    for i in range(runs):
        result_filename = f"{model_name}_{i}.json"
        proc = subprocess.run(
            f"sudo lit -v -j 1 --time-tests -o {result_filename} .",
            shell=True,
            capture_output=True,
            cwd=LLVM_TEST_SUITE_TEST_PATH_O3_COMPILED,
        )
        if proc.returncode != 0:
            print("ERROR: tests failed")
            print(proc.stderr)
        shutil.copy(
            os.path.join(LLVM_TEST_SUITE_TEST_PATH_O3_COMPILED, result_filename),
            os.path.join(results_path, result_filename),
        )


def main():
    for i, model_name in enumerate(MODELS):
        print(f"Model {i + 1}/{len(MODELS)}: {model_name}")
        if model_name == "__o3_opt":
            generate_optimizations(model_name, o3=True)
            compile_with_optimizations(model_name)
            save_runtimes(model_name)
        elif model_name == "__o0_opt":
            generate_optimizations(model_name, o0=True)
            compile_with_optimizations(model_name)
            save_runtimes(model_name)
        else:
            generate_optimizations(model_name)
            compile_with_optimizations(model_name)
            save_runtimes(model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hack",
        help="eval_mode",
        action="store_true",
    )
    # parser.add_argument(
    # "--last_iter",
    # help="last_iter",
    # action="store_true",
    # )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    main()
