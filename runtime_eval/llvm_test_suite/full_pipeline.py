import argparse
import os
import shutil
import subprocess

from runtime_eval.llvm_test_suite.consts import LLVM_TEST_SUITE_TEST_PATH, TMP_DATA_DIR

MODELS = [
    # представления
    "autophase-264",
    # "ir2vec-265",
    # "instcount-271",
    # # пространства действий
    # "poset_odg-259",
    # "poset_manual_o3-260",
    # "micomp_actions-270",
    # "poset_o23_manual_with_o23-272",
    # "o3_with_full_action_set-275",
]


def generate_optimizations(model_name: str):
    command = [
        "python",
        "runtime_eval/llvm_test_suite/generate_optimizations.py",
        "--last_iter",
        model_name,
    ]
    if args.hack:
        command.append("--hack")
    if args.last_iter:
        command.append("--last_iter")
    proc = subprocess.run(command)
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"generate_optimizations: error for model {model_name}")


def compile_with_optimizations(model_name: str):
    command = [
        "python",
        "runtime_eval/llvm_test_suite/compile_with_optimizations.py",
        model_name,
    ]
    proc = subprocess.run(command)
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"compile_with_optimizations: error for model {model_name}")


def replace_with_generated_bins(model_name: str):
    command = [
        "python",
        "runtime_eval/llvm_test_suite/replace_with_generated_bins.py",
        model_name,
    ]
    proc = subprocess.run(command)
    if proc.returncode != 0:
        print(proc.stderr)
        raise Exception(f"replace_with_generated_bins: error for model {model_name}")


def lit_run(model_name: str, runs: int = 1):
    run_dir_path = f"{TMP_DATA_DIR}/{model_name}"
    results_path = f"{run_dir_path}/results"
    os.makedirs(results_path, exist_ok=True)
    for i in range(runs):
        if args.hack:
            result_filename = f"{model_name}_hack_{i}.json"
        else:
            result_filename = f"{model_name}_{i}.json"
        proc = subprocess.run(
            f"sudo lit -v -j 1 --time-tests -o {result_filename} .",
            shell=True,
            capture_output=True,
            cwd=LLVM_TEST_SUITE_TEST_PATH,
        )
        if proc.returncode != 0:
            print("ERROR: tests failed")
            print(proc.stderr)
            # raise Exception(f"lit_run: error for model {model_name}")
        shutil.copy(
            os.path.join(LLVM_TEST_SUITE_TEST_PATH, result_filename),
            os.path.join(results_path, result_filename),
        )


def main():
    for i, model_name in enumerate(MODELS):
        print(f"Model {i + 1}/{len(MODELS)}: {model_name}")
        generate_optimizations(model_name)
        compile_with_optimizations(model_name)
        replace_with_generated_bins(model_name)
        lit_run(model_name, runs=args.lit_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hack",
        help="eval_mode",
        action="store_true",
    )
    parser.add_argument(
        "--last_iter",
        help="last_iter",
        action="store_true",
    )
    parser.add_argument(
        "--lit_runs",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    main()
