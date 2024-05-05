import os
import os.path
import shutil

from tqdm import tqdm

from runtime_eval.llvm_test_suite.consts import LLVM_TEST_SUITE_PATH


def _find_bc_files(directory: str) -> list:
    bc_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bc"):
                bc_files.append(os.path.join(root, file))
    return bc_files


def main():
    os.makedirs("../../_my_data/llvm_test_suite", exist_ok=True)
    bc_files = _find_bc_files(LLVM_TEST_SUITE_PATH)
    for filepath in tqdm(bc_files):
        filename = os.path.basename(filepath)
        if "Shootout-C++" in filepath:
            filename = f"Shootout-C++-{filename}"
        elif "Shootout" in filepath:
            filename = f"Shootout-{filename}"
        shutil.copy(filepath, f"../../_my_data/llvm_test_suite/{filename}")
    assert len(os.listdir("../../_my_data/llvm_test_suite")) == 128


if __name__ == "__main__":
    main()
