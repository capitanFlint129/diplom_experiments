import os
import os.path
import shutil

from tqdm import tqdm

from config.config import LLVM_TEST_SUITE_DATASET_PATH
from runtime_eval.llvm_test_suite.consts import LLVM_TEST_SUITE_PATH


def _find_bc_files(directory: str) -> list:
    bc_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bc"):
                bc_files.append(os.path.join(root, file))
    return bc_files


def main():
    os.makedirs(f"../../{LLVM_TEST_SUITE_DATASET_PATH}", exist_ok=True)
    bc_files = _find_bc_files(LLVM_TEST_SUITE_PATH)
    for filepath in tqdm(bc_files):
        filename = os.path.basename(filepath)
        if "Shootout-C++" in filepath:
            filename = f"Shootout-C++-{filename}"
        elif "Shootout" in filepath:
            filename = f"Shootout-{filename}"
        shutil.copy(filepath, f"../../{LLVM_TEST_SUITE_DATASET_PATH}/{filename}")
    assert len(os.listdir("../../_my_data/llvm_test_suite")) == 128


if __name__ == "__main__":
    main()
