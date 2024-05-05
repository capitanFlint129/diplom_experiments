import argparse
import os.path
import shutil

import pandas as pd

from runtime_eval.llvm_test_suite.consts import TMP_DATA_DIR


def main():
    optimizations_filename = f"{RUN_DIR_PATH}/optimizations.csv"
    print(optimizations_filename)
    model_optimizations = pd.read_csv(optimizations_filename)
    name_to_path = dict(zip(model_optimizations.benchmark, model_optimizations.path))

    for name in os.listdir(f"{RUN_DIR_PATH}/bin"):
        if name.startswith("Shootout-C++"):
            dst_path = name_to_path[name.split("-")[2]].rsplit(".", maxsplit=1)[0]
            # before, after = dst_path.split("Shootout-C++")
            before, after = (
                dst_path.split("Shootout-C++")
                if "Shootout-C++" in dst_path
                else dst_path.split("Shootout")
            )
            after = after.rsplit("/", maxsplit=1)[0]
            dst_path = f"{before}Shootout-C++{after}/{name}"
        elif name.startswith("Shootout"):
            dst_path = name_to_path[name.split("-")[1]].rsplit(".", maxsplit=1)[0]
            before, after = (
                dst_path.split("Shootout-C++")
                if "Shootout-C++" in dst_path
                else dst_path.split("Shootout")
            )
            after = after.rsplit("/", maxsplit=1)[0]
            dst_path = f"{before}Shootout{after}/{name}"
        else:
            dst_path = name_to_path[name].rsplit(".", maxsplit=1)[0]
        assert os.path.isfile(dst_path)
        shutil.copyfile(f"{RUN_DIR_PATH}/bin/{name}", dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name")
    parser.add_argument(
        "--debug",
        help="debug",
        action="store_true",
    )
    args = parser.parse_args()

    RUN_DIR_PATH = f"{TMP_DATA_DIR}/{args.run_name}"

    main()
