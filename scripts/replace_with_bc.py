import os
import shutil


def replace_source_with_bytecode(source_dir, bytecode_dir):
    """Replaces C/C++ source files in the source directory with corresponding bytecode files from the bytecode directory.

    Args:
        source_dir (str): Path to the directory containing the source files.
        bytecode_dir (str): Path to the directory containing the bytecode files.
    """

    for root, _, files in os.walk(source_dir):
        for filename in files:
            base, extension = os.path.splitext(filename)
            if extension in (".c", ".cc", ".cpp"):
                source_path = os.path.join(root, filename)
                bytecode_path = os.path.join(
                    bytecode_dir, root[len(source_dir) + 1 :], base + ".bc"
                )

                # Check if the corresponding bytecode file exists
                if os.path.exists(bytecode_path):
                    # Remove the source file
                    os.remove(source_path)
                    # Copy the bytecode file to the source directory
                    shutil.copy(bytecode_path, source_path.split(".")[0] + ".bc")
                else:
                    print(f"Warning: Bytecode file not found for {source_path}")


if __name__ == "__main__":
    source_directory = (
        "/home/flint/diplom/eval_llvm_test_suite/llvm-test-suite-bc-sources"
    )
    bytecode_directory = "/home/flint/diplom/eval_llvm_test_suite/build-O0-save-temps"
    replace_source_with_bytecode(source_directory, bytecode_directory)
