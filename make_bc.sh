DATASET_DIR="$HOME/diplom/datasets/raw/angha_kernels_largest_10k"
OUTPUT_DIR="$HOME/diplom/datasets/bc/angha_kernels_largest_10k"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR" || exit

for file in "$DATASET_DIR"/*; do
  if [ -f "$file" ]; then
    ~/.local/share/compiler_gym/llvm-v0/bin/clang -emit-llvm -c -O0 -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes "$file"
  fi
done