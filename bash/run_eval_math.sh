#!/bin/bash

cd ..

base_dir=
model_folder=Llama-2-7b-hf
trainer_state_file=trainer_state.json
filename=result_math.txt

# Initialize an empty array
matching_dirs=()
# Find all directories that contain trainer_state_file but not filename
while IFS= read -r -d '' dir; do
    if [ -f "$dir/$trainer_state_file" ] && [ ! -f "$dir/$filename" ] && [[ ! $dir =~ "checkpoint-" ]]; then
        matching_dirs+=("$dir")
    fi
done < <(find "$base_dir" -type d -print0)

# Generate a random 32-character hexadecimal hash
random_hash=$(openssl rand -hex 16)

for path in "${matching_dirs[@]}"; do
    # save merged checkpoint to a random subdirectory
    merged_dir="$path/$random_hash"
    python3 save_merged_cp.py \
        --cp_path $path \
        --dst_path $merged_dir \
        --device cpu
    # run eval
    python3 eval_math.py \
        --model $merged_dir \
        --data_file data/test/MATH_test.jsonl \
        --batch_size 1 \
        --tensor_parallel_size 1 \
        --filepath_output "$path/$filename"
    # remove temp directory
    rm -rf $merged_dir
done