#!/bin/bash

# user config
work_dir=
data_dir=
bigcode_repo_dir=

num_gpu=1
model_folder=Llama-2-7b-hf
dataset_folder=Code-Feedback

tasks=humaneval,humanevalplus,mbpp,mbppplus
top_p=0.7
temperature=0.2
max_length_generation=2048

base_dir=$data_dir/$dataset_folder/$model_folder
trainer_state_file=trainer_state.json
tasks_safe=$(echo $tasks | sed 's/,/_/g')
filename="evaluation_results_temp${temperature}_p${top_p}_${tasks_safe}.json"
save_generations_fileref="temp${temperature}_p${top_p}."

cd $work_dir

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
    merged_dir="${path}/${random_hash}"
    save_generations_path="${path}/${save_generations_fileref}"
    metric_output_path="${path}/${filename}"
    python3 save_merged_cp.py \
        --cp_path $path \
        --dst_path $merged_dir \
        --device cpu
    # run eval
    accelerate launch --num_processes $num_gpu --num_machines 1 --mixed_precision no --dynamo_backend no $bigcode_repo_dir/main.py \
        --model $merged_dir \
        --tasks $tasks \
        --max_length_generation $max_length_generation \
        --temperature $temperature \
        --top_p $top_p \
        --do_sample True \
        --n_samples 1 \
        --batch_size 1 \
        --precision bf16 \
        --allow_code_execution \
        --save_generations \
        --save_generations_path $save_generations_path \
        --metric_output_path $metric_output_path \
        --max_memory_per_gpu auto
    # remove temp directory
    rm -rf $merged_dir
done