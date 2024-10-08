#!/bin/bash

# user config
work_dir=
data_dir=
baseline_output_path=

num_gpu=1
# meta-llama/Llama-2-7b-hf
# meta-llama/Meta-Llama-3.1-8B
# google/gemma-2-9b
model_name=meta-llama/Llama-2-7b-hf
model_folder=${model_name##*/}
dataset_folder=qa_datasets
tasks=custom_hellaswag,custom_winogrande,custom_arc_challenge,custom_arc_easy,custom_piqa,custom_siqa,custom_openbookqa,custom_boolq
#num_fewshot=5

base_dir="${data_dir}/${dataset_folder}/${model_folder}"
trainer_state_file=trainer_state.json
tasks_safe=$(echo $tasks | sed 's/,/_/g')
dirname="tasks_${tasks_safe}"

cd $work_dir

# Check if variable is empty
if [[ -n "$num_fewshot" ]]; then
    num_fewshot_arg="--num_fewshot ${num_fewshot}"
    dirname="${dirname}_fewshot_${num_fewshot}"
else
    num_fewshot_arg=""
fi

# Initialize an empty array
matching_dirs=()
# Find all directories that contain trainer_state_file but not filename
while IFS= read -r -d '' dir; do
    if [ -f "$dir/$trainer_state_file" ] && [[ ! $dir =~ "checkpoint-" ]] && [[ ! -d "$dir/$dirname" ]]; then
        matching_dirs+=("$dir")
    fi
done < <(find "$base_dir" -type d -print0)

echo "matching directories found: ${matching_dirs[@]}"

# Generate a random 32-character hexadecimal hash
random_hash=$(openssl rand -hex 16)

# evaluate baseline
if [ ! -d $baseline_output_path ]; then
    lm_eval \
        --model vllm \
        --model_args pretrained=$model_name,trust_remote_code=True,tensor_parallel_size=$num_gpu \
        --tasks $tasks \
        --output_path $baseline_output_path \
        $num_fewshot_arg
fi

for path in "${matching_dirs[@]}"; do
    # save merged checkpoint to a random subdirectory
    merged_dir="$path/$random_hash"
    python3 $work_dir/save_merged_cp.py \
        --cp_path $path \
        --dst_path $merged_dir \
        --device cpu
    # run eval
    lm_eval \
        --model vllm \
        --model_args pretrained=$merged_dir,trust_remote_code=True,tensor_parallel_size=$num_gpu \
        --tasks $tasks \
        --output_path $path/$dirname \
        $num_fewshot_arg
    # remove temp directory
    rm -rf $merged_dir
done