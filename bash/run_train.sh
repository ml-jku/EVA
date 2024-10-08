#!/usr/bin/env bash
export WANDB_ENTITY=
export WANDB_PROJECT=
export WANDB_DISABLED=true

cd ..

master_addr="localhost"
min_port=49152
max_port=65535
range=$((max_port - min_port + 1))
random_offset=$((RANDOM % range))
random_port=$((min_port + random_offset))

base_path=
n_proc=1
# meta-llama/Llama-2-7b-hf
# meta-llama/Meta-Llama-3.1-8B
# google/gemma-2-9b
model_names=(meta-llama/Llama-2-7b-hf)
target_modules=()
ignore_modules=()

# meta-math/MetaMathQA 512
# qa_datasets 1024
# m-a-p/Code-Feedback 2048
dataset="qa_datasets 1024"
filter_long_context_examples=True
batch_size=4
gradient_accumulation_steps=2
num_train_epochs=1
batch_size_svd=16
svd_early_stop_sim_thresh=0.99
adapter_types=(eva_rho1) # eva_rho1 eva_rho2 lora pissa olora adalora eva_rho1_dora eva_rho2_dora dora
seeds=(0) # 0 10 101
lrs=(5e-4)
adapter_dims=(16)

read dataset_name model_max_length <<< "$dataset"

# convert target modules to string
target_modules_string="${target_modules[@]}"

# add filter_long_context_examples argument
if [ "${filter_long_context_examples,,}" = "true" ]; then
    filter_long_context_examples_arg="--whiten"
else
    filter_long_context_examples_arg=""
fi

for model_name in "${model_names[@]}"; do
  for adapter_type in "${adapter_types[@]}"; do
    for seed in "${seeds[@]}"; do
      for lr in "${lrs[@]}"; do
        for adapter_dim in "${adapter_dims[@]}"; do
          #convert model and dataset names
          model_name_safe=${model_name##*/}
          dataset_name_safe=${dataset_name##*/}
          # construct svd file path
          svd_filename="${model_name_safe}_${dataset_name_safe}_len${model_max_length}_r${adapter_dim}_rho_bs${batch_size_svd}_seed${seed}_ratio0.99_svd.bin"
          svd_filepath="${base_path}/${dataset_name_safe}/${model_name_safe}/${svd_filename}"
          dst_path="${base_path}/${dataset_name_safe}/${model_name_safe}/${adapter_type}/${adapter_dim}/${lr}/${seed}"
          if [ ! -d $dst_path ]; then
              # Conditional script execution based on adapter_type
              bash bash/_run_train_${adapter_type}.sh \
                  "${master_addr}" \
                  "${random_port}" \
                  "${n_proc}" \
                  "${model_name}" \
                  "${dataset_name}" \
                  "${dst_path}" \
                  "${num_train_epochs}" \
                  "${batch_size}" \
                  "${gradient_accumulation_steps}" \
                  "${lr}" \
                  "${adapter_dim}" \
                  "${seed}" \
                  "${target_modules_string}" \
                  "${model_max_length}" \
                  "${svd_filepath}" \
                  "${filter_long_context_examples_arg}"
          fi
        done
      done
    done
  done
done