#!/bin/bash

cd ..

# meta-llama/Llama-2-7b-hf
# meta-llama/Meta-Llama-3.1-8B
# google/gemma-2-9b
model_names=(meta-llama/Llama-2-7b-hf)
target_modules=()
# for the code feedback dataset set use_label_mask to True
# "m-a-p/Code-Feedback 2048"
datasets=(
    "meta-math/MetaMathQA 512"
    "qa_datasets 1024" 
)

base_path=
batch_sizes=(16)
seeds=(0) # 0 10 101
ranks=(16)
rhos=(1) # 2 1
early_stop_sim_thresh=0.99
early_stop_redist_metric=ratio
whiten=False
use_label_mask=False
min_batches=0
log_convergence_stats=True

# Check if target_modules is not empty
if [ "${#target_modules[@]}" -gt 0 ]; then
    target_modules_arg="--target_modules ${target_modules[*]}"
else
    target_modules_arg=""
fi

# add whiten argument
if [ "${whiten,,}" = "true" ]; then
    whiten_arg="--whiten"
else
    whiten_arg=""
fi

# add use_label_mask argument
if [ "${use_label_mask,,}" = "true" ]; then
    use_label_mask_arg="--use_label_mask"
else
    use_label_mask_arg=""
fi

# add log_convergence_stats argument
if [ "${log_convergence_stats,,}" = "true" ]; then
    log_convergence_stats_arg="--log_convergence_stats"
else
    log_convergence_stats_arg=""
fi

for model_name in "${model_names[@]}"; do
    for dataset in "${datasets[@]}"; do
        read dataset_name model_max_length <<< "$dataset"
        model_name_safe=${model_name##*/}
        dataset_name_safe=${dataset_name##*/}
        svd_path="${base_path}/${dataset_name_safe}/${model_name_safe}"
        for batch_size in "${batch_sizes[@]}"; do
            for seed in "${seeds[@]}"; do
                for rank in "${ranks[@]}"; do
                    for rho in "${rhos[@]}"; do
                        echo "Running SVD precompute for ${model_name} on ${dataset_name} with batch size ${batch_size}, seed ${seed}, rank ${rank}, rho ${rho}"
                        python3 svd_precompute.py \
                            --model_name $model_name \
                            --dataset_name $dataset_name \
                            --svd_path $svd_path \
                            --rank $rank \
                            --rho $rho \
                            --early_stop_sim_thresh $early_stop_sim_thresh \
                            --early_stop_redist_metric $early_stop_redist_metric \
                            --batch_size $batch_size \
                            --model_max_length $model_max_length \
                            --seed $seed \
                            --min_batches $min_batches \
                            --filter_long_context_samples \
                            $target_modules_arg \
                            $whiten_arg \
                            $use_label_mask_arg \
                            $log_convergence_stats_arg
                    done
                done
            done
        done
    done
done
