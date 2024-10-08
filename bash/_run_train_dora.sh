#!/usr/bin/env bash
master_addr=$1
random_port=$2
n_proc=$3
model_name=$4
dataset_name=$5
dst_path=$6
num_train_epochs=$7
batch_size=$8
gradient_accumulation_steps=$9
lr=${10}
adapter_dim=${11}
seed=${12}
target_modules_string=${13}
model_max_length=${14}
svd_filepath=${15}
filter_long_context_examples_arg=${16}

# dora doesnt fit in memory so we need to reduce the batch size
batch_size=$((batch_size / 2))
gradient_accumulation_steps=$((gradient_accumulation_steps * 2))

# convert to array
if [ -n "$target_modules_string" ]; then
    IFS=' ' read -ra target_modules <<< "$target_modules_string"
    target_modules_arg="--target_modules ${target_modules[*]}"
else
    target_modules_arg=""
fi

torchrun --standalone --master_addr ${master_addr} --master_port ${random_port} --nproc_per_node ${n_proc} train.py \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --bf16 True \
    --output_dir $dst_path \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate $lr \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 500 \
    --tf32 True \
    --lora_dim $adapter_dim \
    --adapter_type dora \
    --seed $seed \
    --model_max_length $model_max_length \
    --lora_init true \
    $filter_long_context_examples_arg \
    $target_modules_arg