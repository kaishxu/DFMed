output_dir="./train"
train_name="demo"
checkpoint="step-*"

python main.py \
        --mode evaluate \
        --data_name kamed \
        --data_type test \
        --output_dir "${output_dir}" \
        --eval_model_path "${output_dir}/models/${train_name}/${checkpoint}" \
        --per_gpu_eval_batch_size 16 \
        --result_save_dir "${output_dir}/results/${train_name}" \
        --train_name "${train_name}" \
