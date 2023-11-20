output_dir="./train"
train_name="demo"

checkpoint="epoch-*"
python main.py \
        --mode inference \
        --data_name kamed \
        --data_type test \
        --output_dir "${output_dir}" \
        --eval_model_path "${output_dir}/models/${train_name}/${checkpoint}" \
        --per_gpu_eval_batch_size 16 \
        --decode_max_length 150 \
        --act_entity_dir ../results/df_results/kamed \
        --result_save_dir "${output_dir}/results/${train_name}" \
        --k_entity 10 \
        --top_p 0.35 \
        --num_beams 5 \

#if you want to evaluate our checkpoints, you can replace the "eval_model_path" with the downloaded checkpoints path
#and the "result_save_dir" with your expected path.
