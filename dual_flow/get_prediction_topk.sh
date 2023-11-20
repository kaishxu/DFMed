output_dir="./train"
train_name="demo"

# For MedDG
python get_prediction_topk.py \
        --data_name meddg \
        --data_type test \
        --log_dir "${output_dir}/log/${train_name}" \
        --saved_model_dir "${output_dir}/models/${train_name}" \
        --saved_result_dir "${output_dir}/results/${train_name}" \
        --top_k 3 \
        --get_acts \
        --get_selected_entitis \ # get selected entities, only for MedDG

# For KaMed
# python get_prediction_topk.py \
#         --data_name kamed \
#         --data_type test \
#         --log_dir "${output_dir}/log/${train_name}" \
#         --saved_model_dir "${output_dir}/models/${train_name}" \
#         --saved_result_dir "${output_dir}/results/${train_name}" \
#         --top_k 3 \
#         --get_acts \
#         --get_ranked_entitis \ # get ranked entities, only for KaMed
