python main.py \
        --mode train \
        --data_name meddg \
        --data_type train \ #choose the division of the dataset
        --evaluate_during_training \
        --eval_steps 2500 \
        --per_gpu_train_batch_size 12 \
        --entity_update_steps 10000 \ #update entity each 100 steps before the first 10k steps
        --output_dir ./train \ #dir to save log and fine-tuned models
        --lr 3e-5 \
        --num_train_epochs 5 \
        --act_weight 0.05 \
        --entity_weight 1 \
        --for_meddg_160 \ #training entity predictor
        --train_name demo_meddg \ #training name
