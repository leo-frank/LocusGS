# base training
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train train_re10k_base_2_input_views \
    --workspace output_dir/locusgs/re10k/locusgs_full \
    --batch_size 8


# eval for base
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_re10k_2view_base \
    --workspace output_dir/locusgs_eval_results/re10k/2view/locusgs_full \
    --resume output_dir/locusgs/re10k/locusgs_full/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 130

# show videos for base 
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_re10k_2view_base \
    --workspace output_dir/locusgs_eval_results/re10k/2view/locusgs_full_show \
    --resume output_dir/locusgs/re10k/locusgs_full/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 130 \
    --dataset_kwargs evaluation_json assets/evaluation_index_re10k_128_random_all_middle.json

# finetune
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train finetune_re10k_2view \
    --workspace output_dir/locusgs/re10k/locusgs_full_finetune \
    --batch_size 2 \
    --resume output_dir/locusgs/re10k/locusgs_full/checkpoints/checkpoint_latest/model.safetensors

# eval for finetune
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_re10k_2view \
    --workspace output_dir/locusgs_eval_results/re10k/2view/locusgs_full_finetune \
    --resume output_dir/locusgs/re10k/locusgs_full_finetune/checkpoints/checkpoint_latest/model.safetensors \
    --batch_size 16
