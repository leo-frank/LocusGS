#!/usr/bin/env bash
# tokengs base training (RE10K)
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train train_re10k_base_2_input_views \
    --workspace output_dir/locusgs/re10k/tokenGS \
    --model_type tokengs --batch_size 8 --num_gaussians_per_token 64

# eval for base 
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_re10k_2view_base \
    --workspace output_dir/locusgs_eval_results/re10k/2view/tokengs_1024 \
    --resume output_dir/locusgs/re10k/tokenGS/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 130 \
    --model_type tokengs

# show videos for base 
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_re10k_2view_base \
    --workspace output_dir/locusgs_eval_results/re10k/2view/tokengs_1024_show \
    --resume output_dir/locusgs/re10k/tokenGS/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 130 \
    --model_type tokengs \
    --batch_size 1 --dataset_kwargs evaluation_json assets/evaluation_index_re10k_128_random_all_middle.json