#!/usr/bin/env bash
# tokengs base training (RE10K)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file acc_configs/gpu7.yaml \
    -m tokengs.train train_re10k_base_2_input_views \
    --workspace output_dir/tokengs/re10k/tokenGS \
    --model_type tokengs --batch_size 8 --num_gaussians_per_token 64

# eval for base 
CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_re10k_2view_base \
    --workspace output_dir/tokengs_eval_results/re10k/2view/tokengs_1024 \
    --resume output_dir/tokengs/re10k/tokenGS/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 130 \
    --model_type tokengs

# show videos for base 
CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate_weighted eval_re10k_2view_base \
    --workspace output_dir/tokengs_eval_results/re10k/2view/tokengs_1024_show \
    --resume output_dir/tokengs/re10k/tokenGS/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 130 \
    --model_type tokengs \
    --batch_size 1 --dataset_kwargs evaluation_json assets/evaluation_index_re10k_128_random_all_middle.json