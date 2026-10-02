# base training
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file acc_configs/gpu7.yaml \
    -m tokengs.train train_re10k_base_2_input_views \
    --workspace output_dir/tokengs/re10k/locusgs_full \
    --model_type locusgs --batch_size 8 --num_gaussians_per_token 64 --cross_attn_variant geometric_positional --use_anchor_radius --anchor_radius_refinement


# eval for base
CUDA_VISIBLE_DEVICES=3 python -m debugpy --listen 5678 /home/pdl/miniconda3/envs/spann3r/bin/accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_re10k_2view_base \
    --workspace output_dir/tokengs_eval_results/re10k/2view/locusgs_full \
    --resume output_dir/tokengs/re10k/locusgs_full/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 130 \
    --model_type locusgs --cross_attn_variant geometric_positional --use_anchor_radius --anchor_radius_refinement

# show videos for base 
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate_weighted eval_re10k_2view_base \
    --workspace output_dir/tokengs_eval_results/re10k/2view/locusgs_full_show \
    --resume output_dir/tokengs/re10k/locusgs_full/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 130 \
    --model_type locusgs --cross_attn_variant geometric_positional --use_anchor_radius --anchor_radius_refinement \
    --batch_size 1 --dataset_kwargs evaluation_json assets/evaluation_index_re10k_128_random_all_middle.json

# finetune
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file acc_configs/gpu7.yaml \
    -m tokengs.train finetune_re10k_2view \
    --workspace output_dir/tokengs/re10k/locusgs_full_finetune \
    --model_type locusgs --batch_size 2 --cross_attn_variant geometric_positional --use_anchor_radius --anchor_radius_refinement \
    --resume output_dir/tokengs/re10k/locusgs_full/checkpoints/checkpoint_latest/model.safetensors

# eval for finetune
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate_weighted eval_re10k_2view \
    --workspace output_dir/tokengs_eval_results/re10k/2view/locusgs_full_finetune \
    --resume output_dir/tokengs/re10k/locusgs_full_finetune/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 0 \
    --model_type locusgs --cross_attn_variant geometric_positional  --use_anchor_radius --anchor_radius_refinement \
    --batch_size 16
