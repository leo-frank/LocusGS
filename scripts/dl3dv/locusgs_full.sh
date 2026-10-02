# base training
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file acc_configs/gpu7.yaml \
    -m locusgs.train train_dl3dv_base \
    --workspace output_dir/tokengs/dl3dv/locusgs_full \
    --model_type locusgs --batch_size 8 --num_gaussians_per_token 64 --cross_attn_variant geometric_positional --use_anchor_radius --anchor_radius_refinement

# finetune 80 epochs 
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file acc_configs/gpu7.yaml \
    -m locusgs.train finetune_dl3dv_4view \
    --workspace output_dir/tokengs/dl3dv/locusgs_full_finetune_80epochs \
    --model_type locusgs --batch_size 2 --num_gaussians_per_token 64 --cross_attn_variant geometric_positional \
    --resume output_dir/tokengs/dl3dv/locusgs_full/checkpoints/epoch_000300/model.safetensors \
    --num_epochs 80 --use_anchor_radius --anchor_radius_refinement

# eval for base 
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_4view_256 \
    --workspace output_dir/tokengs_eval_results/dl3dv/locusgs_full \
    --resume output_dir/tokengs/dl3dv/locusgs_full/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 0 \
    --model_type locusgs --cross_attn_variant geometric_positional --use_anchor_radius --anchor_radius_refinement

# eval for finetune
CUDA_VISIBLE_DEVICES=3 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_4view \
    --workspace output_dir/tokengs_eval_results/dl3dv/locusgs_full_finetune_80epochs_4views \
    --resume output_dir/tokengs/dl3dv/locusgs_full_finetune_80epochs/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 140 \
    --model_type locusgs --cross_attn_variant geometric_positional --use_anchor_radius --anchor_radius_refinement

# eval for finetune (2 views)
CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_2view \
    --workspace output_dir/tokengs_eval_results/dl3dv/locusgs_full_finetune_80epochs_2views \
    --resume output_dir/tokengs/dl3dv/locusgs_full_finetune_80epochs/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 140 \
    --model_type locusgs --cross_attn_variant geometric_positional --use_anchor_radius --anchor_radius_refinement

# eval for finetune (6 views)
CUDA_VISIBLE_DEVICES=5 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_6view \
    --workspace output_dir/tokengs_eval_results/dl3dv/locusgs_full_finetune_80epochs_6views \
    --resume output_dir/tokengs/dl3dv/locusgs_full_finetune_80epochs/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 140 \
    --model_type locusgs --cross_attn_variant geometric_positional --use_anchor_radius --anchor_radius_refinement
