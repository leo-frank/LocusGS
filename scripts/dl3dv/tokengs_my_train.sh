# tokengs base training (DL3DV)
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train train_dl3dv_base \
    --workspace output_dir/locusgs/dl3dv/tokenGS \
    --model_type tokengs --batch_size 8 --num_gaussians_per_token 64

# finetune 20 epochs batch size 4
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train finetune_dl3dv_4view \
    --workspace output_dir/locusgs/dl3dv/tokenGS-finetune_dl3dv_4view \
    --model_type tokengs --batch_size 4 --num_gaussians_per_token 64 \
    --resume output_dir/locusgs/dl3dv/tokenGS/checkpoints/epoch_000300/model.safetensors


# eval for base 
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_4view_256 \
    --workspace output_dir/locusgs_eval_results/dl3dv/tokengs \
    --resume output_dir/locusgs/dl3dv/tokenGS/checkpoints/epoch_000300/model.safetensors \
    --eval_n_media_dumps 20 \
    --model_type tokengs

# eval for finetune (4 input views)
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_4view \
    --workspace output_dir/locusgs_eval_results/dl3dv/tokengs_448_v4 \
    --resume output_dir/locusgs/dl3dv/tokenGS_finetune/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 0 \
    --model_type tokengs

# eval for finetune (2 input views)
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_2view \
    --workspace output_dir/locusgs_eval_results/dl3dv/tokengs_448_v2 \
    --resume output_dir/locusgs/dl3dv/tokenGS_finetune/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 0 \
    --model_type tokengs

# eval for finetune (6 input views)
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_6view \
    --workspace output_dir/locusgs_eval_results/dl3dv/tokengs_448_v6 \
    --resume output_dir/locusgs/dl3dv/tokenGS_finetune/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 0 \
    --model_type tokengs
