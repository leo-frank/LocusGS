# tokengs base training (DL3DV)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file acc_configs/gpu8.yaml \
    -m tokengs.train train_dl3dv_base \
    --workspace output_dir/tokengs/dl3dv/tokenGS \
    --model_type tokengs --batch_size 8 --num_gaussians_per_token 64

# finetune 20 epochs batch size 4
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file acc_configs/gpu8.yaml \
    -m tokengs.train finetune_dl3dv_4view \
    --workspace output_dir/tokengs/dl3dv/tokenGS-finetune_dl3dv_4view_b4 \
    --model_type tokengs --batch_size 4 --num_gaussians_per_token 64 \
    --resume output_dir/tokengs/dl3dv/tokenGS/checkpoints/epoch_000300/model.safetensors


# eval for base 
CUDA_VISIBLE_DEVICES=6 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_4view_256 \
    --workspace output_dir/tokengs_eval_results/dl3dv/tokengs \
    --resume output_dir/tokengs/dl3dv/tokenGS/checkpoints/epoch_000300/model.safetensors \
    --eval_n_media_dumps 20 \
    --model_type tokengs

# eval for finetune (4 input views)
CUDA_VISIBLE_DEVICES=6 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_4view \
    --workspace output_dir/tokengs_eval_results/dl3dv/tokengs_448_b4_v4 \
    --resume output_dir/tokengs/dl3dv/tokenGS_finetune_b4/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 0 \
    --model_type tokengs

# eval for finetune (2 input views)
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_2view \
    --workspace output_dir/tokengs_eval_results/dl3dv/tokengs_448_b4_v2 \
    --resume output_dir/tokengs/dl3dv/tokenGS_finetune_b4/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 0 \
    --model_type tokengs

# eval for finetune (6 input views)
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_6view \
    --workspace output_dir/tokengs_eval_results/dl3dv/tokengs_448_b4_v6 \
    --resume output_dir/tokengs/dl3dv/tokenGS_finetune_b4/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 0 \
    --model_type tokengs



###### 别再用 b2 这个啦

# finetune 20 epochs batch size 2
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --config_file acc_configs/gpu7.yaml \
    -m tokengs.train finetune_dl3dv_4view \
    --workspace output_dir/tokengs/dl3dv/tokenGS-finetune_dl3dv_4view \
    --model_type tokengs --batch_size 2 --num_gaussians_per_token 64 \
    --resume output_dir/tokengs/dl3dv/tokenGS/checkpoints/epoch_000300/model.safetensors


# eval for finetune batch size 2 
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_4view \
    --workspace output_dir/tokengs_eval_results/dl3dv/tokengs_448 \
    --resume output_dir/tokengs/dl3dv/tokenGS-finetune_dl3dv_4view_b2/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 0 \
    --model_type tokengs