
# eval for base 
CUDA_VISIBLE_DEVICES=5 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_4view_256 \
    --workspace output_dir/tokengs_eval_results/dl3dv/tokengs_offical_base_v4 \
    --resume ./official_ckpts/dl3dv_base.safetensors \
    --eval_n_media_dumps 140 \
    --model_type tokengs

# eval for finetune (4 input views)
CUDA_VISIBLE_DEVICES=6 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_4view \
    --workspace output_dir/tokengs_eval_results/dl3dv/tokengs_offical_v4 \
    --resume ./official_ckpts/dl3dv_4v.safetensors \
    --eval_n_media_dumps 140 \
    --model_type tokengs

# eval for finetune (2 input views)
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_2view \
    --workspace output_dir/tokengs_eval_results/dl3dv/tokengs_offical_v2 \
    --resume ./official_ckpts/dl3dv_4v.safetensors \
    --eval_n_media_dumps 140 \
    --model_type tokengs

# eval for finetune (6 input views)
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file acc_configs/gpu1.yaml \
    -m tokengs.evaluate eval_dl3dv_6view \
    --workspace output_dir/tokengs_eval_results/dl3dv/tokengs_offical_v6 \
    --resume ./official_ckpts/dl3dv_4v.safetensors \
    --eval_n_media_dumps 140 \
    --model_type tokengs
