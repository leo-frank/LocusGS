# base training
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train train_dl3dv_base \
    --workspace output_dir/locusgs/dl3dv/locusgs_full \
    --batch_size 8

# finetune 80 epochs 
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train finetune_dl3dv_4view \
    --workspace output_dir/locusgs/dl3dv/locusgs_full_finetune \
    --batch_size 2 \
    --resume output_dir/locusgs/dl3dv/locusgs_full/checkpoints/epoch_000300/model.safetensors \
    

# eval for base 
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_4view_256 \
    --workspace output_dir/locusgs_eval_results/dl3dv/locusgs_full \
    --resume output_dir/locusgs/dl3dv/locusgs_full/checkpoints/checkpoint_latest/model.safetensors

# eval for finetune
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_4view \
    --workspace output_dir/locusgs_eval_results/dl3dv/locusgs_full_finetune_4views \
    --resume output_dir/locusgs/dl3dv/locusgs_full_finetune/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 140

# eval for finetune (2 views)
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_2view \
    --workspace output_dir/locusgs_eval_results/dl3dv/locusgs_full_finetune_2views \
    --resume output_dir/locusgs/dl3dv/locusgs_full_finetune/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 140

# eval for finetune (6 views)
accelerate launch --config_file acc_configs/gpu1.yaml \
    -m locusgs.evaluate eval_dl3dv_6view \
    --workspace output_dir/locusgs_eval_results/dl3dv/locusgs_full_finetune_6views \
    --resume output_dir/locusgs/dl3dv/locusgs_full_finetune/checkpoints/checkpoint_latest/model.safetensors \
    --eval_n_media_dumps 140
