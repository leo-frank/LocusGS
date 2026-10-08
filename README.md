# <img src="./logo.svg" alt="LocusGS logo" width="48" /> LocusGS
**LocusGS: Spatially Grounded Tokens for Feed-Forward 3D Gaussian Splatting** <br>
Wenyu Li, Sidun Liu, Tongrui Hu, Peng Qiao, Yong Dou <br>
National University of Defense Technology

[**Paper**](https://arxiv.org/abs/2608.12825) · [**Project Page**](https://leo-frank.github.io/LocusGS_viewer/)

## Pretrained Checkpoints

Pretrained LocusGS checkpoints are now available on [Google Drive](https://drive.google.com/drive/folders/1_MontNvxTtXkwgluIhUUvOxd-UTyRlip?usp=drive_link).

LocusGS follows a **two-stage training** procedure:
1. **Base training:** Train the model with 1,024 Gaussian tokens at a resolution of 256 × 256.
2. **Fine-tuning:** Initialize from the base checkpoint and increase the token budget to 4,096 Gaussian tokens. For RE10K, the resolution remains 256 × 256; for DL3DV, it increases to 448 × 256.

Each token predicts 64 Gaussians. RE10K uses two input views for training and evaluation, while DL3DV uses four input views for training and is evaluated with two, four, or six input views.

## How to Train

Run the following commands from the repository root after preparing the datasets and configuring their paths. These commands follow the training blocks in [`scripts/re10k/locusgs_2_input_views.sh`](scripts/re10k/locusgs_2_input_views.sh) and [`scripts/dl3dv/locusgs.sh`](scripts/dl3dv/locusgs.sh), using the current `locusgs.train` module. The commands rely on the defaults in `locusgs/options.py` and the dataset-specific presets, so only options that override those defaults are shown.

### RE10K

**Stage 1: Base training**

```bash
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train train_re10k_base_2_input_views \
    --workspace output_dir/locusgs/re10k/locusgs \
    --batch_size 8
```

**Stage 2: Fine-tuning**

```bash
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train finetune_re10k_2view \
    --workspace output_dir/locusgs/re10k/locusgs_finetune \
    --batch_size 2 \
    --resume output_dir/locusgs/re10k/locusgs/checkpoints/checkpoint_latest/model.safetensors
```

### DL3DV

**Stage 1: Base training**

```bash
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train train_dl3dv_base \
    --workspace output_dir/locusgs/dl3dv/locusgs \
    --batch_size 8
```

**Stage 2: Fine-tuning**

```bash
accelerate launch --config_file acc_configs/gpu8.yaml \
    -m locusgs.train finetune_dl3dv_4view \
    --workspace output_dir/locusgs/dl3dv/locusgs_finetune \
    --batch_size 2 \
    --resume output_dir/locusgs/dl3dv/locusgs/checkpoints/checkpoint_latest/model.safetensors
```


## Acknowledgements

LocusGS is inspired by [TokenGS](https://github.com/nv-tlabs/TokenGS). We thank the authors for their inspiring work and for making their code publicly available.

## Citation

If you use LocusGS in your research, please cite:

```bibtex
@misc{li2026locusgsspatiallygroundedtokens,
      title={LocusGS: Spatially Grounded Tokens for Feed-Forward 3D Gaussian Splatting}, 
      author={Wenyu Li and Sidun Liu and Tongrui Hu and Peng Qiao and Yong Dou},
      year={2026},
      eprint={2608.12825},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.12825}, 
}
```
