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
