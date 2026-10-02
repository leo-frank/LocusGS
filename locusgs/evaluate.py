# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a checkpoint on the test loader; supports optional `--config` YAML for tyro defaults."""

import os
import sys
import time
import warnings
from typing import Any
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
import tyro
from accelerate import Accelerator
from safetensors.torch import load_file
from tqdm import tqdm

from locusgs.data import get_multi_dataloader
from locusgs.models import model_registry
from locusgs.options import AllConfigs, Options
from locusgs.utils import MetricsCalculator, MetricsTracker
from locusgs.utils.images import save_image_grid_square, visualize_depth

warnings.filterwarnings("ignore")

torch.manual_seed(0)


def _pop_config_path_from_argv() -> str | None:
    """If argv contains `--config PATH`, remove both tokens and return PATH."""
    if "--config" not in sys.argv:
        return None
    idx = sys.argv.index("--config")
    if idx + 1 >= len(sys.argv):
        return None
    path = sys.argv[idx + 1]
    del sys.argv[idx : idx + 2]
    return path


def _load_options(accelerator: Accelerator, config_path: str | None) -> Options:
    if config_path is not None:
        model_path = os.path.join(os.path.dirname(config_path), "model.safetensors")
        accelerator.print(f"[INFO] loading config from --config={config_path}")
        with open(config_path, encoding="utf-8") as f:
            default_opt = tyro.extras.from_yaml(Options, f)
        opt = tyro.cli(Options, default=default_opt)
        if opt.resume is None:
            accelerator.print(f"[INFO] resume not provided, deducing {model_path=} from {config_path=}")
            if not os.path.exists(model_path):
                raise ValueError(f"{model_path=} does not exist")
            opt.resume = model_path
        return opt

    accelerator.print("[INFO] no config provided, using default options")
    return tyro.cli(AllConfigs)


def _load_checkpoint_state_dict(resume: str) -> dict[str, Any]:
    if resume.endswith("safetensors"):
        return load_file(resume, device="cpu")
    ckpt = torch.load(resume, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def _copy_matching_checkpoint(model: nn.Module, ckpt: dict[str, Any], log) -> None:
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k not in state_dict:
            log(f"[WARN] unexpected param {k}: {v.shape}")
            continue
        if state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
        else:
            log(
                f"[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored."
            )


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _views_chw_to_uint8_nhwc(views: torch.Tensor) -> np.ndarray:
    """(V, C, H, W) float in ~[0, 1] -> (V, H, W, C) uint8."""
    x = views.detach().cpu().numpy().transpose(0, 2, 3, 1)
    return (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)


def _blend_mask_overlay(vis: torch.Tensor, masks: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    return vis * masks + alpha * (1 - masks) * vis + (1 - alpha) * (1 - masks)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_depthsplat_cfg(opt: Options):
    depthsplat_root = _repo_root() / "third_party" / "depthsplat"
    if not depthsplat_root.is_dir():
        raise FileNotFoundError(f"DepthSplat root not found: {depthsplat_root}")

    if str(depthsplat_root) not in sys.path:
        sys.path.insert(0, str(depthsplat_root))

    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config

    overrides = [
        f"+experiment={opt.depthsplat_experiment}",
        "mode=test",
        f"model.encoder.num_scales={opt.depthsplat_num_scales}",
        f"model.encoder.upsample_factor={opt.depthsplat_upsample_factor}",
        f"model.encoder.lowest_feature_resolution={opt.depthsplat_lowest_feature_resolution}",
        f"model.encoder.monodepth_vit_type={opt.depthsplat_monodepth_vit_type}",
        "test.compute_scores=false",
        "test.save_image=false",
        "test.save_video=false",
        "test.save_gt_image=false",
        "test.save_input_images=false",
        "test.save_depth=false",
        "test.save_depth_npy=false",
        "test.save_depth_concat_img=false",
        "test.save_gaussian=false",
    ]

    with initialize_config_dir(version_base=None, config_dir=str(depthsplat_root / "config")):
        cfg_dict = compose(config_name="main", overrides=overrides)
    cfg = load_typed_root_config(cfg_dict)
    return cfg, depthsplat_root


def _build_depthsplat_model(opt: Options, accelerator: Accelerator):
    cfg, _depthsplat_root = _build_depthsplat_cfg(opt)

    from src.loss import get_losses
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker=None,
        eval_data_cfg=None,
    )
    if opt.resume is None or opt.resume == "None":
        raise ValueError("Resume path is required when evaluating a DepthSplat model")
    accelerator.print(f"[INFO] loading DepthSplat model from {opt.resume=}")
    ckpt = _load_checkpoint_state_dict(opt.resume)
    model.load_state_dict(ckpt, strict=True)
    accelerator.print("[INFO] DepthSplat model loaded!")
    return model, cfg


def _intrinsics_vec_to_mat_normalized(
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    scale_x = float(image_width)
    scale_y = float(image_height)
    fx = intrinsics[..., 0] / scale_x
    fy = intrinsics[..., 1] / scale_y
    cx = intrinsics[..., 2] / scale_x
    cy = intrinsics[..., 3] / scale_y
    out = torch.zeros(*intrinsics.shape[:-1], 3, 3, dtype=intrinsics.dtype, device=intrinsics.device)
    out[..., 0, 0] = fx
    out[..., 1, 1] = fy
    out[..., 0, 2] = cx
    out[..., 1, 2] = cy
    out[..., 2, 2] = 1.0
    return out


def _resolve_depthsplat_bounds(depthsplat_cfg) -> tuple[float, float]:
    near = float(depthsplat_cfg.dataset.near)
    far = float(depthsplat_cfg.dataset.far)
    if near == -1.0:
        near = 0.1
    if far == -1.0:
        far = 1000.0
    return near, far


def _tokengs_batch_to_depthsplat_batch(
    opt: Options,
    depthsplat_cfg,
    batch: dict[str, Any],
) -> dict[str, Any]:
    input_images = batch["images_input"]
    target_images = batch["images_output"]
    _, _, _, input_h, input_w = input_images.shape
    _, _, _, target_h, target_w = target_images.shape
    context_extrinsics = batch.get("raw_cam_to_world_input", batch["cam_to_world_input"])
    target_extrinsics = batch.get(
        "raw_cam_to_world_output",
        torch.linalg.inv(batch["cam_view"].transpose(-1, -2)),
    )
    context_intrinsics = _intrinsics_vec_to_mat_normalized(
        batch.get("raw_intrinsics_input", batch["intrinsics_input"]),
        input_h,
        input_w,
    )
    target_intrinsics = _intrinsics_vec_to_mat_normalized(
        batch.get("raw_intrinsics_output", batch["intrinsics"]),
        target_h,
        target_w,
    )

    batch_size, num_context = input_images.shape[:2]
    num_target = target_images.shape[1]
    device = input_images.device
    near_bound, far_bound = _resolve_depthsplat_bounds(depthsplat_cfg)
    near_ctx = torch.full((batch_size, num_context), near_bound, dtype=input_images.dtype, device=device)
    far_ctx = torch.full((batch_size, num_context), far_bound, dtype=input_images.dtype, device=device)
    near_tgt = torch.full((batch_size, num_target), near_bound, dtype=input_images.dtype, device=device)
    far_tgt = torch.full((batch_size, num_target), far_bound, dtype=input_images.dtype, device=device)
    context_index = torch.arange(num_context, device=device, dtype=torch.int64)[None].expand(batch_size, -1)
    target_index = torch.arange(num_target, device=device, dtype=torch.int64)[None].expand(batch_size, -1)
    scene_names = [f"sample_{idx}" for idx in range(batch_size)]

    return {
        "context": {
            "image": input_images,
            "extrinsics": context_extrinsics,
            "intrinsics": context_intrinsics,
            "near": near_ctx,
            "far": far_ctx,
            "index": context_index,
        },
        "target": {
            "image": target_images,
            "extrinsics": target_extrinsics,
            "intrinsics": target_intrinsics,
            "near": near_tgt,
            "far": far_tgt,
            "index": target_index,
        },
        "scene": scene_names,
}


def _run_depthsplat_inference(
    opt: Options,
    depthsplat_cfg,
    model: nn.Module,
    batch: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor | None]:
    batch = _tokengs_batch_to_depthsplat_batch(opt, depthsplat_cfg, batch)
    batch = model.data_shim(batch)
    _, _, _, h, w = batch["target"]["image"].shape

    pred_depths = None
    gaussians = model.encoder(
        batch["context"],
        model.global_step,
        deterministic=False,
        visualization_dump=None,
    )
    if isinstance(gaussians, dict):
        pred_depths = gaussians.get("depths")
        gaussians = gaussians["gaussians"]

    camera_poses = batch["target"]["extrinsics"]
    render_chunk_size = model.test_cfg.render_chunk_size
    if render_chunk_size is not None:
        num_chunks = int(np.ceil(camera_poses.shape[1] / render_chunk_size))
        output = None
        for chunk_idx in range(num_chunks):
            start = render_chunk_size * chunk_idx
            end = render_chunk_size * (chunk_idx + 1)
            curr_output = model.decoder.forward(
                gaussians,
                camera_poses[:, start:end],
                batch["target"]["intrinsics"][:, start:end],
                batch["target"]["near"][:, start:end],
                batch["target"]["far"][:, start:end],
                (h, w),
                depth_mode=None,
            )
            if output is None:
                output = curr_output
            else:
                output.color = torch.cat((output.color, curr_output.color), dim=1)
    else:
        output = model.decoder.forward(
            gaussians,
            camera_poses,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=None,
        )
    return batch, output.color, pred_depths


def _compute_compactness_centroid_per_sample(
    gaussians: torch.Tensor,
    num_gaussians_per_token: int,
) -> torch.Tensor:
    """Return per-sample token-group compactness averaged over tokens."""
    if num_gaussians_per_token <= 0:
        raise ValueError(f"num_gaussians_per_token must be positive, got {num_gaussians_per_token}")

    xyz = gaussians[..., :3]
    batch_size, total_gaussians, _ = xyz.shape
    if total_gaussians % num_gaussians_per_token != 0:
        raise ValueError(
            f"gaussian count {total_gaussians} is not divisible by num_gaussians_per_token={num_gaussians_per_token}"
        )

    num_tokens = total_gaussians // num_gaussians_per_token
    xyz_grouped = xyz.reshape(batch_size, num_tokens, num_gaussians_per_token, 3)
    group_centroids = xyz_grouped.mean(dim=2, keepdim=True)
    distances = torch.linalg.norm(xyz_grouped - group_centroids, dim=-1)
    return distances.mean(dim=2).mean(dim=1)


def main() -> None:
    accelerator = Accelerator()
    config_path = _pop_config_path_from_argv()
    opt = _load_options(accelerator, config_path)
    opt.use_input_supervision = False

    accelerator.print(f"[INFO] {'' if opt.use_ttt_for_eval else 'not '}using TTT for evaluation")
    is_depthsplat = opt.model_type == "depthsplat"
    if is_depthsplat:
        model, depthsplat_cfg = _build_depthsplat_model(opt, accelerator)
        _train_dl, test_dataloader, _, _ = get_multi_dataloader(opt, accelerator)
        model, _train_dl, test_dataloader = accelerator.prepare(model, _train_dl, test_dataloader)
        model.eval()
        base_model = accelerator.unwrap_model(model)
        num_gaussians_per_token = None
    else:
        model = model_registry[opt.model_type](opt)
        if opt.resume is None or opt.resume == "None":
            raise ValueError("Resume path is required when evaluating a model")

        accelerator.print(f"[INFO] loading model from {opt.resume=}")
        ckpt = _load_checkpoint_state_dict(opt.resume)
        _copy_matching_checkpoint(model, ckpt, accelerator.print)
        accelerator.print("[INFO] Model loaded!")

        _train_dl, test_dataloader, _, _ = get_multi_dataloader(opt, accelerator)
        model, _train_dl, test_dataloader = accelerator.prepare(model, _train_dl, test_dataloader)
        model.eval()
        base_model = accelerator.unwrap_model(model)
        num_gaussians_per_token = int(base_model.activation_head.num_gaussians_per_token)

    ws = opt.workspace
    os.makedirs(ws, exist_ok=True)
    if accelerator.is_main_process:
        config_save_path = os.path.join(ws, "config.yaml")
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(tyro.extras.to_yaml(opt))
        accelerator.print(f"[INFO] Config saved to {config_save_path=}")
    if opt.eval_n_media_dumps > 0:
        for sub in ("images", "videos", "gaussians", "depths"):
            os.makedirs(os.path.join(ws, sub), exist_ok=True)

    metrics_calc = MetricsCalculator(device=accelerator.device)
    metrics_tracker = MetricsTracker()
    worker_rank = accelerator.process_index

    pbar = tqdm(test_dataloader, disable=not accelerator.is_main_process)
    with torch.no_grad(), pbar:
        for i, data in enumerate(pbar):
            _cuda_sync()
            t0 = time.perf_counter()
            if is_depthsplat:
                batch, pred_images, depth_maps = _run_depthsplat_inference(
                    opt, depthsplat_cfg, base_model, data
                )
                results = {"images_pred": pred_images}
            else:
                results = model(data)
                depth_maps = results.get("depths_pred")
            _cuda_sync()
            inference_time = time.perf_counter() - t0

            if is_depthsplat:
                input_images = batch["context"]["image"]
                pred_images = results["images_pred"]
                gt_images = batch["target"]["image"]
                gaussians = None
                masks = None
                has_mask = None
                mask_kw = {}
            else:
                input_images = data["images_input"]
                pred_images = results["images_pred"]
                gt_images = data["images_output"]
                gaussians = results["gaussians"]
                masks = data["masks_output"]
                has_mask = data["has_mask"]
                mask_kw = {"mask": masks} if has_mask.any() else {}

            metrics = metrics_calc.calculate_all_metrics(
                pred_images, gt_images, reduction="mean", **mask_kw
            )
            metrics["inference_time"] = inference_time
            compactness_centroid = None
            if not is_depthsplat:
                compactness_centroid = _compute_compactness_centroid_per_sample(
                    gaussians, num_gaussians_per_token
                )

            gathered = accelerator.gather(
                {k: torch.tensor(v, device=accelerator.device) for k, v in metrics.items()}
            )
            gathered_compactness = (
                accelerator.gather(compactness_centroid) if compactness_centroid is not None else None
            )
            if accelerator.is_main_process:
                metrics_tracker.update({k: v.mean().item() for k, v in gathered.items()})
                if gathered_compactness is not None:
                    for value in gathered_compactness.detach().cpu().tolist():
                        metrics_tracker.update(compactness_centroid=value)

            postfix: dict[str, str] = {}
            if torch.cuda.is_available():
                mem_free, mem_total = torch.cuda.mem_get_info()
                postfix["mem_used"] = f"{(mem_total - mem_free) / 1024**3:.1f}GB"
            for k, st in metrics_tracker.get_stats().items():
                if st:
                    postfix[k] = f"{st['mean']:.4f} ± {st['std']:.4f}"
            pbar.set_postfix(**postfix)

            if i < opt.eval_n_media_dumps:
                pred_vis = pred_images.detach().clone()
                gt_vis = gt_images.detach().clone()
                input_vis = input_images.detach().clone()
                if has_mask is not None and has_mask.any():
                    pred_vis = _blend_mask_overlay(pred_vis, masks)
                    gt_vis = _blend_mask_overlay(gt_vis, masks)

                for b_idx in range(pred_images.shape[0]):
                    tag = f"{b_idx}-{worker_rank}-{i}"
                    save_image_grid_square(
                        input_vis[b_idx], os.path.join(ws, "images", f"input_{tag}.png")
                    )
                    save_image_grid_square(pred_vis[b_idx], os.path.join(ws, "images", f"pred_{tag}.png"))
                    save_image_grid_square(gt_vis[b_idx], os.path.join(ws, "images", f"gt_{tag}.png"))

                    if depth_maps is not None:
                        depth_vis = visualize_depth(depth_maps[b_idx])
                        save_image_grid_square(
                            depth_vis, os.path.join(ws, "depths", f"depth_{tag}.png")
                        )
                        depth_u8 = _views_chw_to_uint8_nhwc(depth_vis)
                        imageio.mimwrite(
                            os.path.join(ws, "videos", f"depth_{tag}.mp4"), depth_u8, fps=30
                        )

                    pred_u8 = _views_chw_to_uint8_nhwc(pred_vis[b_idx])
                    input_u8 = _views_chw_to_uint8_nhwc(input_vis[b_idx])
                    gt_u8 = _views_chw_to_uint8_nhwc(gt_vis[b_idx])
                    combined = np.concatenate([gt_u8, pred_u8], axis=-2)

                    for name, arr in (
                        ("comparison", combined),
                        ("pred", pred_u8),
                        ("gt", gt_u8),
                        ("input", input_u8),
                    ):
                        imageio.mimwrite(os.path.join(ws, "videos", f"{name}_{tag}.mp4"), arr, fps=30)

                    if not is_depthsplat:
                        base_model.gs.save_ply(
                            gaussians[b_idx : b_idx + 1],
                            os.path.join(ws, "gaussians", f"gaussians_{tag}.ply"),
                        )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if accelerator.is_main_process:
        metrics_tracker.print_summary(title="Final Evaluation Metrics")
        metrics_tracker.save_to_file(os.path.join(ws, "metrics.txt"))

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
