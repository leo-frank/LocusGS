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

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from locusgs.rendering.fused_ssim import fused_ssim
from locusgs.models.input_types import ModelSupervision
from locusgs.options import Options


def compute_anchor_visibility_loss(
    anchor_xyz: torch.Tensor,
    cam_view: torch.Tensor,
    intrinsics: torch.Tensor,
    img_size: tuple[int, int] | list[int],
    visibility_distance_threshold: float,
) -> torch.Tensor:
    """Visibility loss for anchors by projecting to each supervision view."""
    H, W = int(img_size[0]), int(img_size[1])
    B, N, _ = anchor_xyz.shape
    _, V, _, _ = cam_view.shape

    ones = torch.ones(B, N, 1, dtype=anchor_xyz.dtype, device=anchor_xyz.device)
    anchor_h = torch.cat([anchor_xyz, ones], dim=-1)  # [B, N, 4]

    # Provider stores cam_view = inverse(c2w).T, so world->camera is x_world_h @ cam_view.
    cam_view_expand = cam_view.unsqueeze(2)  # [B, V, 1, 4, 4]
    anchor_expand = anchor_h.unsqueeze(1).unsqueeze(-2)  # [B, 1, N, 1, 4]
    cam_xyz_h = torch.matmul(anchor_expand, cam_view_expand).squeeze(-2)  # [B, V, N, 4]
    x, y, z = cam_xyz_h[..., 0], cam_xyz_h[..., 1], cam_xyz_h[..., 2]

    fx, fy, cx, cy = [intrinsics[..., i].unsqueeze(-1) for i in range(4)]  # [B, V, 1]
    z_safe = torch.where(z.abs() < 1e-3, torch.full_like(z, 1e-3), z)
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    uv = torch.stack([u, v], dim=-1)  # [B, V, N, 2]
    uv = torch.nan_to_num(uv, nan=0.0, posinf=1e6, neginf=-1e6)

    uv_norm = (uv / torch.tensor([W, H], device=uv.device, dtype=uv.dtype)) * 2 - 1
    out_of_bounds = F.relu(torch.abs(uv_norm) - 1.0).sum(-1)  # [B, V, N]
    vis_loss = out_of_bounds.min(dim=1).values  # [B, N]
    vis_loss = torch.nan_to_num(vis_loss, nan=0.0, posinf=1e6, neginf=1e6)

    if visibility_distance_threshold > 0:
        vis_loss = vis_loss.clamp(max=visibility_distance_threshold)
    return vis_loss.mean()


def compute_loss_from_renders(
    opt: Options,
    img_size: tuple[int, int] | list[int],
    pred_images: torch.Tensor,
    pred_masks: torch.Tensor,
    means2d_pred: torch.Tensor,
    gt_images: torch.Tensor,
    gt_masks: torch.Tensor,
    has_mask: torch.Tensor,
    _means3d_pred: torch.Tensor,
    rays_os: torch.Tensor,
    rays_ds: torch.Tensor,
) -> dict:
    """Per-scene loss from rendered predictions (intended for use inside torch.vmap)."""
    H, W = int(img_size[0]), int(img_size[1])
    results: dict = {}
    assert has_mask.shape == (pred_images.shape[0],), "has_mask must have the same batch size as pred_images"

    bg_color = torch.ones(3, dtype=pred_images.dtype, device=pred_images.device)
    gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

    loss_mse = F.mse_loss(pred_images, gt_images)
    loss = loss_mse
    results["loss_mse"] = loss_mse

    if opt.lambda_mask > 0:
        loss_mse_mask = F.mse_loss(pred_masks, gt_masks, reduction="none").mean(dim=(1, 2, 3, 4))
        loss_mse_mask = torch.where(has_mask, loss_mse_mask, torch.zeros_like(loss_mse_mask)).sum()
        results["loss_mask"] = loss_mse_mask
        loss = loss + opt.lambda_mask * loss_mse_mask

    if opt.lambda_ssim > 0:
        ssim = fused_ssim(
            pred_images.view(-1, 3, H, W),
            gt_images.view(-1, 3, H, W),
        )
        loss_ssim = (1 - ssim) / 2
        loss = loss + opt.lambda_ssim * loss_ssim
        results["loss_ssim"] = loss_ssim

    if opt.lambda_visibility > 0:
        uv = torch.nan_to_num(means2d_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        uv_norm = (uv / torch.tensor([W, H], device=uv.device)) * 2 - 1
        out_of_bounds = F.relu(torch.abs(uv_norm) - 1.0)
        out_of_bounds = out_of_bounds.sum(-1)
        vis_loss = out_of_bounds.min(dim=1).values
        if opt.visibility_distance_threshold > 0:
            vis_loss = vis_loss.clamp(max=opt.visibility_distance_threshold)
        vis_loss = vis_loss.mean()
        loss = loss + vis_loss * opt.lambda_visibility
        results["loss_visibility"] = vis_loss

    results["loss"] = loss

    with torch.no_grad():
        psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2, dim=(-1, -2, -3)))
        results["psnr"] = psnr.mean()

    return results


def compute_tokengs_loss(
    opt: Options,
    img_size: tuple[int, int] | list[int],
    render_results: dict,
    supervision: ModelSupervision,
    gaussians_xyz: torch.Tensor,
    lpips_loss: Optional[nn.Module],
) -> dict:
    """Aggregate training loss from renderer outputs and supervision (includes LPIPS outside vmap)."""
    pred_images = render_results["images_pred"]
    pred_masks = render_results["alphas_pred"]
    means2d_pred = render_results["means2d_pred"]
    depths_pred = render_results["depths_pred"]
    gt_images = supervision.images_output

    per_scene = partial(compute_loss_from_renders, opt, img_size)

    if supervision.rays_os is None:
        in_dims = (0,) * 7 + ((None,) * 2)
        rays_os = supervision.rays_os
        rays_ds = supervision.rays_ds
    else:
        in_dims = (0,) * 9
        rays_os = supervision.rays_os.unsqueeze(1)
        rays_ds = supervision.rays_ds.unsqueeze(1)

    results_per_scene = torch.vmap(per_scene, in_dims=in_dims)(
        pred_images.unsqueeze(1),
        pred_masks.unsqueeze(1),
        means2d_pred.unsqueeze(1),
        supervision.images_output.unsqueeze(1),
        supervision.masks_output.unsqueeze(1),
        supervision.has_mask.unsqueeze(1),
        gaussians_xyz.unsqueeze(1),
        rays_os,
        rays_ds,
    )

    if "loss_mask" in results_per_scene:
        results_per_scene["loss_mask"] = results_per_scene["loss_mask"] / (supervision.has_mask.float().sum() + 1e-6)

    result_means = {k: v.mean() for k, v in results_per_scene.items()}
    results_per_scene = {f"{k}_per_scene": v for k, v in results_per_scene.items()}

    if opt.lambda_lpips > 0 and lpips_loss is not None:
        H, W = int(img_size[0]), int(img_size[1])
        gt_flat = gt_images.view(-1, 3, H, W)
        pred_flat = pred_images.view(-1, 3, H, W)
        loss_lpips = lpips_loss(gt_flat, pred_flat, normalize=True).mean()
        result_means["loss_lpips"] = loss_lpips
        result_means["loss"] = result_means["loss"] + opt.lambda_lpips * loss_lpips

    return {
        **result_means,
        **results_per_scene,
        "images_pred": pred_images,
        "alphas_pred": pred_masks,
        "images_output": gt_images,
        "depths_pred": depths_pred,
    }
