"""Geometry utilities used by anchor-aware decoder attention."""

import math

import torch


def point_to_plucker_ray_bias(
    anchor_xyz: torch.Tensor,
    plucker_rays: torch.Tensor,
    sigma: torch.Tensor | float,
    plucker_order: str = "md",
    moment_convention: str = "o_cross_d",
    clamp_min: float = -20.0,
) -> torch.Tensor:
    if plucker_order == "dm":
        d = plucker_rays[..., :3]
        m = plucker_rays[..., 3:]
    elif plucker_order == "md":
        m = plucker_rays[..., :3]
        d = plucker_rays[..., 3:]
    else:
        raise ValueError(f"Unknown plucker_order: {plucker_order}")

    d_norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    d = d / d_norm
    m = m / d_norm
    A = anchor_xyz[:, :, None, :]
    d = d[:, None, :, :]
    m = m[:, None, :, :]

    if moment_convention == "o_cross_d":
        residual = torch.cross(A, d, dim=-1) - m
    elif moment_convention == "d_cross_o":
        residual = torch.cross(d, A, dim=-1) - m
    else:
        raise ValueError(f"Unknown moment_convention: {moment_convention}")

    dist2 = residual.pow(2).sum(dim=-1)
    sigma = torch.as_tensor(sigma, dtype=anchor_xyz.dtype, device=anchor_xyz.device)
    sigma2 = sigma.pow(2).clamp_min(1e-3)
    while sigma2.ndim < dist2.ndim:
        sigma2 = sigma2.unsqueeze(-1)
    return (-dist2 / (2.0 * sigma2)).clamp(min=clamp_min, max=0.0)


def gen_sineembed_for_3d_anchor(
    gs_anchor_normalized: torch.Tensor,
    num_feats: int = 128,
    temperature: float = 10000.0,
    scale: float = 2 * math.pi,
    eps: float = 1e-6,
) -> torch.Tensor:
    if gs_anchor_normalized.dim() != 3 or gs_anchor_normalized.size(-1) != 3:
        raise ValueError(
            f"Expected gs_anchor_normalized with shape (B, N_q, 3), "
            f"but got {tuple(gs_anchor_normalized.shape)}"
        )
    if num_feats % 2 != 0:
        raise ValueError(f"num_feats must be even, but got {num_feats}")

    pos = (gs_anchor_normalized + 1.0).mul(0.5).clamp(0.0, 1.0).mul(scale)
    dim_t = torch.arange(num_feats, dtype=pos.dtype, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_feats)
    pos = pos[..., None] / (dim_t + eps)
    pos_embed = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)
    return pos_embed.flatten(-2)


def sample_sparse_indices(
    proposal_probs: torch.Tensor,
    sampling_k: int,
    mode: str = "multinomial",
) -> torch.Tensor:
    if mode == "multinomial":
        batch_size, num_queries, _ = proposal_probs.shape
        return torch.multinomial(
            proposal_probs.reshape(batch_size * num_queries, -1),
            num_samples=sampling_k,
            replacement=False,
        ).reshape(batch_size, num_queries, sampling_k)
    if mode == "topk":
        return torch.topk(proposal_probs, k=sampling_k, dim=-1).indices
    raise ValueError(f"Unknown geo_sparse_sampling_mode: {mode}")
