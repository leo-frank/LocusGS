import math
import torch
from typing import Any, Optional, cast
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attention import Attention, LayerScale, Mlp
from .geometry import point_to_plucker_ray_bias, gen_sineembed_for_3d_anchor, sample_sparse_indices


class MultiViewDeformableAttention(nn.Module):
    """FlexSplat-style deformable attention over encoder feature maps.

    Each 3D anchor is projected into every input view and sampled at a small
    set of learned offsets.  The implementation intentionally uses
    ``grid_sample`` so it works without MMCV or a custom CUDA extension.
    """

    def __init__(self, dim: int, num_heads: int, num_points: int, max_views: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        if num_points <= 0 or max_views <= 0:
            raise ValueError("num_points and max_views must be positive")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_points = num_points
        self.max_views = max_views
        self.sampling_offsets = nn.Linear(dim, num_heads * max_views * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * max_views * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack((angles.cos(), angles.sin()), dim=-1)
        grid = grid / grid.abs().amax(dim=-1, keepdim=True)
        grid = grid.view(self.num_heads, 1, 1, 2).repeat(1, self.max_views, self.num_points, 1)
        for point_idx in range(self.num_points):
            grid[:, :, point_idx] *= point_idx + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    @staticmethod
    def _project_anchors(
        anchor_xyz: torch.Tensor,
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _ = anchor_xyz.shape
        anchor_xyz = anchor_xyz.float()
        intrinsics = intrinsics.float()
        extrinsics = torch.linalg.inv(cam_to_world.float())
        points_h = torch.cat((anchor_xyz, anchor_xyz.new_ones(batch_size, anchor_xyz.shape[1], 1)), dim=-1)
        points_cam = torch.einsum("bvij,bnj->bvni", extrinsics, points_h)[..., :3]
        x, y, z = points_cam.unbind(dim=-1)
        z_safe = z.clamp_min(1e-6)
        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
        u = fx.unsqueeze(-1) * x / z_safe + cx.unsqueeze(-1)
        v = fy.unsqueeze(-1) * y / z_safe + cy.unsqueeze(-1)
        reference = torch.stack((u / image_width, v / image_height), dim=-1)
        valid = (z > 1e-6) & (reference[..., 0] >= 0.0) & (reference[..., 0] <= 1.0)
        valid = valid & (reference[..., 1] >= 0.0) & (reference[..., 1] <= 1.0)
        return reference, valid

    def forward(
        self,
        query: torch.Tensor,
        image_features: torch.Tensor,
        anchor_xyz: torch.Tensor,
        intrinsics: torch.Tensor,
        cam_to_world: torch.Tensor,
        image_height: int,
        image_width: int,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        batch_size, num_queries, _ = query.shape
        _, num_views, feature_height, feature_width, _ = image_features.shape
        if num_views > self.max_views:
            raise ValueError(f"got {num_views} views but module supports at most {self.max_views}")
        if intrinsics.shape[:2] != (batch_size, num_views) or cam_to_world.shape[:2] != (batch_size, num_views):
            raise ValueError("deformable attention camera tensors do not match image features")

        reference, valid_views = self._project_anchors(
            anchor_xyz, intrinsics, cam_to_world, image_height, image_width
        )
        offsets = self.sampling_offsets(query).view(
            batch_size, num_queries, self.num_heads, self.max_views, self.num_points, 2
        )[:, :, :, :num_views]
        weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads, self.max_views, self.num_points
        )[:, :, :, :num_views]
        valid_query_views = valid_views.permute(0, 2, 1)[:, :, None, :, None]
        weights = weights.masked_fill(~valid_query_views, float("-inf"))
        weights = weights.reshape(batch_size, num_queries, self.num_heads, num_views * self.num_points)
        weights = torch.nan_to_num(weights.softmax(dim=-1), nan=0.0)

        grid_scale = query.new_tensor((1.0 / feature_width, 1.0 / feature_height))
        locations = reference.permute(0, 2, 1, 3)[:, :, None, :, None] + offsets * grid_scale
        grid = (2.0 * locations - 1.0).permute(0, 3, 2, 1, 4, 5)
        grid = grid.reshape(batch_size * num_views * self.num_heads, num_queries, self.num_points, 2)

        values = self.value_proj(image_features).permute(0, 1, 4, 2, 3)
        grid = grid.to(dtype=values.dtype)
        values = values.reshape(batch_size * num_views, self.num_heads, self.head_dim, feature_height, feature_width)
        values = values.reshape(batch_size * num_views * self.num_heads, self.head_dim, feature_height, feature_width)
        sampled = F.grid_sample(values, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(batch_size, num_views, self.num_heads, self.head_dim, num_queries, self.num_points)
        sampled = sampled.permute(0, 4, 2, 1, 5, 3).reshape(
            batch_size, num_queries, self.num_heads, num_views * self.num_points, self.head_dim
        )
        output = (sampled * weights.unsqueeze(-1)).sum(dim=3).reshape(batch_size, num_queries, self.dim)
        output = self.output_proj(output)
        if return_attn:
            return output, {
                "attn": weights,
                "sampling_locations": locations,
                "valid_views": valid_views,
                "mode": "multiview_deformable_grid_sample",
            }
        return output

class DecoderBlock(nn.Module):
    """
    Decoder block for encoder-decoder architecture.
    Works like a transformer decoder layer except the keys/values are provided by the encoder (already normalized).
    """

    class SelfAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            qk_norm: bool,
            flex_attn_score_mod=None,
        ):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.gs_self_attn = Attention(
                dim,
                num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                fused_attn=True,
                flex_attn_score_mod=flex_attn_score_mod,
            )

        def forward(
            self,
            gs_tokens: torch.Tensor,
            return_attn: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None | str]]:
            queries_normed = self.norm(gs_tokens)
            if not return_attn:
                return self.gs_self_attn(queries_normed)

            qkv = (
                self.gs_self_attn.qkv(queries_normed)
                .reshape(queries_normed.shape[0], queries_normed.shape[1], 3, self.gs_self_attn.num_heads, self.gs_self_attn.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q, k = self.gs_self_attn.q_norm(q), self.gs_self_attn.k_norm(k)
            q = q * self.gs_self_attn.scale
            logits = torch.einsum("bhqd,bhkd->bhqk", q, k)
            attn = logits.softmax(dim=-1)
            x = torch.einsum("bhqk,bhkd->bhqd", attn, v)
            x = x.transpose(1, 2).reshape(queries_normed.shape[0], queries_normed.shape[1], self.gs_self_attn.num_heads * self.gs_self_attn.head_dim)
            x = self.gs_self_attn.proj(x)
            x = self.gs_self_attn.proj_drop(x)
            return x, {
                "attn": attn,
                "logits": logits,
                "self_attn": attn,
                "mode": "self_attn",
            }

    class CrossAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            q_norm: bool,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.gs_token_norm = nn.LayerNorm(dim)
            self.q_norm = nn.LayerNorm(dim // num_heads) if q_norm else nn.Identity()
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.out_proj = nn.Linear(dim, dim)

        def forward(
            self,
            gs_tokens: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            return_attn: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None | str]]:
            gs_tokens_normed = self.gs_token_norm(gs_tokens)
            q = rearrange(self.q_proj(gs_tokens_normed), "b n (h d) -> b h n d", h=self.num_heads)
            q = self.q_norm(q)
            attn: torch.Tensor | None = None
            logits: torch.Tensor | None = None
            logits_c: torch.Tensor | None = None

            if return_attn:
                logits_c = torch.einsum("bhqd,bhkd->bhqk", q, keys)
                logits = logits_c / (q.shape[-1] ** 0.5)
                assert logits is not None
                attn = logits.softmax(dim=-1)
                cross_attn_output = torch.einsum("bhqk,bhkd->bhqd", attn, values)
            else:
                cross_attn_output = F.scaled_dot_product_attention(q, keys, values)
            cross_attn_output = rearrange(cross_attn_output, "b h n d -> b n (h d)")
            cross_attn_output = self.out_proj(cross_attn_output)
            if return_attn:
                assert attn is not None and logits is not None and logits_c is not None
                return cross_attn_output, {
                    "attn": attn,
                    "attn_c": attn,
                    "attn_p": None,
                    "logits": logits,
                    "logits_c": logits_c,
                    "logits_p": None,
                    "gamma_p": None,
                    "gamma_g": None,
                    "geo_bias": None,
                    "sampled_idx": None,
                    "mode": "handcraft_tokengs_full",
                }
            return cross_attn_output

    class MlpBlock(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float, ffn_bias: bool):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, bias=ffn_bias)

        def forward(self, gs_tokens: torch.Tensor) -> torch.Tensor:
            return self.mlp(self.norm(gs_tokens))

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ffn_bias: bool,
        qk_norm: bool,
        init_values: float | None = None,
        attn_score_mod=None,
    ):
        super().__init__()

        def make_scale() -> nn.Module:
            return LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.gs_self_attn = DecoderBlock.SelfAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            qk_norm=qk_norm,
            flex_attn_score_mod=attn_score_mod,
        )
        self.gs_self_attn_scale = make_scale()

        self.gs_cross_attn = DecoderBlock.CrossAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            q_norm=qk_norm,
        )
        self.gs_cross_attn_scale = make_scale()

        self.mlp = DecoderBlock.MlpBlock(dim, mlp_ratio, ffn_bias)
        self.mlp_scale = make_scale()

    def forward(
        self,
        gs_tokens: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None | str]]:
        cross_attn_out = self.gs_cross_attn(
            gs_tokens,
            keys,
            values,
            return_attn=return_attn,
        )
        attn_info: dict[str, torch.Tensor | None | str] = {}
        self_attn_info: dict[str, torch.Tensor | None | str] = {}
        if return_attn:
            cross_attn_out, attn_info = cross_attn_out
        gs_tokens = gs_tokens + self.gs_cross_attn_scale(cross_attn_out)
        self_attn_out = self.gs_self_attn(gs_tokens, return_attn=return_attn)
        if return_attn:
            self_attn_out, self_attn_info = self_attn_out
            attn_info = {
                "attn": attn_info.get("attn"),
                "attn_c": attn_info.get("attn_c"),
                "attn_p": attn_info.get("attn_p"),
                "cross_attn": attn_info.get("attn"),
                "cross_attn_c": attn_info.get("attn_c"),
                "cross_attn_p": attn_info.get("attn_p"),
                "cross_logits": attn_info.get("logits"),
                "cross_logits_c": attn_info.get("logits_c"),
                "cross_logits_p": attn_info.get("logits_p"),
                "cross_gamma_p": attn_info.get("gamma_p"),
                "cross_gamma_g": attn_info.get("gamma_g"),
                "cross_geo_bias": attn_info.get("geo_bias"),
                "cross_sampled_idx": attn_info.get("sampled_idx"),
                "cross_mode": attn_info.get("mode"),
                "self_attn": self_attn_info.get("attn"),
                "self_logits": self_attn_info.get("logits"),
                "self_mode": self_attn_info.get("mode"),
            }
        gs_tokens = gs_tokens + self.gs_self_attn_scale(self_attn_out)
        gs_tokens = gs_tokens + self.mlp_scale(self.mlp(gs_tokens))
        if return_attn:
            return gs_tokens, attn_info
        return gs_tokens




class DecoderBlockWithAnchor(nn.Module):
    """
    Decoder block for encoder-decoder architecture.
    Works like a transformer decoder layer except the keys/values are provided by the encoder (already normalized).
    """

    class SelfAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            qk_norm: bool,
            flex_attn_score_mod=None,
        ):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.gs_self_attn = Attention(
                dim,
                num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                fused_attn=True,
                flex_attn_score_mod=flex_attn_score_mod,
            )

        def forward(
            self,
            gs_tokens: torch.Tensor,
            self_attn_anchor_positional_feat: Optional[torch.Tensor],
            return_attn: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None | str]]:
            if self_attn_anchor_positional_feat is not None:
                gs_tokens = self_attn_anchor_positional_feat + gs_tokens
            queries_normed = self.norm(gs_tokens)
            if not return_attn:
                return self.gs_self_attn(queries_normed)

            qkv = (
                self.gs_self_attn.qkv(queries_normed)
                .reshape(queries_normed.shape[0], queries_normed.shape[1], 3, self.gs_self_attn.num_heads, self.gs_self_attn.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q, k = self.gs_self_attn.q_norm(q), self.gs_self_attn.k_norm(k)
            q = q * self.gs_self_attn.scale
            logits = torch.einsum("bhqd,bhkd->bhqk", q, k)
            attn = logits.softmax(dim=-1)
            x = torch.einsum("bhqk,bhkd->bhqd", attn, v)
            x = x.transpose(1, 2).reshape(queries_normed.shape[0], queries_normed.shape[1], self.gs_self_attn.num_heads * self.gs_self_attn.head_dim)
            x = self.gs_self_attn.proj(x)
            x = self.gs_self_attn.proj_drop(x)
            return x, {
                "attn": attn,
                "logits": logits,
                "self_attn": attn,
                "mode": "self_attn",
            }

    class CrossAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            q_norm: bool,
            cross_attn_variant: str = "learned_positional",
            geo_sparse_sampling_k: int = 0,
            geo_sparse_sampling_tau_init: float = 1.0,
            geo_sparse_sampling_mode: str = "multinomial",
            use_dense_sparse_attn_mask: bool = False,
            deformable_num_points: int = 4,
            deformable_max_views: int = 16,
            pos_scale_init: float = 0.1,  # TODO: modify it to learnable
            geo_scale_init: float = -2.25,  # TODO: modify it to learnable
            geo_sigma_fixed: float = 0.1,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.dim = dim

            # content branch
            self.gs_token_norm = nn.LayerNorm(dim)
            self.q_content_proj = nn.Linear(dim, dim, bias=qkv_bias)

            # query position branch
            self.anchor_norm = nn.LayerNorm(dim)
            self.q_pos_proj = None

            # key position branch
            self.plucker_norm = nn.LayerNorm(dim)
            self.k_pos_proj = None

            # Note: `keys` are already projected keys
            # Else using following:
            # self.key_norm = nn.LayerNorm(dim)
            # self.k_content_proj = nn.Linear(dim, dim, bias=qkv_bias)

            self.q_norm = nn.LayerNorm(dim // num_heads) if q_norm else nn.Identity()
            self.q_pos_norm = nn.LayerNorm(dim // num_heads) if q_norm else nn.Identity()
            self.k_pos_norm = nn.LayerNorm(dim // num_heads) if q_norm else nn.Identity()

            self.cross_attn_variant = cross_attn_variant
            self.geo_sparse_sampling_k = geo_sparse_sampling_k
            self.geo_sparse_sampling_mode = geo_sparse_sampling_mode
            self.use_dense_sparse_attn_mask = use_dense_sparse_attn_mask
            self.deformable_attn = None
            if self.cross_attn_variant == "geometric_positional":
                self.geo_scale = nn.Parameter(torch.tensor(geo_scale_init))
                self.tau_sample = nn.Parameter(torch.tensor(float(geo_sparse_sampling_tau_init)))
                # Fixed sigma for point-to-ray bias (non-learnable).
                self.register_buffer("geo_sigma", torch.tensor(float(geo_sigma_fixed)), persistent=False)
                self.pos_scale = None
            elif self.cross_attn_variant == "learned_positional":
                # learnable gate for positional branch
                self.geo_scale = None
                self.pos_scale = nn.Parameter(torch.tensor(pos_scale_init))
                self.q_pos_proj = nn.Linear(dim, dim, bias=qkv_bias)
                self.k_pos_proj = nn.Linear(dim, dim, bias=qkv_bias)
                self.tau_sample = None
            elif self.cross_attn_variant == "content_only":
                self.geo_scale = None
                self.tau_sample = None
                self.pos_scale = None
            elif self.cross_attn_variant == "deformable":
                self.geo_scale = None
                self.tau_sample = None
                self.pos_scale = None
                self.deformable_attn = MultiViewDeformableAttention(
                    dim, num_heads, deformable_num_points, deformable_max_views
                )
            else:
                raise ValueError(f"Unknown cross_attn_variant: {self.cross_attn_variant}")

            self.out_proj = nn.Linear(dim, dim)

        def _forward_content_only(
            self,
            q_c: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            return_attn: bool,
        ) -> tuple[torch.Tensor, dict[str, Any] | None]:
            """Run the ablation path using only learned content similarity.

            No anchor coordinates, Plucker positional features, or geometric
            attention bias are used in this mode.
            """
            attn_info = None
            out = F.scaled_dot_product_attention(q_c, keys, values)
            if return_attn:
                attn_info = {
                    "attn": None,
                    "attn_c": None,
                    "attn_p": None,
                    "logits": None,
                    "logits_c": None,
                    "logits_p": None,
                    "gamma_p": None,
                    "gamma_g": None,
                    "geo_bias": None,
                    "sampled_idx": None,
                    "mode": "content_only_sdpa",
                }
            return out, attn_info

        def _forward_geometric_positional(
            self,
            q_c: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            anchor_xyz: torch.Tensor,
            anchor_radius: Optional[torch.Tensor],
            plucker_rays_raw: torch.Tensor,
            handcraft: bool,
            return_attn: bool,
        ) -> tuple[torch.Tensor, dict[str, Any] | None]:
            """Run explicit point-to-ray attention.

            The geometric distance is converted to an additive attention bias.
            When ``geo_sparse_sampling_k`` is positive, the bias first proposes
            a subset of rays; otherwise all rays remain in the attention set.
            ``handcraft`` only changes how logits are materialized for
            diagnostics, not the mathematical attention definition.
            """
            attn_info = None
            B, H, Nq, _ = q_c.shape
            k_c = keys
            v = values
            sigma = self.geo_sigma if anchor_radius is None else self.geo_sigma * anchor_radius
            assert self.geo_scale is not None
            gamma_g = F.softplus(self.geo_scale)
            geo_bias = point_to_plucker_ray_bias(
                anchor_xyz=anchor_xyz,          # (B, N_q, 3)
                plucker_rays=plucker_rays_raw,  # (B, N, 6)
                sigma=sigma,
                plucker_order="md",
                moment_convention="o_cross_d",
            )  # (B, N_q, N)
            attn_mask = gamma_g * geo_bias[:, None, :, :]  # (B, 1, Nq, N), broadcast over heads

            sampling_k = min(self.geo_sparse_sampling_k, k_c.shape[-2])
            use_sparse_sampling = sampling_k > 0

            if use_sparse_sampling:
                # Treat geometric bias as a proposal distribution and sample K keys per query.
                # NOTE: `tau_sample` is used only to form `proposal_probs` for DISCRETE index
                # selection (`multinomial` / `topk`). This index path is non-differentiable, so
                # `tau_sample` does not receive meaningful gradients from the training loss in
                # the current implementation.
                assert self.tau_sample is not None
                tau_sample = F.softplus(self.tau_sample).clamp_min(1e-6)
                proposal_probs = torch.softmax(geo_bias / tau_sample, dim=-1)  # (B, Nq, N)
                sampled_idx = sample_sparse_indices(
                    proposal_probs=proposal_probs,
                    sampling_k=sampling_k,
                    mode=self.geo_sparse_sampling_mode,
                )  # (B, Nq, K)

                gathered_geo_base = torch.gather(gamma_g * geo_bias, dim=-1, index=sampled_idx)  # (B, Nq, K)

                if self.use_dense_sparse_attn_mask:
                    dense_sparse_mask = torch.full(
                        (B, 1, Nq, k_c.shape[-2]),
                        float("-inf"),
                        device=q_c.device,
                        dtype=q_c.dtype,
                    )
                    dense_sparse_mask.scatter_(
                        dim=-1,
                        index=sampled_idx[:, None, :, :].expand(B, 1, Nq, sampling_k),
                        src=gathered_geo_base[:, None, :, :],
                    )
                    if not handcraft:
                        out = F.scaled_dot_product_attention(q_c, k_c, v, attn_mask=dense_sparse_mask)
                        attn = None
                        if return_attn:
                            logits_c_dense = torch.einsum("bhqd,bhkd->bhqk", q_c, k_c) / math.sqrt(q_c.shape[-1])
                            logits_sparse_dense = logits_c_dense + dense_sparse_mask
                            attn_info = {
                                "attn": logits_sparse_dense.softmax(dim=-1),
                                "attn_c": logits_c_dense.softmax(dim=-1),
                                "attn_p": dense_sparse_mask.softmax(dim=-1),
                                "logits": logits_sparse_dense,
                                "logits_c": logits_c_dense,
                                "logits_p": dense_sparse_mask,
                                "gamma_p": None,
                                "gamma_g": gamma_g,
                                "geo_bias": geo_bias,
                                "sampled_idx": sampled_idx,
                                "mode": "dense_mask_sparse_geo_sdpa",
                            }
                    else:
                        logits_c_dense = torch.einsum("bhqd,bhkd->bhqk", q_c, k_c)
                        logits_sparse_dense = logits_c_dense / math.sqrt(q_c.shape[-1]) + dense_sparse_mask
                        attn_dense = logits_sparse_dense.softmax(dim=-1)
                        out = torch.einsum("bhqk,bhkd->bhqd", attn_dense, v)
                        attn = attn_dense
                        if return_attn:
                            attn_info = {
                                "attn": attn_dense,
                                "attn_c": (logits_c_dense / math.sqrt(q_c.shape[-1])).softmax(dim=-1),
                                "attn_p": dense_sparse_mask.softmax(dim=-1),
                                "logits": logits_sparse_dense,
                                "logits_c": logits_c_dense / math.sqrt(q_c.shape[-1]),
                                "logits_p": dense_sparse_mask,
                                "gamma_p": None,
                                "gamma_g": gamma_g,
                                "geo_bias": geo_bias,
                                "sampled_idx": sampled_idx,
                                "mode": "dense_mask_sparse_geo_handcraft",
                            }
                else:
                    bh = B * H
                    n_tokens = k_c.shape[-2]
                    dh = k_c.shape[-1]
                    sampled_idx_bh = sampled_idx[:, None, :, :].expand(B, H, Nq, sampling_k).reshape(bh, Nq, sampling_k)

                    if not handcraft:
                        gathered_k = None
                        gathered_v = None
                        gathered_geo = None
                        batch_idx = torch.arange(bh, device=q_c.device)[:, None, None]
                        k_flat = k_c.reshape(bh, n_tokens, dh)
                        v_flat = v.reshape(bh, n_tokens, dh)
                        gathered_k = k_flat[batch_idx, sampled_idx_bh].reshape(B, H, Nq, sampling_k, dh)
                        gathered_v = v_flat[batch_idx, sampled_idx_bh].reshape(B, H, Nq, sampling_k, dh)
                        gathered_geo = gathered_geo_base[:, None, :, :].expand(B, H, Nq, sampling_k)  # (B, H, Nq, K)
                        q_sdpa = q_c.reshape(B * H * Nq, 1, q_c.shape[-1])
                        k_sdpa = gathered_k.reshape(B * H * Nq, sampling_k, gathered_k.shape[-1])
                        v_sdpa = gathered_v.reshape(B * H * Nq, sampling_k, gathered_v.shape[-1])
                        mask_sdpa = gathered_geo.reshape(B * H * Nq, 1, sampling_k)
                        out_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask_sdpa)
                        out = out_sdpa.reshape(B, H, Nq, q_c.shape[-1])
                        attn = None

                        if return_attn:
                            if gathered_k is None:
                                batch_idx = torch.arange(bh, device=q_c.device)[:, None, None]
                                k_flat = k_c.reshape(bh, n_tokens, dh)
                                v_flat = v.reshape(bh, n_tokens, dh)
                                gathered_k = k_flat[batch_idx, sampled_idx_bh].reshape(B, H, Nq, sampling_k, dh)
                                gathered_v = v_flat[batch_idx, sampled_idx_bh].reshape(B, H, Nq, sampling_k, dh)
                                gathered_geo = gathered_geo_base[:, None, :, :].expand(B, H, Nq, sampling_k)
                            logits_c_sparse = (q_c[:, :, :, None, :] * gathered_k).sum(dim=-1) / math.sqrt(q_c.shape[-1])
                            logits_sparse = logits_c_sparse + gathered_geo
                            attn_sparse = logits_sparse.softmax(dim=-1)
                            attn_info = {
                                "attn": attn_sparse,
                                "attn_c": logits_c_sparse.softmax(dim=-1),
                                "attn_p": gathered_geo.softmax(dim=-1),
                                "logits": logits_sparse,
                                "logits_c": logits_c_sparse,
                                "logits_p": gathered_geo,
                                "gamma_p": None,
                                "gamma_g": gamma_g,
                                "geo_bias": geo_bias,
                                "sampled_idx": sampled_idx,
                                "mode": "sparse_multinomial_geo_sdpa",
                            }
                    else:
                        batch_idx = torch.arange(bh, device=q_c.device)[:, None, None]
                        k_flat = k_c.reshape(bh, n_tokens, dh)
                        v_flat = v.reshape(bh, n_tokens, dh)
                        gathered_k = k_flat[batch_idx, sampled_idx_bh].reshape(B, H, Nq, sampling_k, dh)
                        gathered_v = v_flat[batch_idx, sampled_idx_bh].reshape(B, H, Nq, sampling_k, dh)
                        gathered_geo = gathered_geo_base[:, None, :, :].expand(B, H, Nq, sampling_k)  # (B, H, Nq, K)
                        logits_c_sparse = (q_c[:, :, :, None, :] * gathered_k).sum(dim=-1)
                        logits_sparse = logits_c_sparse / math.sqrt(q_c.shape[-1]) + gathered_geo
                        attn_sparse = logits_sparse.softmax(dim=-1)
                        out = (attn_sparse[..., None] * gathered_v).sum(dim=-2)
                        attn = None

                        if return_attn:
                            attn_info = {
                                "attn": attn_sparse,
                                "attn_c": (logits_c_sparse / math.sqrt(q_c.shape[-1])).softmax(dim=-1),
                                "attn_p": gathered_geo.softmax(dim=-1),
                                "logits": logits_sparse,
                                "logits_c": logits_c_sparse / math.sqrt(q_c.shape[-1]),
                                "logits_p": gathered_geo,
                                "gamma_p": None,
                                "gamma_g": gamma_g,
                                "geo_bias": geo_bias,
                                "sampled_idx": sampled_idx,
                                "mode": "sparse_multinomial_geo_handcraft",
                            }
            else:
                if not handcraft:
                    # Fast path: SDPA with additive geometric bias mask.
                    out = F.scaled_dot_product_attention(q_c, k_c, v, attn_mask=attn_mask)
                    attn = None

                    if return_attn:
                        attn_info = {
                            "attn": None,
                            "attn_c": None,
                            "attn_p": None,
                            "logits": None,
                            "logits_c": None,
                            "logits_p": None,
                            "gamma_p": None,
                            "gamma_g": gamma_g,
                            "geo_bias": geo_bias,
                            "mode": "sdpa_geo_no_attn",
                        }
                else:
                    # Handcrafted path for diagnostics/visualization.
                    logits_c = torch.einsum("bhqd,bhkd->bhqk", q_c, k_c)
                    logits = logits_c / math.sqrt(q_c.shape[-1])
                    logits = logits + attn_mask
                    attn = logits.softmax(dim=-1)
                    out = torch.einsum("bhqk,bhkd->bhqd", attn, v)

                    if return_attn:
                        logits_c_scaled = logits_c / math.sqrt(q_c.shape[-1])
                        logits_p_geo = attn_mask
                        attn_info = {
                            "attn": attn,
                            "attn_c": logits_c_scaled.softmax(dim=-1),
                            "attn_p": logits_p_geo.softmax(dim=-1),
                            "logits": logits,
                            "logits_c": logits_c_scaled,
                            "logits_p": logits_p_geo,
                            "gamma_p": None,
                            "gamma_g": gamma_g,
                            "geo_bias": geo_bias,
                            "mode": "handcraft_geo_full",
                        }
            return out, attn_info

        def _forward_learned_positional(
            self,
            q_c: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            cross_attn_anchor_positional_feat: torch.Tensor,
            plucker_emb_for_concat: torch.Tensor,
            handcraft: bool,
            return_attn: bool,
        ) -> tuple[torch.Tensor, dict[str, Any] | None]:
            """Run learned positional query/key cross-attention.

            Anchor sine features form a positional query and Plucker features
            form a positional key. Their learned similarity is concatenated
            with content similarity; no explicit point-to-ray distance is
            computed in this path.
            """
            attn_info = None
            H = q_c.shape[1]
            k_c = keys
            v = values
            gamma_p = F.softplus(cast(torch.Tensor, self.pos_scale))
            # positional query from 3D anchor
            q_pos_proj = cast(nn.Linear, self.q_pos_proj)
            k_pos_proj = cast(nn.Linear, self.k_pos_proj)
            q_p = q_pos_proj(self.anchor_norm(cross_attn_anchor_positional_feat))  # (B, Nq, D)
            q_p = rearrange(q_p, "b n (h d) -> b h n d", h=H)
            q_p = self.q_pos_norm(q_p)
            # positional key from Plücker feature
            k_p = k_pos_proj(self.plucker_norm(plucker_emb_for_concat))  # (B, N, D)
            k_p = rearrange(k_p, "b n (h d) -> b h n d", h=H)
            k_p = self.k_pos_norm(k_p)

            if not handcraft:
                # Option A: use cat exactly like DAB-style decomposition.
                q = torch.cat([q_c, gamma_p * q_p], dim=-1)  # (B, H, Nq, 2Dh)
                k = torch.cat([k_c, k_p], dim=-1)          # (B, H, N, 2Dh)
                out = F.scaled_dot_product_attention(q, k, v)
                attn = None

                if return_attn:
                    attn_info = {
                        "attn": None,
                        "attn_c": None,
                        "attn_p": None,
                        "logits": None,
                        "logits_c": None,
                        "logits_p": None,
                        "gamma_p": gamma_p,
                        "mode": "sdpa_no_attn",
                    }
            else:
                # Option B: hand-crafted compute matched to SDPA temperature.
                logits_c = torch.einsum("bhqd,bhkd->bhqk", q_c, k_c)
                logits_p = torch.einsum("bhqd,bhkd->bhqk", q_p, k_p)
                dh = q_c.shape[-1]
                logits = (logits_c + gamma_p * logits_p) / math.sqrt(2 * dh)
                attn = logits.softmax(dim=-1)
                out = torch.einsum("bhqk,bhkd->bhqd", attn, v)

                if return_attn:
                    logits_c_scaled = logits_c / math.sqrt(2 * dh)
                    logits_p_scaled = (gamma_p * logits_p) / math.sqrt(2 * dh)
                    attn_info = {
                        "attn": attn,
                        "attn_c": logits_c_scaled.softmax(dim=-1),
                        "attn_p": logits_p_scaled.softmax(dim=-1),
                        "logits": logits,
                        "logits_c": logits_c,
                        "logits_p": logits_p,
                        "gamma_p": gamma_p,
                    }

            return out, attn_info

        def forward(
            self,
            gs_tokens: torch.Tensor,              # (B, N_q, D)
            keys: torch.Tensor,                   # expected (B, H, N, Dh), already projected
            values: torch.Tensor,                 # expected (B, H, N, Dh)
            cross_attn_anchor_positional_feat: torch.Tensor, # (B, N_q, D)
            plucker_emb_for_concat: Optional[torch.Tensor], # (B, N, D)
            anchor_xyz: torch.Tensor,             # (B, N_q, 3)
            anchor_radius: Optional[torch.Tensor],# (B, N_q, 1)
            plucker_rays_raw: torch.Tensor,       # (B, N, 6)
            image_features: Optional[torch.Tensor] = None,
            input_intrinsics: Optional[torch.Tensor] = None,
            input_cam_to_world: Optional[torch.Tensor] = None,
            input_image_size: Optional[tuple[int, int]] = None,
            handcraft: bool = False,
            return_attn: bool = False,
        ):
            B, Nq, D = gs_tokens.shape
            H = self.num_heads

            # content query
            q_c = self.q_content_proj(self.gs_token_norm(gs_tokens))       # (B, Nq, D)
            q_c = rearrange(q_c, "b n (h d) -> b h n d", h=H)
            q_c = self.q_norm(q_c)

            # content keys are assumed to be already shaped as (B, H, N, Dh)
            k_c = keys

            # values
            v = values

            # Select exactly one cross-attention formulation. The first branch
            # has priority so content-only remains a clean ablation even when
            # geometric options are present in the global configuration.
            if self.cross_attn_variant == "content_only":
                out, attn_info = self._forward_content_only(q_c, k_c, v, return_attn)
            elif self.cross_attn_variant == "geometric_positional":
                out, attn_info = self._forward_geometric_positional(
                    q_c, k_c, v, anchor_xyz, anchor_radius, plucker_rays_raw, handcraft, return_attn
                )
            elif self.cross_attn_variant == "deformable":
                if self.deformable_attn is None or image_features is None or input_intrinsics is None or input_cam_to_world is None:
                    raise ValueError("deformable cross-attention requires image features and input camera parameters")
                if input_image_size is None:
                    raise ValueError("deformable cross-attention requires the input image size")
                deformable_out = self.deformable_attn(
                    self.gs_token_norm(gs_tokens),
                    image_features,
                    anchor_xyz,
                    input_intrinsics,
                    input_cam_to_world,
                    input_image_size[0],
                    input_image_size[1],
                    return_attn=return_attn,
                )
                if return_attn:
                    deformable_out, deformable_info = deformable_out
                    attn_info = {
                        "attn": deformable_info["attn"],
                        "attn_c": None,
                        "attn_p": None,
                        "logits": None,
                        "logits_c": None,
                        "logits_p": None,
                        "gamma_p": None,
                        "gamma_g": None,
                        "geo_bias": None,
                        "sampled_idx": None,
                        "sampling_locations": deformable_info["sampling_locations"],
                        "valid_views": deformable_info["valid_views"],
                        "mode": deformable_info["mode"],
                    }
                out = rearrange(deformable_out, "b n (h d) -> b h n d", h=H)
            else:
                assert cross_attn_anchor_positional_feat is not None
                assert plucker_emb_for_concat is not None
                out, attn_info = self._forward_learned_positional(
                    q_c,
                    k_c,
                    v,
                    cross_attn_anchor_positional_feat,
                    plucker_emb_for_concat,
                    handcraft,
                    return_attn,
                )

            out = rearrange(out, "b h n d -> b n (h d)")
            out = self.out_proj(out)
            if return_attn:
                return out, attn_info
            return out

    class MlpBlock(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float, ffn_bias: bool):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, bias=ffn_bias)

        def forward(self, gs_tokens: torch.Tensor) -> torch.Tensor:
            return self.mlp(self.norm(gs_tokens))

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ffn_bias: bool,
        qk_norm: bool,
        init_values: float | None = None,
        attn_score_mod=None,
        cross_attn_variant: str = "learned_positional",
        geo_sparse_sampling_k: int = 0,
        geo_sparse_sampling_tau_init: float = 1.0,
        geo_sparse_sampling_mode: str = "multinomial",
        use_dense_sparse_attn_mask: bool = False,
        deformable_num_points: int = 4,
        deformable_max_views: int = 16,
        self_attn_variant: str = "learned_positional",
    ):
        super().__init__()
        self.cross_attn_variant = cross_attn_variant
        self.self_attn_variant = self_attn_variant
        if self_attn_variant not in {"content_only", "learned_positional"}:
            raise ValueError(f"Unknown self_attn_variant: {self_attn_variant}")

        def make_scale() -> nn.Module:
            return LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.gs_self_attn = DecoderBlockWithAnchor.SelfAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            qk_norm=qk_norm,
            flex_attn_score_mod=attn_score_mod,
        )
        self.gs_self_attn_scale = make_scale()

        self.gs_cross_attn = DecoderBlockWithAnchor.CrossAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            q_norm=qk_norm,
            cross_attn_variant=cross_attn_variant,
            geo_sparse_sampling_k=geo_sparse_sampling_k,
            geo_sparse_sampling_tau_init=geo_sparse_sampling_tau_init,
            geo_sparse_sampling_mode=geo_sparse_sampling_mode,
            use_dense_sparse_attn_mask=use_dense_sparse_attn_mask,
            deformable_num_points=deformable_num_points,
            deformable_max_views=deformable_max_views,
        )
        self.gs_cross_attn_scale = make_scale()

        self.mlp = DecoderBlockWithAnchor.MlpBlock(dim, mlp_ratio, ffn_bias)
        self.mlp_scale = make_scale()

                
        self.anchor_head_cross_attn = Mlp(3 * 128, dim, dim) if cross_attn_variant == "learned_positional" else None
        self.anchor_head_self_attn = Mlp(3 * 128, dim, dim) if self_attn_variant == "learned_positional" else None

    def _build_anchor_positional_features(
        self,
        gs_anchor_normalized: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build the optional positional inputs for cross- and self-attention.

        Both branches share one 3D sine/cosine embedding. The cross-attention
        feature is omitted for attention variants that use explicit geometry
        or image-plane sampling, while the self-attention feature follows its
        own ablation flag.
        """
        use_cross_pos = self.cross_attn_variant == "learned_positional"
        use_self_pos = self.self_attn_variant == "learned_positional"
        if not use_cross_pos and not use_self_pos:
            return None, None

        sine_embed = gen_sineembed_for_3d_anchor(gs_anchor_normalized, num_feats=128)
        cross_feature = None
        self_feature = None

        if use_cross_pos:
            assert self.anchor_head_cross_attn is not None
            cross_feature = self.anchor_head_cross_attn(sine_embed)
        if use_self_pos:
            assert self.anchor_head_self_attn is not None
            self_feature = self.anchor_head_self_attn(sine_embed)
        return cross_feature, self_feature

    def forward(
        self, 
        gs_tokens: torch.Tensor, # (B, N_q, D) 
        gs_anchor_normalized: torch.Tensor, # (B, N_q, 3) in [-1, 1]
        gs_anchor_radius: Optional[torch.Tensor],
        keys: torch.Tensor, 
        values: torch.Tensor,
        plucker_emb_for_concat: Optional[torch.Tensor],
        plucker_rays_patch_raw: torch.Tensor,
        gs_anchor_raw: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        input_intrinsics: Optional[torch.Tensor] = None,
        input_cam_to_world: Optional[torch.Tensor] = None,
        input_image_size: Optional[tuple[int, int]] = None,
        handcraft: bool = False,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:

        attn: dict[str, Any] = {}
        self_attn: dict[str, Any] = {}

        # Build the optional positional inputs for cross- and self-attention.
        cross_attn_anchor_positional_feat, self_attn_anchor_positional_feat = (
            self._build_anchor_positional_features(gs_anchor_normalized)
        )

        # Cross-Attention updates GS tokens from encoder image/ray features.
        # Its internal helper selects content-only, geometric, or learned
        # positional attention according to the configured cross-attn variant.
        cross_attn_out = self.gs_cross_attn(
            gs_tokens,
            keys,
            values,
            cross_attn_anchor_positional_feat,
            plucker_emb_for_concat,
            anchor_xyz=gs_anchor_raw,
            anchor_radius=gs_anchor_radius,
            plucker_rays_raw=plucker_rays_patch_raw,
            image_features=image_features,
            input_intrinsics=input_intrinsics,
            input_cam_to_world=input_cam_to_world,
            input_image_size=input_image_size,
            handcraft=handcraft,
            return_attn=return_attn,
        )
        if return_attn:
            cross_attn_out, attn = cross_attn_out

        # Residual update after Cross-Attention.
        gs_tokens = gs_tokens + self.gs_cross_attn_scale(cross_attn_out)

        # Self-Attention optionally receives the current anchor position
        # directly as an additive feature before token-token interaction.
        self_attn_out = self.gs_self_attn(gs_tokens, self_attn_anchor_positional_feat, return_attn=return_attn)
        if return_attn:
            self_attn_out, self_attn = self_attn_out

        # Residual update after Self-Attention, followed by the feed-forward block.
        gs_tokens = gs_tokens + self.gs_self_attn_scale(self_attn_out)
        gs_tokens = gs_tokens + self.mlp_scale(self.mlp(gs_tokens))

        if return_attn:
            return gs_tokens, {
                "attn": attn.get("attn"),
                "attn_c": attn.get("attn_c"),
                "attn_p": attn.get("attn_p"),
                "cross_attn": attn.get("attn"),
                "cross_attn_c": attn.get("attn_c"),
                "cross_attn_p": attn.get("attn_p"),
                "cross_logits": attn.get("logits"),
                "cross_logits_c": attn.get("logits_c"),
                "cross_logits_p": attn.get("logits_p"),
                "cross_gamma_p": attn.get("gamma_p"),
                "cross_gamma_g": attn.get("gamma_g"),
                "cross_geo_bias": attn.get("geo_bias"),
                "cross_sampled_idx": attn.get("sampled_idx"),
                "cross_sampling_locations": attn.get("sampling_locations"),
                "cross_valid_views": attn.get("valid_views"),
                "cross_mode": attn.get("mode"),
                "self_attn": self_attn.get("attn"),
                "self_logits": self_attn.get("logits"),
                "self_mode": self_attn.get("mode"),
            }
        return gs_tokens
