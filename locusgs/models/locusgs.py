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

"""
TokenGS: Encoder-decoder model for 3D scene reconstruction from sparse views.
"""

from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial

import torch.nn.functional as F
from lpips import LPIPS

from locusgs.options import Options
from locusgs.models.input_types import (
    EncoderLatent,
    ModelInputDecoder,
    ModelInputEncoder,
    ModelSupervision,
    split_data,
)
from .attention import PatchEmbed
from locusgs.models.enc_dec import EncDecBackbone
from locusgs.rendering.gs import GaussianRenderer
from locusgs.models.activations import ClipActivationHead
from locusgs.models.losses import compute_anchor_visibility_loss, compute_tokengs_loss
from locusgs.models.attention import Mlp


def patchify_raw_plucker(
    plucker: torch.Tensor,  # (B, V, 6, H, W)
    patch_size: int,
) -> torch.Tensor:
    """
    Returns:
        plucker_patch: (B, V * H_p * W_p, 6)
    """
    B, V, C, H, W = plucker.shape
    assert C == 6

    x = plucker.reshape(B * V, C, H, W)

    # First version: average raw Plücker inside each patch.
    # Better version: use exact patch-center ray if you can compute it.
    x = F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)

    H_p, W_p = x.shape[-2:]
    x = x.reshape(B, V, C, H_p, W_p)
    x = x.permute(0, 1, 3, 4, 2).reshape(B, V * H_p * W_p, C)

    return x


def inverse_softplus(x: float) -> float:
    x_t = torch.tensor(float(x), dtype=torch.float32)
    return torch.log(torch.expm1(x_t)).item()

class LocusGS(nn.Module):
    """
    LocusGS model with encoder-decoder architecture.
    
    Uses separate encoder and decoder with cross-attention for Gaussian token processing.
    """
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt
        
        self.img_size = self.opt.img_size if not isinstance(self.opt.img_size, int) else [self.opt.img_size, self.opt.img_size]

        norm_layer_factory = partial(nn.LayerNorm, bias=True)

        self.patch_embed = PatchEmbed(
            img_size=self.opt.img_size,
            patch_size=self.opt.patch_size,
            in_chans=3,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory,
        )

        # both for plucker embedding
        self.patch_plucker_embed = PatchEmbed(
            img_size=self.opt.img_size,
            patch_size=self.opt.patch_size,
            in_chans=6,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory,
        )
        if self.opt.cross_attn_variant == "learned_positional":
            self.patch_plucker_embed_for_concat = PatchEmbed(
                img_size=self.opt.img_size,
                patch_size=self.opt.patch_size,
                in_chans=6,
                embed_dim=self.opt.enc_embed_dim,
                norm_layer=norm_layer_factory,
            )

        # Encoder-decoder architecture
        self.enc_dec_backbone = EncDecBackbone(opt)

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # Activation head (always clip)
        self.activation_head = ClipActivationHead(opt)

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

        # Learnable GS tokens
        self.gs_tokens = nn.Parameter(
            self.opt.gs_token_std * torch.randn(self.opt.num_gs_tokens, self.opt.token_dim)
        )

        # Learnable anchor XYZ and optional anchor radius.
        anchor_std = 1
        self.gs_anchor_xyz_raw = nn.Parameter(
            anchor_std * torch.randn(self.opt.num_gs_tokens, 3)
        )
        with torch.no_grad():
            self.gs_anchor_xyz_raw[:, 2] += self.opt.gaussian_z_offset
        if self.opt.use_anchor_radius:
            radius_init_raw = inverse_softplus(max(self.opt.anchor_radius_init, self.opt.anchor_radius_min))
            self.gs_anchor_radius_raw = nn.Parameter(
                torch.full((self.opt.num_gs_tokens, 1), radius_init_raw)
            )
        else:
            self.gs_anchor_radius_raw = None

        self.scene_radius = 1
        self.xyz_embed = Mlp(self.opt.enc_embed_dim, self.opt.enc_embed_dim, 3)
        if self.opt.anchor_radius_refinement:
            self.radius_embed = Mlp(self.opt.enc_embed_dim, self.opt.enc_embed_dim, 1)

    def state_dict(self, **kwargs):
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if "lpips_loss" in k:
                del state_dict[k]
        return state_dict

    def _background_color(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.opt.bg_color == "white":
            return torch.ones(3, dtype=dtype, device=device)
        if self.opt.bg_color == "black":
            return torch.zeros(3, dtype=dtype, device=device)
        if self.opt.bg_color == "grey":
            return torch.ones(3, dtype=dtype, device=device) * 0.5
        raise ValueError(f"Invalid background color: {self.opt.bg_color}")

    def _check_finite_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            finite_mask = torch.isfinite(tensor)
            total = tensor.numel()
            bad = total - int(finite_mask.sum().item())
            raise RuntimeError(
                f"{name} contains non-finite values: bad={bad}/{total}, "
                f"min={torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).min().item():.6g}, "
                f"max={torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).max().item():.6g}"
            )

    def forward_encoder(self, encoder_input: ModelInputEncoder) -> EncoderLatent:
        """
        Encode input views into latent representation (keys and values for cross-attention).
        
        Args:
            encoder_input: ModelInputEncoder containing input view data (only input views)
            
        Returns:
            EncoderLatent containing keys and values for cross-attention
        """
        B, V, _, H, W = encoder_input.images_rgb.shape
        height = int(H // self.opt.patch_size)
        width = int(W // self.opt.patch_size)

        assert height * self.opt.patch_size == H, f"H={H} must be divisible by patch_size={self.opt.patch_size}"
        assert width * self.opt.patch_size == W, f"W={W} must be divisible by patch_size={self.opt.patch_size}"

        # Reshape for tokenization
        images_rgb_reshaped = encoder_input.images_rgb.reshape(B * V, 3, H, W)
        plucker_reshaped = encoder_input.plucker.reshape(B * V, 6, H, W)
        
        # Embed RGB patches
        x = self.patch_embed(images_rgb_reshaped)
        # Add Plucker embeddings
        x_plucker_emb = self.patch_plucker_embed(plucker_reshaped)

        if self.opt.cross_attn_variant == "learned_positional":
            plucker_emb_for_concat = self.patch_plucker_embed_for_concat(plucker_reshaped)
            plucker_emb_for_concat = rearrange(plucker_emb_for_concat, "(b v) n c -> b (v n) c", b=B, v=V)
        else:
            plucker_emb_for_concat = None
        plucker_rays_patch_raw = patchify_raw_plucker(encoder_input.plucker, self.opt.patch_size)
        x = x + x_plucker_emb  # B*V, N, C
        
        # Reshape from (B*V, N, C) to (B, V*N, C) so the encoder sees one sequence per batch
        x = rearrange(x, "(b v) n c -> b (v n) c", b=B, v=V)

        # Run encoder on joint views and produce keys/values
        # x shape: (B, V*N, C) where all views are processed jointly
        image_features, image_feature_keys, image_feature_values = self.enc_dec_backbone._encode_to_kv(x)
        image_features = image_features.reshape(B, V, height, width, self.opt.enc_embed_dim)
        
        return EncoderLatent(
            keys=image_feature_keys,
            values=image_feature_values,
            plucker_emb_for_concat=plucker_emb_for_concat,
            plucker_rays_patch_raw=plucker_rays_patch_raw,
            image_features=image_features
        )

    def get_gs_tokens(self, batch_size: int) -> torch.Tensor:
        """
        Get initial GS tokens (without time conditioning).
        Useful for test-time training where you want to optimize the GS tokens.
        Time conditioning is applied later in forward_decoder.
        
        Args:
            batch_size: Batch size
            
        Returns:
            GS tokens with shape [B, num_gs_tokens, C]
        """
        batch_gs_tokens = self.gs_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        return batch_gs_tokens

    def normalize_anchor(
        self,
        anchor_xyz_raw,
        batch_size=None,
    ):
        anchor_xyz = self.scene_radius * torch.tanh(anchor_xyz_raw)
        if batch_size is not None:
            anchor_xyz = anchor_xyz.unsqueeze(0).repeat(batch_size, 1, 1) # (B N_query, 3)
        return anchor_xyz

    def activate_anchor_radius(
        self,
        anchor_radius_raw: torch.Tensor,
    ) -> torch.Tensor:
        return F.softplus(anchor_radius_raw).clamp_min(self.opt.anchor_radius_min)

    def compose_gaussian_xyz(
        self,
        gaussians_xyz: torch.Tensor,
        gs_anchor_raw: torch.Tensor,
        gs_anchor_radius: Optional[torch.Tensor],
        num_gaussians_per_token: int,
    ) -> torch.Tensor:
        variant = self.opt.gaussian_center_variant
        if variant == "free_center":
            return gaussians_xyz

        gs_anchor_raw_layer = gs_anchor_raw.repeat_interleave(num_gaussians_per_token, dim=1)
        if variant == "anchor_offset":
            return gaussians_xyz + gs_anchor_raw_layer

        if variant == "radius_scaled_offset":
            if gs_anchor_radius is None:
                raise ValueError("gaussian_center_variant='radius_scaled_offset' requires anchor radius.")
            gs_anchor_radius_layer = gs_anchor_radius.repeat_interleave(num_gaussians_per_token, dim=1)
            return gaussians_xyz * gs_anchor_radius_layer + gs_anchor_raw_layer

        raise ValueError(f"Unknown gaussian_center_variant: {variant}")
    
    def forward_decoder(
        self, 
        encoder_latent: EncoderLatent,
        decoder_input: ModelInputDecoder,
        gs_tokens: Optional[torch.Tensor] = None,
        return_intermediate_gaussians: bool = False,
        return_attn: bool = False,
    ):
        """
        Process latent representation (keys/values) to Gaussians using decoder with cross-attention.
        
        Args:
            encoder_latent: EncoderLatent containing keys and values from the encoder
            decoder_input: ModelInputDecoder containing rendering parameters and target time
            gs_tokens: Optional precomputed GS tokens [B, num_gs_tokens, C]. If None, creates new ones.
            return_intermediate_gaussians: Whether to decode selected intermediate layers.
            return_attn: Whether to collect and return per-layer attention diagnostics.
            
        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians.
            Optionally also returns intermediate gaussians/anchors and/or attention maps.
        """
        B = encoder_latent.keys.shape[0]
        # intermediate = []
        
        # Get or use provided GS tokens (without time conditioning)
        if gs_tokens is None:
            gs_tokens = self.get_gs_tokens(batch_size=B) # (B N_query, D)

        # Keep anchors in raw space and only activate when feeding decoder.
        gs_anchor_raw = self.gs_anchor_xyz_raw.unsqueeze(0).repeat(B, 1, 1)
        gs_anchor_radius_raw = self.gs_anchor_radius_raw.unsqueeze(0).repeat(B, 1, 1) if self.opt.use_anchor_radius else None
        gs_anchor_radius = self.activate_anchor_radius(gs_anchor_radius_raw) if gs_anchor_radius_raw is not None else None

        supervised_layers = set(self.opt.anchor_supervision_layers)
        intermediate_gaussians: dict[int, torch.Tensor] = {}
        intermediate_anchors_raw: dict[int, torch.Tensor] = {}
        intermediate_anchor_radii: dict[int, torch.Tensor] = {}
        attn_maps: list[dict[str, torch.Tensor | float | str | None]] = []
        total_layers = len(self.enc_dec_backbone.decoder_blocks)
        if total_layers not in supervised_layers:
            raise ValueError(
                f"anchor_supervision_layers must include the final decoder layer {total_layers}, "
                f"got {sorted(supervised_layers)}"
            )
        
        # Run decoder with precomputed keys/values
        for layer_idx, layer in enumerate(self.enc_dec_backbone.decoder_blocks):
            layer_out = layer(
                gs_tokens=gs_tokens,
                gs_anchor_normalized=self.normalize_anchor(gs_anchor_raw),
                gs_anchor_radius=gs_anchor_radius if self.opt.anchor_radius_affects_bias else None,
                keys=encoder_latent.keys,
                values=encoder_latent.values,
                plucker_emb_for_concat=encoder_latent.plucker_emb_for_concat,
                plucker_rays_patch_raw=encoder_latent.plucker_rays_patch_raw,
                gs_anchor_raw=gs_anchor_raw,
                handcraft=self.opt.cross_attn_handcraft,
                return_attn=return_attn,
            )
            if return_attn:
                gs_tokens, attn_info = layer_out
                attn_maps.append(
                    {
                        k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in attn_info.items()
                    }
                )
            else:
                gs_tokens = layer_out
            # Predict anchor delta in the same raw xyz space as the activation head.
            delta_raw = self.activation_head.pos_act(self.xyz_embed(gs_tokens))
            gs_anchor_raw = gs_anchor_raw + delta_raw
            if self.opt.anchor_radius_refinement:
                delta_radius_raw = self.radius_embed(gs_tokens)
                gs_anchor_radius_raw = gs_anchor_radius_raw + delta_radius_raw
                gs_anchor_radius = self.activate_anchor_radius(gs_anchor_radius_raw)

            # Optionally decode selected intermediate layers to Gaussians for deep supervision.
            layer_1based = layer_idx + 1
            if return_intermediate_gaussians and layer_1based in supervised_layers:
                gaussians_layer = self.activation_head(gs_tokens)
                num_gaussians_per_token = self.activation_head.num_gaussians_per_token
                gaussians_xyz = gaussians_layer[..., :3]
                gaussians_rest = gaussians_layer[..., 3:]
                gaussians_xyz = self.compose_gaussian_xyz(
                    gaussians_xyz,
                    gs_anchor_raw,
                    gs_anchor_radius,
                    num_gaussians_per_token,
                )
                gaussians_layer = torch.cat([gaussians_xyz, gaussians_rest], dim=-1)
                intermediate_gaussians[layer_1based] = gaussians_layer
                intermediate_anchors_raw[layer_1based] = gs_anchor_raw
                if gs_anchor_radius is not None:
                    intermediate_anchor_radii[layer_1based] = gs_anchor_radius

        if return_intermediate_gaussians:
            gaussians = intermediate_gaussians[total_layers]
        else:
            gaussians = self.activation_head(gs_tokens)
            num_gaussians_per_token = self.activation_head.num_gaussians_per_token
            gaussians_xyz = gaussians[..., :3]
            gaussians_rest = gaussians[..., 3:]
            gaussians_xyz = self.compose_gaussian_xyz(
                gaussians_xyz,
                gs_anchor_raw,
                gs_anchor_radius,
                num_gaussians_per_token,
            )
            gaussians = torch.cat([gaussians_xyz, gaussians_rest], dim=-1)

        if return_intermediate_gaussians and return_attn:
            return gaussians, intermediate_gaussians, intermediate_anchors_raw, intermediate_anchor_radii, attn_maps
        if return_intermediate_gaussians:
            return gaussians, intermediate_gaussians, intermediate_anchors_raw, intermediate_anchor_radii
        if return_attn:
            return gaussians, attn_maps
        return gaussians

    def render_gaussians(self, gaussians: torch.Tensor, decoder_input: ModelInputDecoder) -> dict:
        """
        Render Gaussians to images.
        
        Args:
            gaussians: Gaussian parameters [B, N, 14]
            decoder_input: ModelInputDecoder containing camera parameters
            
        Returns:
            Dictionary with 'images_pred', 'alphas_pred', etc.
        """
        bg_color = self._background_color(gaussians.dtype, gaussians.device)
        results = self.gs.render(
            gaussians,
            decoder_input.cam_view,
            bg_color=bg_color,
            intrinsics=decoder_input.intrinsics,
        )
        return results

    def compute_loss(
        self,
        gaussians: torch.Tensor,
        decoder_input: ModelInputDecoder,
        supervision: ModelSupervision,
        anchor_raw: Optional[torch.Tensor] = None,
    ) -> dict:
        render_results = self.render_gaussians(gaussians, decoder_input)
        results = compute_tokengs_loss(
            opt=self.opt,
            img_size=self.img_size,
            render_results=render_results,
            supervision=supervision,
            gaussians_xyz=gaussians[..., :3],
            lpips_loss=getattr(self, "lpips_loss", None),
        )
        if self.opt.lambda_anchor_visibility > 0 and anchor_raw is not None:
            anchor_vis = compute_anchor_visibility_loss(
                anchor_xyz=anchor_raw,
                cam_view=decoder_input.cam_view,
                intrinsics=decoder_input.intrinsics,
                img_size=self.img_size,
                visibility_distance_threshold=self.opt.visibility_distance_threshold,
            )
            results["loss_anchor_visibility"] = anchor_vis
            results["loss"] = results["loss"] + self.opt.lambda_anchor_visibility * anchor_vis
        return results

    def compute_multi_layer_loss(
        self,
        gaussians_by_layer: dict[int, torch.Tensor],
        anchors_raw_by_layer: dict[int, torch.Tensor],
        decoder_input: ModelInputDecoder,
        supervision: ModelSupervision,
    ) -> dict:
        selected_layers = sorted(
            layer for layer in self.opt.anchor_supervision_layers if layer in gaussians_by_layer
        )
        # Increasing weights from early selected layers to later ones.
        weights = torch.arange(
            1,
            len(selected_layers) + 1,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        weights = weights / weights.sum()

        layer_results: dict[int, dict] = {}
        total_loss = 0.0
        for idx, layer in enumerate(selected_layers):
            res = self.compute_loss(
                gaussians_by_layer[layer],
                decoder_input,
                supervision,
                anchor_raw=anchors_raw_by_layer.get(layer),
            )
            layer_results[layer] = res
            total_loss = total_loss + weights[idx] * res["loss"]

        final_layer = max(selected_layers)
        final_results = layer_results[final_layer]
        merged_results = {k: v for k, v in final_results.items()}
        merged_results["loss"] = total_loss
        for idx, layer in enumerate(selected_layers):
            merged_results[f"loss_layer_{layer}"] = layer_results[layer]["loss"]
            merged_results[f"loss_weight_layer_{layer}"] = weights[idx]
        return merged_results

    def forward(self, data, skip_loss=False):
        """
        Forward pass of the TokenGS model.
        
        Args:
            data: Dictionary from dataloader
            skip_loss: If True, skip loss computation
            
        Returns:
            Dictionary containing results including 'loss', 'gaussians', 'images_pred', etc.
        """
        # Split data into structured input and supervision
        model_input, supervision = split_data(data, self.opt)

        # Encode once, decode once. Optionally supervise multiple decoder layers.
        encoder_latent = self.forward_encoder(model_input.encoder)
        if len(self.opt.anchor_supervision_layers) == 0:
            raise ValueError("anchor_supervision_layers must be non-empty (e.g. [12]).")
        gaussians, gaussians_by_layer, anchors_raw_by_layer, _ = self.forward_decoder(
            encoder_latent,
            model_input.decoder,
            return_intermediate_gaussians=True,
        )
        self._check_finite_tensor("gaussians(final)", gaussians)
        for layer_id, g_layer in gaussians_by_layer.items():
            self._check_finite_tensor(f"gaussians(layer={layer_id})", g_layer)

        # Compute loss or just render
        if skip_loss:
            results = self.render_gaussians(gaussians, model_input.decoder)
            results['gaussians'] = gaussians
            return results
        
        # Compute loss
        results = self.compute_multi_layer_loss(
            gaussians_by_layer,
            anchors_raw_by_layer,
            model_input.decoder,
            supervision,
        )
        results['gaussians'] = gaussians
        results['images_output'] = supervision.images_output
        return results
