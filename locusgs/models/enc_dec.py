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

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from .attention import Block
from locusgs.options import Options
from .decoder_blocks import DecoderBlock, DecoderBlockWithAnchor

class EncDecBackbone(nn.Module):
    """
    An encoder-decoder backbone for the EncDec architecture.

    Encoder: a stack of ViT blocks which produce a latent representation. This is followed by a key-value projection to produce a key and value for the decoder.

    Decoder: a stack of transformer decoder layers which attend from GS tokens to the encoder output and among themselves.
    """

    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        self.encoder = nn.Sequential(
            *[
                Block(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01,
                    qk_norm=True,
                    rope=None,
                    flex_attn_block_mask=None,
                )
                for _ in range(self.opt.enc_depth)
            ]
        )

        self.encoder_norm = nn.LayerNorm(self.opt.enc_embed_dim)

        self.kv_proj = nn.Linear(
            self.opt.enc_embed_dim, self.opt.enc_embed_dim * 2, bias=True
        )

        # normalize the k projection, q are normalized in the attention block
        self.k_proj_norm = nn.LayerNorm(self.opt.enc_embed_dim // self.opt.enc_num_heads)

        if self.opt.num_dynamic_gs_tokens > 0:
            block_ids = [0] * self.opt.num_gs_tokens + [1] * self.opt.num_dynamic_gs_tokens
            self.register_buffer(
                "_decoder_block_ids",
                torch.tensor(block_ids, dtype=torch.long),
                persistent=False,
            )

            def block_causal_score_mod(score, b, h, q_idx, kv_idx, block_ids_tensor: torch.Tensor):
                same_block_mask = block_ids_tensor[q_idx] == block_ids_tensor[kv_idx]
                causal_mask = q_idx >= kv_idx
                return torch.where(same_block_mask | causal_mask, score, float("-inf"))

            score_mod = partial(block_causal_score_mod, block_ids_tensor=self._decoder_block_ids)
        else:
            score_mod = None
        if self.opt.model_type == "locusgs":
            decoder_block_cls = DecoderBlockWithAnchor
            decoder_block_kwargs = {
                "cross_attn_variant": self.opt.cross_attn_variant,
                "geo_sparse_sampling_k": self.opt.geo_sparse_sampling_k,
                "geo_sparse_sampling_tau_init": self.opt.geo_sparse_sampling_tau_init,
                "geo_sparse_sampling_mode": self.opt.geo_sparse_sampling_mode,
                "use_dense_sparse_attn_mask": self.opt.use_dense_sparse_attn_mask,
                "self_attn_variant": self.opt.self_attn_variant,
            }
        elif self.opt.model_type == "tokengs":
            decoder_block_cls = DecoderBlock
            decoder_block_kwargs = {}
        else:
            raise AssertionError(f"unknown model type {self.opt.model_type}")

        self.decoder_blocks = nn.ModuleList(
            [
                decoder_block_cls(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    ffn_bias=True,
                    init_values=5e-3 * self.opt.gs_token_std,
                    qk_norm=True,
                    attn_score_mod=score_mod,
                    **decoder_block_kwargs,
                )
                for _ in range(self.opt.dec_depth)
            ]
        )

    def _encode_to_kv(
        self, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encoder(image_features)
        image_features = self.encoder_norm(image_features)
        image_feature_keys, image_feature_values = rearrange(
            self.kv_proj(image_features),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=self.opt.enc_num_heads,
        )
        image_feature_keys = self.k_proj_norm(image_feature_keys)
        return image_features, image_feature_keys, image_feature_values
