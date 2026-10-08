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

"""Tyro CLI options and named presets (`AllConfigs` subcommands)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import tyro

from locusgs.options_dl3dv import register_dl3dv_presets
from locusgs.options_dl3dv_ablation import register_dl3dv_ablation_presets
from locusgs.options_re10k import register_re10k_presets


@dataclass
class Options:
    # --- general
    evaluating: bool = False
    workspace: str = "./workspace"
    resume: str | None = None
    model_type: str = "locusgs"
    seed: int = 42

    # --- logging
    experiment_name: str = "locusgs"

    # --- model architecture
    img_size: tuple[int, int] = (256, 256)
    patch_size: int = 8
    enc_depth: int = 3
    dec_depth: int = 12
    enc_embed_dim: int = 1024
    enc_num_heads: int = 16
    mlp_ratio: int = 4

    # --- gaussian splatting
    bg_color: Literal["white", "black", "grey"] = "grey"
    gaussian_scale_cap: float = 0.075
    gaussian_z_offset: float = 1.0
    num_gs_tokens: int = 1024
    token_dim: int = 1024
    gs_token_std: float = 1e-2
    num_gaussians_per_token: int | None = 64
    num_dynamic_gs_tokens: int = 0
    init_dynamic_tokens_from_static: bool = False
    init_tokens_from_existing: bool = False
    anchor_supervision_layers: tuple[int, ...] = (6, 12)
    geo_sparse_sampling_k: int = 0
    geo_sparse_sampling_tau_init: float = 1.0
    geo_sparse_sampling_mode: Literal["multinomial", "topk"] = "multinomial"
    use_dense_sparse_attn_mask: bool = False
    cross_attn_handcraft: bool = False
    cross_attn_variant: Literal["content_only", "learned_positional", "geometric_positional"] = "geometric_positional"
    # Select whether LocusGS self-attention uses learned anchor positional features.
    # "content_only" isolates token-token interaction from anchor position.
    self_attn_variant: Literal["content_only", "learned_positional"] = "learned_positional"
    # --- anchor radius experiment (default off; preserves existing LocusGS behavior)
    use_anchor_radius: bool = True
    anchor_radius_init: float = 1.0
    anchor_radius_min: float = 1e-3
    # If True, radius scales the point-to-ray geometric bias bandwidth in cross-attention.
    # Larger radius makes the geometric matching less strict; smaller radius makes it sharper.
    anchor_radius_affects_bias: bool = True
    # If True, decoder layers also refine anchor radius progressively; otherwise radius stays fixed within a forward pass.
    anchor_radius_refinement: bool = True
    gaussian_center_variant: Literal["radius_scaled_offset", "anchor_offset", "free_center"] = "radius_scaled_offset"

    # --- dataset
    data_mode: tuple[tuple[str, int], ...] = (("dl3dv_scaled_0.15", 6),)
    num_views: int = 8
    num_input_views: int = 4
    znear: float = 0.025
    zfar: float = 125.0
    camera_normalization_method: Literal["mean_cam", "first_cam"] = "first_cam"
    camera_scale_method: Literal["constant", "distance", "bound"] = "constant"
    num_workers: int = 16
    dataset_kwargs: dict[str, str] | None = None
    dataset_max_gap: int | None = None
    dataset_min_gap: int | None = None

    # --- training
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 30
    max_iters_per_epoch: int = 1_000_000
    checkpoint_save_every: int = 20
    lr: float = 4e-4
    pct_start_steps: int = 1000
    final_div_factor: float = 1000.0
    gradient_clip: float = 1.0
    mixed_precision: str = "bf16"
    deferred_bp: bool = False
    use_input_supervision: bool = False
    unposed_input: bool = False

    # --- loss weights
    lambda_lpips: float = 0.0
    lambda_mask: float = 0.0
    lambda_ssim: float = 0.2
    lambda_visibility: float = 1.0
    visibility_distance_threshold: float = 1.0
    lambda_anchor_visibility: float = 0.1

    # --- logging frequency
    print_freq: int = 10
    log_image_freq: int = 100
    debug_dump_iteration: int = -1
    debug_dump_path: str | None = None
    debug_replay_batch_path: str | None = None
    detect_anomaly: bool = False

    # --- evaluation
    eval_n_media_dumps: int = 0

    # --- test-time training (eval)
    use_ttt_for_eval: bool = False
    ttt_n_steps: int = 50
    ttt_lr: float = 1e-4

    # --- depthsplat evaluation interop
    depthsplat_experiment: str = "dl3dv"
    depthsplat_num_scales: int = 2
    depthsplat_upsample_factor: int = 4
    depthsplat_lowest_feature_resolution: int = 8
    depthsplat_monodepth_vit_type: str = "vitb"
    depthsplat_near: float = 0.5
    depthsplat_far: float = 200.0

    # --- dynamic scenes
    time_embedding: bool = False
    time_embedding_dim: int = 2
    use_interp_target: bool = False

    # --- augmentation
    random_reflect: bool = True

    def __post_init__(self) -> None:
        if self.evaluating:
            assert not self.use_input_supervision, "use_input_supervision must be False when evaluating"

    def evolve(self, **changes: Any) -> Options:
        """Return a deep copy with the given fields replaced."""
        new_instance = copy.deepcopy(self)
        for key, value in changes.items():
            if not hasattr(new_instance, key):
                raise AttributeError(f"Options has no attribute '{key}'")
            setattr(new_instance, key, value)
        return new_instance


config_defaults: dict[str, Options] = {}
config_doc: dict[str, str] = {}

register_dl3dv_presets(config_defaults, config_doc, Options)
register_dl3dv_ablation_presets(config_defaults, config_doc, Options)
register_re10k_presets(config_defaults, config_doc, Options)
config_doc["eval_re10k_2view_256"] = "RE10K eval preset: 2 input views, 256x256, 1024 GS tokens."
config_defaults["eval_re10k_2view_256"] = config_defaults["eval_re10k_2view"].evolve(
    img_size=(256, 256),
    num_gs_tokens=1024,
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
