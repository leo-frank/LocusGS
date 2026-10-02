# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def register_dl3dv_ablation_presets(config_defaults, config_doc, options_cls):
    """Register DL3DV LocusGS structural ablation presets."""
    # See Table 4: ablation of the structural components under the 4-view setting.
    config_doc["train_dl3dv_locusgs_full"] = (
        "DL3DV LocusGS ablation full model: XYZ+Radius anchor state/refinement, "
        "radius-adaptive cross-attention, radius-scaled Gaussian offset."
    )
    config_defaults["train_dl3dv_locusgs_full"] = config_defaults["train_dl3dv_base"]

    config_doc["train_dl3dv_locusgs_ablation_no_radius_state"] = (
        "Anchor Representation ablation: remove anchor radius state and use anchor-to-ray attention "
        "with anchor-offset decoding."
    )
    config_defaults["train_dl3dv_locusgs_ablation_no_radius_state"] = config_defaults[
        "train_dl3dv_locusgs_full"
    ].evolve(
        use_anchor_radius=False,
        anchor_radius_affects_bias=False,
        anchor_radius_refinement=False,
        gaussian_center_variant="anchor_offset",
    )

    config_doc["train_dl3dv_locusgs_ablation_no_radius_refinement"] = (
        "Anchor Representation ablation: keep radius state but disable progressive radius refinement."
    )
    config_defaults["train_dl3dv_locusgs_ablation_no_radius_refinement"] = config_defaults[
        "train_dl3dv_locusgs_full"
    ].evolve(
        anchor_radius_refinement=False,
    )

    config_doc["train_dl3dv_locusgs_ablation_content_attention"] = (
        "Geometry-aware Cross-Attention ablation: content-only cross-attention with the rest of the model unchanged."
    )
    config_defaults["train_dl3dv_locusgs_ablation_content_attention"] = config_defaults[
        "train_dl3dv_locusgs_full"
    ].evolve(
        cross_attn_variant="content_only",
    )

    config_doc["train_dl3dv_locusgs_ablation_anchor_to_ray_attention"] = (
        "Geometry-aware Cross-Attention ablation: keep point-to-ray geometry but disable radius-adaptive bias."
    )
    config_defaults["train_dl3dv_locusgs_ablation_anchor_to_ray_attention"] = config_defaults[
        "train_dl3dv_locusgs_full"
    ].evolve(
        anchor_radius_affects_bias=False,
    )

    config_doc["train_dl3dv_locusgs_ablation_no_self_attn_anchor_pos"] = (
        "Self-Attention ablation: remove explicit current-anchor 3D positional feature injection from LocusGS self-attention."
    )
    config_defaults["train_dl3dv_locusgs_ablation_no_self_attn_anchor_pos"] = config_defaults[
        "train_dl3dv_locusgs_full"
    ].evolve(
        self_attn_variant="content_only",
    )

    config_doc["train_dl3dv_locusgs_ablation_free_gaussian_center"] = (
        "Anchor-guided Gaussian Decoding ablation: predict free Gaussian centers without anchor-guided offsets."
    )
    config_defaults["train_dl3dv_locusgs_ablation_free_gaussian_center"] = config_defaults[
        "train_dl3dv_locusgs_full"
    ].evolve(
        gaussian_center_variant="free_center",
    )

    config_doc["train_dl3dv_locusgs_ablation_anchor_offset"] = (
        "Anchor-guided Gaussian Decoding ablation: decode Gaussian centers as anchor plus unscaled offset."
    )
    config_defaults["train_dl3dv_locusgs_ablation_anchor_offset"] = config_defaults[
        "train_dl3dv_locusgs_full"
    ].evolve(
        gaussian_center_variant="anchor_offset",
    )
