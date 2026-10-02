# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def register_dl3dv_presets(config_defaults, config_doc, options_cls):
    config_doc["train_dl3dv_base"] = "DL3DV training defaults (long schedule, capped iters/epoch)."
    config_defaults["train_dl3dv_base"] = options_cls(
        model_type="locusgs",
        cross_attn_variant="geometric_positional",
        use_anchor_radius=True,
        anchor_radius_affects_bias=True,
        anchor_radius_refinement=True,
        gaussian_center_variant="radius_scaled_offset",
        num_epochs=300,
        max_iters_per_epoch=500,
        pct_start_steps=2000,
    )

    config_doc["finetune_dl3dv_2view"] = "Short finetune from existing tokens, 2 input views, wide images."
    config_defaults["finetune_dl3dv_2view"] = config_defaults["train_dl3dv_base"].evolve(
        num_epochs=20,
        pct_start_steps=400,
        lr=4e-5,
        num_gs_tokens=4096,
        init_tokens_from_existing=True,
        num_input_views=2,
        img_size=(256, 448),
    )

    config_doc["finetune_dl3dv_4view"] = "Like finetune_dl3dv_2view with 4 input views."
    config_defaults["finetune_dl3dv_4view"] = config_defaults["finetune_dl3dv_2view"].evolve(
        num_input_views=4,
    )

    config_doc["finetune_dl3dv_6view"] = "Like finetune_dl3dv_2view with 6 input views and 10 total views."
    config_defaults["finetune_dl3dv_6view"] = config_defaults["finetune_dl3dv_2view"].evolve(
        num_input_views=6,
        num_views=10,
    )
    config_doc["finetune_dl3dv_12view"] = "Like finetune_dl3dv_2view with 12 input views and 10 total views."
    config_defaults["finetune_dl3dv_12view"] = config_defaults["finetune_dl3dv_2view"].evolve(
        num_input_views=12,
        num_views=24,
        img_size=(256, 256),
        dataset_max_gap=100,
        dataset_min_gap=20,
        num_epochs=160,
        init_tokens_from_existing=True,
    )

    config_doc["eval_dl3dv_2view"] = "DL3DV eval preset: 2 views, eval JSON, single batch."
    config_defaults["eval_dl3dv_2view"] = options_cls(
        data_mode=(("dl3dv_eval_scaled_0.15", 1),),
        dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_2v.json"},
        num_input_views=2,
        img_size=(256, 448),
        evaluating=True,
        num_gs_tokens=4096,
        use_input_supervision=False,
        batch_size=1,
    )

    config_doc["eval_dl3dv_4view"] = "DL3DV eval preset: 4 input views."
    config_defaults["eval_dl3dv_4view"] = config_defaults["eval_dl3dv_2view"].evolve(
        num_input_views=4,
        dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_4v.json"},
    )

    config_doc["eval_dl3dv_6view"] = "DL3DV eval preset: 6 input views."
    config_defaults["eval_dl3dv_6view"] = config_defaults["eval_dl3dv_2view"].evolve(
        num_input_views=6,
        dataset_kwargs={"evaluation_json": "assets/evaluation_idx_dl3dv_depthsplat_6v.json"},
    )

    config_doc["eval_dl3dv_2view_256"] = "DL3DV eval preset: 2 input views, 256x256, 1024 GS tokens."
    config_defaults["eval_dl3dv_2view_256"] = config_defaults["eval_dl3dv_2view"].evolve(
        img_size=(256, 256),
        num_gs_tokens=1024,
    )

    config_doc["eval_dl3dv_4view_256"] = "DL3DV eval preset: 4 input views, 256x256, 1024 GS tokens."
    config_defaults["eval_dl3dv_4view_256"] = config_defaults["eval_dl3dv_4view"].evolve(
        img_size=(256, 256),
        num_gs_tokens=1024,
    )

    config_doc["eval_dl3dv_6view_256"] = "DL3DV eval preset: 6 input views, 256x256, 1024 GS tokens."
    config_defaults["eval_dl3dv_6view_256"] = config_defaults["eval_dl3dv_6view"].evolve(
        img_size=(256, 256),
        num_gs_tokens=1024,
    )
