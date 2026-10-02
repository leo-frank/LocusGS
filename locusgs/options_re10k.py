# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def register_re10k_presets(config_defaults, config_doc, options_cls):
    config_doc["train_re10k_base_4_input_views"] = "RE10K training defaults (4 input views, min/max gap 45/250)."
    config_defaults["train_re10k_base_4_input_views"] = options_cls(
        num_epochs=300,
        max_iters_per_epoch=500,
        pct_start_steps=2000,
        num_input_views=4,
        data_mode=(("re10k_4view_gap250", 6),),
        camera_scale_method="distance",
    )
    config_doc["train_re10k_base_2_input_views"] = "RE10K training defaults (2 input views, min/max gap 45/135)."
    config_defaults["train_re10k_base_2_input_views"] = options_cls(
        num_epochs=300,
        max_iters_per_epoch=500,
        pct_start_steps=2000,
        num_input_views=2,
        data_mode=(("re10k_2view_gap135", 6),),
        camera_scale_method="distance",
    )

    config_doc["train_re10k_base_2_input_views_unposed"] = (
        "RE10K training defaults (2 input views, min/max gap 45/135, unposed input with input supervision)."
    )
    config_defaults["train_re10k_base_2_input_views_unposed"] = config_defaults[
        "train_re10k_base_2_input_views"
    ].evolve(
        unposed_input=True,
        use_input_supervision=False,
    )

    config_doc["finetune_re10k_2view"] = "Short RE10K finetune from existing tokens, 2 input views, wide images."
    config_defaults["finetune_re10k_2view"] = config_defaults["train_re10k_base_2_input_views"].evolve(
        num_epochs=20,
        pct_start_steps=400,
        lr=4e-5,
        num_gs_tokens=4096,
        init_tokens_from_existing=True,
        num_input_views=2,
        img_size=(256, 256),
    )

    config_doc["eval_re10k_2view_base"] = "RE10K eval preset: 2 views, eval JSON, single batch."
    config_defaults["eval_re10k_2view_base"] = options_cls(
        data_mode=(("re10k_eval", 1),),
        dataset_kwargs={"evaluation_json": "assets/evaluation_index_re10k_128_random.json"},
        num_input_views=2,
        img_size=(256, 256),
        camera_scale_method="distance",
        evaluating=True,
        num_gs_tokens=1024,
        use_input_supervision=False,
        batch_size=1,
    )
    config_doc["eval_re10k_2view"] = "RE10K eval preset: 2 views, eval JSON, single batch."
    config_defaults["eval_re10k_2view"] = config_defaults["eval_re10k_2view_base"].evolve(
        img_size=(256, 256),
        num_gs_tokens=4096,
    )
