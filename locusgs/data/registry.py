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

from pathlib import Path

from locusgs.data.static.dl3dv import DL3DV10K, DL3DVEval
from locusgs.data.static.re10k import RE10K, RE10KEval

# Repository root (parent of the `tokengs` package)
_DEFAULT_DL3DV_ROOT = Path("/data1T") / "DL3DV"
_DEFAULT_DL3DV_EVAL_ROOT = Path("/data1T") / "DL3DV"
_DEFAULT_RE10K_ROOT = Path("/data4T") / "re10k_unpack" / "train"
_DEFAULT_RE10K_EVAL_ROOT = Path("/data4T") / "re10k_unpack" / "test"

dataset_registry = {}

# DL3DV: point `data/dl3dv` at your DL3DV-ALL (or compatible) tree, e.g. `ln -snf /path/to/DL3DV-ALL-960P-undistorted data/dl3dv`
dataset_registry["dl3dv"] = {
    "cls": DL3DV10K,
    "kwargs": {
        "root_path": str(_DEFAULT_DL3DV_ROOT),
        "resolution": "960p_images",
    },
    "max_gap": 40,
    "min_gap": 10,
}

# Eval benchmark: point `data/dl3dv_eval` at DL3DV-10K-Benchmark (or compatible)
dataset_registry["dl3dv_eval"] = {
    "cls": DL3DVEval,
    "kwargs": {
        "root_path": str(_DEFAULT_DL3DV_EVAL_ROOT),
        "subset": ["140"],
    },
    "max_gap": 1e6,
    "min_gap": 0,
}

# RE10K variant for 2-input-view training.
dataset_registry["re10k_2view_gap135"] = {
    "cls": RE10K,
    "kwargs": {
        "root_path": str(_DEFAULT_RE10K_ROOT),
    },
    "max_gap": 135,
    "min_gap": 45,
    "scene_scale": 1,
}

# RE10K variant for 4-input-view training.
dataset_registry["re10k_4view_gap250"] = {
    "cls": RE10K,
    "kwargs": {
        "root_path": str(_DEFAULT_RE10K_ROOT),
    },
    "max_gap": 250,
    "min_gap": 45,
    "scene_scale": 1,
}

# Eval benchmark for RE10K. Override `evaluation_json` and optionally `metadata_path` via dataset_kwargs.
dataset_registry["re10k_eval"] = {
    "cls": RE10KEval,
    "kwargs": {
        "root_path": str(_DEFAULT_RE10K_EVAL_ROOT),
    },
    "max_gap": 1e6,
    "min_gap": 0,
    "scene_scale": 1,
}
