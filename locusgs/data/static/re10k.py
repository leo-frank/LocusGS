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

import glob
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from locusgs.data.datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_IMAGE_RGB,
)


class RE10K:
    """Directory/list based RE10K loader with the same interface as DL3DV10K."""

    def __init__(
        self,
        root_path=None,
        metadata_path=None,
        resolution=None,
        patch_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root_path = root_path
        self.metadata_path = metadata_path
        self.resolution = resolution
        self.patch_size = patch_size
        self.sample_list = self._collect_scene_jsons()
        self.is_static = True

    def _collect_scene_jsons(self) -> List[str]:
        sample_list: List[str] = []
        if self.metadata_path is not None:
            with open(self.metadata_path, "r") as f:
                sample_list = [line.strip() for line in f if line.strip()]
        elif self.root_path is not None:
            sample_list = sorted(glob.glob(os.path.join(self.root_path, "**", "*.json"), recursive=True))
        else:
            raise ValueError("Either root_path or metadata_path must be provided for RE10K.")
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def load_video_reader(self, idx):
        scene_json_path = self.sample_list[idx]
        with open(scene_json_path, "r") as f:
            data_json = json.load(f)
        frames = data_json["frames"]
        scene_name = data_json.get("scene_name", Path(scene_json_path).stem)
        return scene_name, frames

    def count_cameras(self, video_idx: int) -> int:
        return 1

    def count_frames(self, idx):
        _, frames = self.load_video_reader(idx)
        return len(frames)

    def _resolve_image_path(self, frame: dict) -> str:
        image_path = frame["image_path"]
        if os.path.isabs(image_path):
            return image_path
        if self.root_path is not None:
            return os.path.join(self.root_path, image_path)
        return image_path

    def _load_and_preprocess_frames(self, frames_chosen):
        images = []
        intrinsics = []
        for frame in frames_chosen:
            image_path = self._resolve_image_path(frame)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_np = np.array(image, dtype=np.uint8)
                image_t = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float() / 255.0

            fxfycxcy = np.array(frame["fxfycxcy"], dtype=np.float32)

            images.append(image_t)
            intrinsics.append(torch.from_numpy(fxfycxcy))

        w2cs = np.stack([np.array(frame["w2c"], dtype=np.float32) for frame in frames_chosen], axis=0)
        c2ws = np.linalg.inv(w2cs).astype(np.float32)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        c2ws = torch.from_numpy(c2ws).contiguous()
        return images, intrinsics, c2ws

    def get_data(
        self,
        idx,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: List[int] = None,
        camera_convention: str = "opencv",
    ):
        assert camera_convention == "opencv"

        scene_name, frames = self.load_video_reader(idx)
        if frame_indices is None:
            frame_indices = range(len(frames))
        frame_indices = list(frame_indices)
        frames_chosen = [frames[i] for i in frame_indices]

        img_seq, intrinsics, c2w = self._load_and_preprocess_frames(frames_chosen)

        output_dict = {"__key__": scene_name}
        for data_field in data_fields:
            if data_field == DF_IMAGE_RGB:
                output_dict[data_field] = img_seq
            elif data_field == DF_CAMERA_C2W_TRANSFORM:
                output_dict[data_field] = c2w
            elif data_field == DF_CAMERA_INTRINSICS:
                output_dict[data_field] = intrinsics

        return output_dict


class RE10KEval(RE10K):
    def __init__(
        self,
        root_path=None,
        metadata_path=None,
        evaluation_json=None,
        resolution=256,
        patch_size=16,
        num_input=16,
    ):
        super().__init__(
            root_path=root_path,
            metadata_path=metadata_path,
            resolution=resolution,
            patch_size=patch_size,
        )
        self.num_input = num_input
        self.evaluation_indices = {}
        if evaluation_json is not None:
            with open(evaluation_json, "r") as f:
                self.evaluation_indices = json.load(f)
            # Align with DL3DVEval behavior: rebuild sample_list from evaluation JSON.
            scene_path_map = {Path(p).stem: p for p in self.sample_list}
            if not isinstance(self.evaluation_indices, dict):
                raise ValueError("RE10KEval expects evaluation_json to be a dict: scene_name -> {context, target}.")
            scene_names = [k for k, v in self.evaluation_indices.items() if isinstance(v, dict)]
            self.sample_list = [scene_path_map[s] for s in scene_names if s in scene_path_map]

    def get_context_target_frames(self, idx):
        scene_name, _ = self.load_video_reader(idx)
        if isinstance(self.evaluation_indices, dict) and scene_name in self.evaluation_indices:
            eval_data = self.evaluation_indices[scene_name]
            context_frames = eval_data["context"]
            target_frames = eval_data["target"]
            return context_frames, target_frames
        else:
            assert "not found", scene_name
