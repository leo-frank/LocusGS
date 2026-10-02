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


class DL3DV10K:
    """Directory-based DL3DV loader for local data layout."""

    def __init__(
        self,
        root_path,
        subset=['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K', '12K'],
        resolution='960p',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.root_path = root_path
        self.subset = subset

        # Local dataset uses images_8: 3840x2160 downsampled by 8 -> 480x270.
        if resolution in ('960p', '960p_images', '960p_images_8', 'images_8'):
            self.resolution = [270, 480]  # H, W
            self.image_folder = 'images_8'
        else:
            raise NotImplementedError(f"Resolution {resolution} not supported")

        self.sample_list = []
        for sub in self.subset:
            base_dir = root_path if sub == '140' else os.path.join(root_path, sub)
            candidates = sorted(glob.glob(os.path.join(base_dir, '*')))
            for sample in candidates:
                if os.path.isdir(sample) and self._has_scene_transforms(sample):
                    self.sample_list.append(sample)

        self.is_static = True

    def __len__(self):
        return len(self.sample_list)

    def _has_scene_transforms(self, scene_dir: str) -> bool:
        return os.path.isfile(os.path.join(scene_dir, 'transforms.json')) or os.path.isfile(
            os.path.join(scene_dir, 'gaussian_splat', 'transforms.json')
        )

    def _resolve_scene_root(self, scene_dir: str) -> str:
        benchmark_root = os.path.join(scene_dir, 'gaussian_splat')
        if os.path.isfile(os.path.join(benchmark_root, 'transforms.json')):
            return benchmark_root
        return scene_dir

    def load_intrinsics(self, data_dict, resolution=None):
        fx = data_dict['fl_x']
        fy = data_dict['fl_y']
        cx = data_dict['cx']
        cy = data_dict['cy']

        intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
        if resolution is not None:
            H_new, W_new = resolution
            W_old = data_dict['w']
            H_old = data_dict['h']

            # Scale intrinsics from original frame size to loaded image size.
            intrinsics[0] *= W_new / W_old
            intrinsics[1] *= H_new / H_old
            intrinsics[2] *= W_new / W_old
            intrinsics[3] *= H_new / H_old

        return intrinsics

    def load_video_reader(self, idx):
        scene_dir = self.sample_list[idx]
        clip_name = Path(scene_dir).name
        scene_root = self._resolve_scene_root(scene_dir)

        with open(os.path.join(scene_root, 'transforms.json'), 'r') as f:
            json_data = json.load(f)

        video_length = len(json_data['frames'])
        intrinsics = self.load_intrinsics(json_data, resolution=self.resolution)
        transform_matrix_all = self.load_cameras(json_data)

        return clip_name, scene_root, video_length, intrinsics, transform_matrix_all, json_data

    def load_cameras(self, data_dict):
        transform_matrix_all = []
        for frame_data in data_dict['frames']:
            transform_matrix = np.array(frame_data['transform_matrix'])
            c2w = transform_matrix
            c2w[2, :] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1
            transform_matrix_all.append(c2w)
        return np.stack(transform_matrix_all, axis=0)

    def count_cameras(self, video_idx: int) -> int:
        return 1

    def count_frames(self, idx):
        _, _, video_length, _, _, _ = self.load_video_reader(idx)
        return video_length

    def get_data(
        self,
        idx,
        data_fields: List[str],
        frame_indices: Optional[List[int]] = None,
        view_indices: List[int] = None,
        camera_convention: str = 'opencv',
    ):
        assert camera_convention == 'opencv'

        clip_name, scene_root, total_frames, intrinsics, transform_matrices, json_data = self.load_video_reader(idx)
        if frame_indices is None:
            frame_indices = range(total_frames)
        frame_indices = list(frame_indices)

        c2w = transform_matrices[frame_indices]

        img_seq = []
        image_paths = []
        for frame_idx in frame_indices:
            img_name = json_data['frames'][frame_idx]['file_path'].split('/')[-1]
            image_path = os.path.join(scene_root, self.image_folder, img_name)
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f'Image not found: {image_path}')
            image_paths.append(os.path.abspath(image_path))
            with Image.open(image_path) as img:
                img_seq.append(np.array(img.convert('RGB')))

        img_seq = np.stack(img_seq, axis=0)  # [N, H, W, C]
        img_seq = torch.from_numpy(img_seq).permute(0, 3, 1, 2).contiguous().float() / 255.0

        c2w = torch.from_numpy(c2w).float().contiguous()
        intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).repeat(len(frame_indices), 1)

        output_dict = {
            '__key__': clip_name,
            'source_metadata': {
                'scene_name': clip_name,
                'scene_path': os.path.abspath(scene_root),
                'transforms_path': os.path.abspath(os.path.join(scene_root, 'transforms.json')),
                'image_paths': image_paths,
            },
        }
        for data_field in data_fields:
            if data_field == DF_IMAGE_RGB:
                output_dict[data_field] = img_seq
            elif data_field == DF_CAMERA_C2W_TRANSFORM:
                output_dict[data_field] = c2w
            elif data_field == DF_CAMERA_INTRINSICS:
                output_dict[data_field] = intrinsics

        return output_dict


class DL3DVEval(DL3DV10K):
    def __init__(
        self,
        root_path,
        evaluation_json,
        subset=['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K', '12K'],
        resolution='960p',
        num_input=16,
    ):
        super().__init__(root_path, subset, resolution)

        self.evaluation_indices = json.load(open(evaluation_json, 'r'))
        self.sample_list = []
        for k in self.evaluation_indices:
            scene_name = k if isinstance(k, str) else k['scene_name']
            scene_path = os.path.join(root_path, scene_name)
            if not os.path.isdir(scene_path):
                # Fallback: search in subset folders.
                matches = glob.glob(os.path.join(root_path, '*', scene_name))
                if not matches:
                    print(f'[WARN] Cannot find evaluation scene directory for {scene_name}, skipping.')
                    continue
                scene_path = matches[0]
            self.sample_list.append(scene_path)

        self.is_static = True
        self.num_input = num_input

    def get_context_target_frames(self, idx):
        if isinstance(self.evaluation_indices, dict):
            scene_name = Path(self.sample_list[idx]).name
            eval_data = self.evaluation_indices[scene_name]
            context_frames = eval_data['context']
            target_frames = eval_data['target']
            return context_frames, target_frames

        eval_data = self.evaluation_indices[idx]
        if self.num_input == 16:
            context_frames = eval_data['fold_8_kmeans_16_input']
        elif self.num_input == 32:
            context_frames = eval_data['fold_8_kmeans_32_input']
        else:
            raise ValueError(f'Unsupported number of input frames: {self.num_input}')
        target_frames = [x for x in range(self.count_frames(idx))]
        return context_frames, target_frames
