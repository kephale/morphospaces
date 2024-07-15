from typing import Tuple, List

import numpy as np
import zarr

from morphospaces.datasets._base import BaseTiledDataset
from copick.impl.filesystem import CopickRootFSSpec

from morphospaces.datasets.utils import (
    FilterSliceBuilder,
    PatchManager,
    SliceBuilder,
)


class LazyTiledCopickDataset(BaseTiledDataset):
    def __init__(
        self,
        copick_config_path,
        session_id,
        user_id,
        run_names: List[str],
        tomo_type,
        segmentation_name,
        transform,
        patch_shape: Tuple[int, ...] = (96, 96, 96),
        stride_shape: Tuple[int, ...] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (0,),
        patch_threshold: float = 0,
        patch_slack_acceptance=0.01,
        voxel_spacing=None,
        store_unique_label_values: bool = False,
    ):
        self.copick_config_path = copick_config_path
        self.session_id = session_id
        self.user_id = user_id
        self.run_names = run_names
        self.tomo_type = tomo_type
        self.segmentation_name = segmentation_name
        self.voxel_spacing = voxel_spacing
        self.transform = transform
        self.patch_shape = patch_shape
        self.stride_shape = stride_shape
        self.patch_filter_ignore_index = patch_filter_ignore_index
        self.patch_threshold = patch_threshold
        self.patch_slack_acceptance = patch_slack_acceptance
        self.store_unique_label_values = store_unique_label_values

        self.data = self._load_data()
        self.patches = self._create_patches()

        if store_unique_label_values:
            self.unique_label_values = self._get_unique_labels()

    def _load_data(self):
        data = {}
        for run_name in self.run_names:
            data[run_name] = {
                'raw': self.get_array(run_name, 'raw'),
                'label': self.get_array(run_name, 'label')
            }
        return data

    def _create_patches(self):
        patches = {}
        for run_name, run_data in self.data.items():
            patches[run_name] = PatchManager(
                data=run_data,
                patch_shape=self.patch_shape,
                stride_shape=self.stride_shape,
                patch_filter_ignore_index=self.patch_filter_ignore_index,
                patch_filter_key='label',
                patch_threshold=self.patch_threshold,
                patch_slack_acceptance=self.patch_slack_acceptance,
            )
        return patches

    def _get_unique_labels(self):
        unique_labels = set()
        for run_name, run_patches in self.patches.items():
            for slice_indices in run_patches.slices:
                label_patch = self.data[run_name]['label'][slice_indices]
                unique_labels.update(np.unique(label_patch))
        return list(unique_labels)

    def get_array(self, run_name, key):
        return LazyCopickFile(
            self.copick_config_path,
            run_name,
            self.session_id,
            self.user_id,
            self.tomo_type,
            self.segmentation_name,
            self.voxel_spacing,
            key
        )

    def __getitem__(self, idx):
        run_name = np.random.choice(self.run_names)
        run_patches = self.patches[run_name]
        
        if idx >= len(run_patches):
            raise StopIteration

        slice_indices = run_patches.slices[idx]
        data_patch = {
            key: self.data[run_name][key][slice_indices]
            for key in ['raw', 'label']
        }

        if self.transform is not None:
            data_patch = self.transform(data_patch)

        return data_patch

    def __len__(self):
        return sum(len(patches) for patches in self.patches.values())

class LazyCopickFile:
    def __init__(self, copick_config_path, run_name, session_id, user_id, tomo_type, segmentation_name, voxel_spacing, key):
        self.copick_config_path = copick_config_path
        self.run_name = run_name
        self.session_id = session_id
        self.user_id = user_id
        self.tomo_type = tomo_type
        self.segmentation_name = segmentation_name
        self.voxel_spacing = voxel_spacing
        self.key = key
        self.array = self._load_array()

    def _load_array(self):
        root = CopickRootFSSpec.from_file(self.copick_config_path)
        run = root.get_run(self.run_name)
        voxel_spacing_obj = run.get_voxel_spacing(self.voxel_spacing)
        
        if self.key == 'raw':
            tomogram = voxel_spacing_obj.get_tomogram(self.tomo_type)
            return zarr.open(tomogram.zarr(), mode='r')['0']
        elif self.key == 'label':
            segmentation = run.get_segmentations(
                user_id=self.user_id,
                session_id=self.session_id,
                is_multilabel=True,
                name=self.segmentation_name,
                voxel_size=self.voxel_spacing
            )
            if segmentation:
                return zarr.open(segmentation[0].zarr(), mode='r')['data']
            else:
                raise ValueError(f"Segmentation not found for run {self.run_name}")
        else:
            raise ValueError(f"Invalid key: {self.key}. Use 'raw' or 'label'.")

    def __getitem__(self, arg):
        return self.array[arg]

    def ravel(self):
        return self.array.ravel()

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim
