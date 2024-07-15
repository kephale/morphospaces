from typing import Tuple, List

import numpy as np
import zarr

from morphospaces.datasets._base import BaseTiledDataset
from copick.impl.filesystem import CopickRootFSSpec


class LazyTiledCopickDataset(BaseTiledDataset):
    def __init__(
        self,
        copick_config_path,
        session_id,
        user_id,
        run_names,
        tomo_type,
        segmentation_name,
        transform,
        patch_shape=(96, 96, 96),
        stride_shape=(24, 24, 24),
        patch_filter_ignore_index=(0,),
        patch_threshold=0,
        patch_slack_acceptance=0.01,
        voxel_spacing=None,
        store_unique_label_values=False,
    ):
        self.copick_config_path = copick_config_path
        self.session_id = session_id
        self.user_id = user_id
        self.run_names = run_names
        self.tomo_type = tomo_type
        self.segmentation_name = segmentation_name
        self.voxel_spacing = voxel_spacing

        # We'll use 'raw' and 'label' as keys, even though they're not directly in run_names
        dataset_keys = ['raw', 'label']

        super().__init__(
            patch_filter_key="label",
            dataset_keys=dataset_keys,
            file_path=copick_config_path,  # We're not really using this, but it's required
            transform=transform,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=patch_filter_ignore_index,
            patch_threshold=patch_threshold,
            patch_slack_acceptance=patch_slack_acceptance,
            store_unique_label_values=store_unique_label_values,
        )
    
    def get_array(self, file_path, key):
        # Here, we'll return a LazyCopickFile for both 'raw' and 'label'
        # The LazyCopickFile will handle the distinction internally
        return LazyCopickFile(
            self.copick_config_path,
            self.run_names[0],  # Assuming we're only using the first run for now
            self.session_id,
            self.user_id,
            self.tomo_type,
            self.segmentation_name,
            self.voxel_spacing,
            key  # Pass 'raw' or 'label' to LazyCopickFile
        )

class LazyCopickFile:
    def __init__(self, copick_config_path, run_name, session_id, user_id, tomo_type, segmentation_name, voxel_spacing, data_type):
        self.copick_config_path = copick_config_path
        self.run_name = run_name
        self.session_id = session_id
        self.user_id = user_id
        self.tomo_type = tomo_type
        self.segmentation_name = segmentation_name
        self.voxel_spacing = voxel_spacing
        self.data_type = data_type  # 'raw' or 'label'
        
        array = self.to_array()
        self.ndim = array.ndim
        self.shape = array.shape

    def to_array(self) -> zarr.core.Array:
        root = CopickRootFSSpec.from_file(self.copick_config_path)
        run = root.get_run(self.run_name)
        voxel_spacing_obj = run.get_voxel_spacing(self.voxel_spacing)
        
        if self.data_type == 'raw':
            tomogram = voxel_spacing_obj.get_tomogram(self.tomo_type)
            return zarr.open(tomogram.path, mode='r')['data']
        elif self.data_type == 'label':
            segmentation = run.get_segmentations(
                user_id=self.user_id,
                session_id=self.session_id,
                is_multilabel=True,
                name=self.segmentation_name,
                voxel_size=self.voxel_spacing
            )
            if segmentation:
                return zarr.open(segmentation[0].path, mode='r')['data']
            else:
                raise ValueError(f"Segmentation not found for run {self.run_name}")

    def ravel(self) -> np.ndarray:
        return np.ravel(self.to_array())

    def __getitem__(self, arg):
        if isinstance(arg, str) and not self.run_name:
            return LazyCopickFile(
                self.copick_config_path,
                arg,
                self.session_id,
                self.user_id,
                self.tomo_type,
                self.segmentation_name,
                self.voxel_spacing
            )

        if arg == Ellipsis:
            return LazyCopickFile(
                self.copick_config_path,
                self.run_name,
                self.session_id,
                self.user_id,
                self.tomo_type,
                self.segmentation_name,
                self.voxel_spacing
            )

        array = self.to_array()
        item = array[arg]
        del array
        return item
