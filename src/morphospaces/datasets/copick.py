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

        super().__init__(
            patch_filter_key="label",
            dataset_keys=run_names,
            file_path=copick_config_path,
            transform=transform,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=patch_filter_ignore_index,
            patch_threshold=patch_threshold,
            patch_slack_acceptance=patch_slack_acceptance,
            store_unique_label_values=store_unique_label_values,
        )
    
    def get_array(self, file_path, run_name):
        return LazyCopickFile(
            self.copick_config_path,
            run_name,
            self.session_id,
            self.user_id,
            self.tomo_type,
            self.segmentation_name,
            self.voxel_spacing
        )

class LazyCopickFile:
    def __init__(self, copick_config_path, run_name, session_id, user_id, tomo_type, segmentation_name, voxel_spacing):
        self.copick_config_path = copick_config_path
        self.run_name = run_name
        self.session_id = session_id
        self.user_id = user_id
        self.tomo_type = tomo_type
        self.segmentation_name = segmentation_name
        self.voxel_spacing = voxel_spacing
        self.raw_array = None
        self.label_array = None
        self._load_arrays()

    def _load_arrays(self):
        root = CopickRootFSSpec.from_file(self.copick_config_path)
        run = root.get_run(self.run_name)
        voxel_spacing_obj = run.get_voxel_spacing(self.voxel_spacing)
        
        # Load raw data
        tomogram = voxel_spacing_obj.get_tomogram(self.tomo_type)
        self.raw_array = zarr.open(tomogram.path, mode='r')['data']
        
        # Load label data
        segmentation = run.get_segmentations(
            user_id=self.user_id,
            session_id=self.session_id,
            is_multilabel=True,
            name=self.segmentation_name,
            voxel_size=self.voxel_spacing
        )
        if segmentation:
            self.label_array = zarr.open(segmentation[0].path, mode='r')['data']
        else:
            raise ValueError(f"Segmentation not found for run {self.run_name}")
        
        # Set shape and ndim based on raw data
        self.shape = self.raw_array.shape
        self.ndim = self.raw_array.ndim

    def __getitem__(self, arg):
        if isinstance(arg, str):
            if arg == 'raw':
                return self.raw_array
            elif arg == 'label':
                return self.label_array
            else:
                raise KeyError(f"Invalid key: {arg}. Use 'raw' or 'label'.")

        if arg == Ellipsis:
            return self

        # For slice indexing, return a dict with both raw and label data
        return {
            'raw': self.raw_array[arg],
            'label': self.label_array[arg]
        }

    def ravel(self):
        return {
            'raw': self.raw_array.ravel(),
            'label': self.label_array.ravel()
        }
