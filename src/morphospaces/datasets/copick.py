from typing import Tuple, List

import numpy as np
import zarr

from morphospaces.datasets._base import BaseTiledDataset
from copick.impl.filesystem import CopickRootFSSpec


class LazyTiledCopickDataset(BaseTiledDataset):
    """Implementation of the Copick dataset which loads the data lazily.
    It's slower, but has a low memory footprint.
    """

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
    
    def get_array(self, run_name):
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
    """Implementation of the LazyCopickFile class for the LazyCopickDataset."""

    def __init__(self, copick_config_path, run_name, session_id, user_id, tomo_type, segmentation_name, voxel_spacing):
        self.copick_config_path = copick_config_path
        self.run_name = run_name
        self.session_id = session_id
        self.user_id = user_id
        self.tomo_type = tomo_type
        self.segmentation_name = segmentation_name
        self.voxel_spacing = voxel_spacing
        array = self.to_array()
        try:
            self.ndim = array.ndim
            self.shape = array.shape
        except AttributeError:
            print(copick_config_path)
        except KeyError:
            print(copick_config_path)

    def to_array(self) -> zarr.core.Array:
        root = CopickRootFSSpec.from_file(self.copick_config_path)
        run = root.get_run(self.run_name)
        voxel_spacing_obj = run.get_voxel_spacing(self.voxel_spacing)
        tomogram = voxel_spacing_obj.get_tomogram(self.tomo_type)
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
