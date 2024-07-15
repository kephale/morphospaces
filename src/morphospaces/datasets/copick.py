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
        mirror_padding=(16, 32, 32),
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
            file_path=copick_config_path,
            stage='',
            transform=transform,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=patch_filter_ignore_index,
            patch_threshold=patch_threshold,
            patch_slack_acceptance=patch_slack_acceptance,
            mirror_padding=mirror_padding,
            raw_internal_path=None,
            label_internal_path=None,
            weight_internal_path=None,
            store_unique_label_values=store_unique_label_values,
        )

    @staticmethod
    def get_array(copick_config_path, internal_path):
        return LazyCopickFile(copick_config_path, internal_path)


class LazyCopickFile:
    """Implementation of the LazyCopickFile class for the LazyCopickDataset."""

    def __init__(self, copick_config_path, internal_path=None):
        self.copick_config_path = copick_config_path
        self.internal_path = internal_path
        if self.internal_path:
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
        arrays = []

        for run_name in self.internal_path['run_names']:
            run = root.get_run(run_name)
            voxel_spacing_obj = run.get_voxel_spacing(self.internal_path['voxel_spacing'])
            tomogram = voxel_spacing_obj.get_tomogram(self.internal_path['tomo_type'])
            segmentation = run.get_segmentations(
                user_id=self.internal_path['user_id'],
                session_id=self.internal_path['session_id'],
                is_multilabel=True,
                name=self.internal_path['segmentation_name'],
                voxel_size=self.internal_path['voxel_spacing']
            )
            if segmentation:
                arrays.append(zarr.open(segmentation[0].path, mode='r')['data'])
            else:
                raise ValueError(f"Segmentation not found for run {run_name}")

        return np.concatenate(arrays, axis=0)

    def ravel(self) -> np.ndarray:
        return np.ravel(self.to_array())

    def __getitem__(self, arg):
        if isinstance(arg, str) and not self.internal_path:
            return LazyCopickFile(self.copick_config_path, arg)

        if arg == Ellipsis:
            return LazyCopickFile(self.copick_config_path, self.internal_path)

        array = self.to_array()
        item = array[arg]
        del array
        return item
