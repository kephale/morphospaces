import os
import numpy as np
import zarr
from copick.impl.filesystem import CopickRootFSSpec
from typing import Dict, List, Tuple, Union
from numpy.typing import ArrayLike
from torch.utils.data import ConcatDataset
from morphospaces.datasets._base import BaseTiledDataset


class CopickDataset(BaseTiledDataset):
    """
    Implementation of the copick dataset that loads both image zarr arrays and their corresponding mask zarr arrays into numpy arrays, 
    constructing a map-style dataset, such as {'zarr_tomogram': Array([...], dtype=np.float32), 'zarr_mask': Array([...], dtype=np.float32)}.
    """

    def __init__(
        self,
        zarr_data: dict,  # {'zarr_tomogram': zarr_array, 'zarr_mask': zarr_array}
        transform=None,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (96, 96, 96),
        stride_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (),
        patch_filter_key: str = "zarr_mask",
        patch_threshold: float = 0.6,
        patch_slack_acceptance=0.01,
        store_unique_label_values: bool = False,
    ):
        
        dataset_keys = zarr_data.keys()
        self.zarr_data = zarr_data
        
        super().__init__(
            dataset_keys=dataset_keys,  # List['zarr_tomogram', 'zarr_mask']
            transform=transform,
            patch_shape=patch_shape,
            stride_shape=stride_shape,
            patch_filter_ignore_index=patch_filter_ignore_index,
            patch_filter_key=patch_filter_key,
            patch_threshold=patch_threshold,
            patch_slack_acceptance=patch_slack_acceptance,
            store_unique_label_values=store_unique_label_values,
        )

        self.data: Dict[str, ArrayLike] = {
            key: zarr_data[key].astype(np.float32) for key in dataset_keys
        }
        self._init_states()

    @classmethod
    def from_copick_project(
        cls,
        copick_config_path: str,
        run_names: List[str],
        tomo_type: str,
        user_id: str,
        session_id: str,
        segmentation_type: str,
        voxel_spacing: float,
        transform=None,
        patch_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (96, 96, 96),
        stride_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (24, 24, 24),
        patch_filter_ignore_index: Tuple[int, ...] = (),
        patch_filter_key: str = "zarr_mask",
        patch_threshold: float = 0.6,
        patch_slack_acceptance=0.01,
        store_unique_label_values: bool = False,
    ):
        root = CopickRootFSSpec.from_file(copick_config_path)
        datasets = []

        for run_name in run_names:
            run = root.get_run(run_name)
            if run is None:
                raise ValueError(f"Run with name '{run_name}' not found.")
            
            voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
            if voxel_spacing_obj is None:
                raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run_name}'.")

            tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
            if tomogram is None:
                raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")

            image = zarr.open(tomogram.zarr(), mode='r')['0']
            
            seg = run.get_segmentations(user_id=user_id, session_id=session_id, is_multilabel=True, name=segmentation_type, voxel_size=voxel_spacing)
            if len(seg) == 0:
                raise ValueError(f"No segmentations found for session '{session_id}' and segmentation type '{segmentation_type}'.")

            segmentation = zarr.open(seg[0].path, mode="r")['data']
            
            zarr_data = {
                'zarr_tomogram': image,
                'zarr_mask': segmentation
            }
            
            dataset = cls(
                zarr_data=zarr_data,
                transform=transform,
                patch_shape=patch_shape,
                stride_shape=stride_shape,
                patch_filter_ignore_index=patch_filter_ignore_index,
                patch_filter_key=patch_filter_key,
                patch_threshold=patch_threshold,
                patch_slack_acceptance=patch_slack_acceptance,
                store_unique_label_values=store_unique_label_values,
            )
            datasets.append(dataset)
        
        if store_unique_label_values:
            unique_label_values = set()
            for dataset in datasets:
                unique_label_values.update(dataset.unique_label_values)

            return ConcatDataset(datasets), list(unique_label_values)

        return ConcatDataset(datasets)

# Example usage:
# copick_dataset = CopickDataset.from_copick_project(copick_config_path='path/to/config', run_names=['run1', 'run2'], tomo_type='tomo_type', user_id='user_id', session_id='session_id', segmentation_type='segmentation_type')
