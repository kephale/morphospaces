import logging
import sys
import argparse

import torch
import pytorch_lightning as pl
from monai.data import DataLoader
from monai.transforms import (
    Compose, RandAffined, RandFlipd, RandRotate90d, EnsureChannelFirstd,
    ScaleIntensityd, Resized, ToTensord, AsDiscrete, EnsureType
)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F

from morphospaces.datasets import CopickDataset
from morphospaces.transforms.label import LabelsAsFloat32
from morphospaces.transforms.image import ExpandDimsd, StandardizeImage    
from monai.networks.nets import UNet
from torch.nn import CrossEntropyLoss

def train_unet_copick(
        copick_config_path,
        train_run_names,
        val_run_names,
        tomo_type,
        user_id,
        session_id,
        segmentation_type,
        voxel_spacing,
        lr=0.0001,
        logdir="checkpoints"
    ):

    # setup logging
    logger = logging.getLogger("lightning.pytorch")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # patch parameters
    batch_size = 1
    patch_shape = (96, 96, 96)
    patch_stride = (96, 96, 96)
    patch_threshold = 0.5

    image_key = "zarr_tomogram"
    labels_key = "zarr_mask"

    learning_rate_string = str(lr).replace(".", "_")
    logdir_path = "./" + logdir

    # training parameters
    n_samples_per_class = 1000
    log_every_n_iterations = 100
    val_check_interval = 0.15
    lr_reduction_patience = 25
    lr_scheduler_step = 1500
    accumulate_grad_batches = 4
    memory_banks: bool = True
    n_pixel_embeddings_per_class: int = 1000
    n_pixel_embeddings_to_update: int = 10
    n_label_embeddings_per_class: int = 50
    n_memory_warmup: int = 1000

    pl.seed_everything(42, workers=True)

    train_transform = Compose(
        [
            LabelsAsFloat32(keys=labels_key),
            StandardizeImage(keys=image_key),
            ExpandDimsd(
                keys=[
                    image_key,
                    labels_key,
                ]
            ),
            RandFlipd(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.2,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.2,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.2,
                spatial_axis=2,
            ),
            RandRotate90d(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.25,
                spatial_axes=(0, 1),
            ),
            RandRotate90d(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.25,
                spatial_axes=(0, 2),
            ),
            RandRotate90d(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.25,
                spatial_axes=(1, 2),
            ),
            RandAffined(
                keys=[
                    image_key,
                    labels_key,
                ],
                prob=0.5,
                mode="nearest",
                rotate_range=(1.5, 1.5, 1.5),
                translate_range=(20, 20, 20),
                scale_range=0.1,
            ),
        ]
    )
    train_ds, unique_train_label_values = CopickDataset.from_copick_project(
        copick_config_path=copick_config_path,
        run_names=train_run_names.split(","),
        tomo_type=tomo_type,
        user_id=user_id,
        session_id=session_id,
        segmentation_type=segmentation_type,
        voxel_spacing=voxel_spacing,
        transform=train_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        store_unique_label_values=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_transform = Compose(
        [
            LabelsAsFloat32(keys=labels_key),
            StandardizeImage(keys=image_key),
            ExpandDimsd(
                keys=[
                    image_key,
                    labels_key,
                ]
            )
        ]
    )

    val_ds, unique_val_label_values = CopickDataset.from_copick_project(
        copick_config_path=copick_config_path,
        run_names=val_run_names.split(","),
        tomo_type=tomo_type,
        user_id=user_id,
        session_id=session_id,
        segmentation_type=segmentation_type,
        voxel_spacing=voxel_spacing,
        transform=val_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        store_unique_label_values=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    unique_label_values = set(unique_train_label_values).union(
        set(unique_val_label_values)
    )

    num_classes = len(unique_label_values)

    # Log dataset info before training starts
    print("Training Dataset Info:")
    print(f"Number of samples: {len(train_ds)}")
    print(f"Image shape: {train_ds[0][image_key].shape}")
    print(f"Label shape: {train_ds[0][labels_key].shape}")
    print(f"Image dtype: {train_ds[0][image_key].dtype}")
    print(f"Label dtype: {train_ds[0][labels_key].dtype}")
    print(f"Unique label values: {unique_train_label_values}")
    print(f"Number of classes: {num_classes}")

    print("\nValidation Dataset Info:")
    print(f"Number of samples: {len(val_ds)}")
    print(f"Image shape: {val_ds[0][image_key].shape}")
    print(f"Label shape: {val_ds[0][labels_key].shape}")
    print(f"Image dtype: {val_ds[0][image_key].dtype}")
    print(f"Label dtype: {val_ds[0][labels_key].dtype}")
    print(f"Unique label values: {unique_val_label_values}")
    print(f"Number of classes: {num_classes}")

    best_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="unet-best",
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="unet-last",
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval="step")

    class UNetSegmentation(pl.LightningModule):
        def __init__(self, lr, num_classes):
            super().__init__()
            self.lr = lr
            self.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=num_classes,  # Use the dynamically determined number of classes
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
            self.loss_function = CrossEntropyLoss()
            self.val_outputs = []

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            labels = labels.squeeze(1).long()  # Convert labels to Long and squeeze
            outputs = self.forward(images)
            loss = self.loss_function(outputs, labels)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            labels = labels.squeeze(1).long()  # Convert labels to Long and squeeze
            outputs = self.forward(images)

            # Debugging information
            if batch_idx == 0:
                self.logger.experiment.add_text(
                    "Debug/Images Shape", str(images.shape), self.current_epoch
                )
                self.logger.experiment.add_text(
                    "Debug/Labels Shape", str(labels.shape), self.current_epoch
                )
                self.logger.experiment.add_text(
                    "Debug/Outputs Shape", str(outputs.shape), self.current_epoch
                )
                self.logger.experiment.add_text(
                    "Debug/Labels Unique Values", str(torch.unique(labels).tolist()), self.current_epoch
                )
                self.logger.experiment.add_text(
                    "Debug/Outputs Unique Values", str(torch.unique(outputs).tolist()), self.current_epoch
                )

            try:
                val_loss = self.loss_function(outputs, labels)
            except RuntimeError as e:
                print(f"Validation loss computation failed: {e}")
                print(f"Output shape: {outputs.shape}")
                print(f"Label shape: {labels.shape}")
                print(f"Output unique values: {torch.unique(outputs)}")
                print(f"Label unique values: {torch.unique(labels)}")
                raise e

            self.log("val_loss", val_loss)
            self.val_outputs.append(val_loss)
            return val_loss

        def on_validation_epoch_end(self):
            self.val_outputs.clear()

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

        
    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    net = UNetSegmentation(lr=lr, num_classes=num_classes)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback, learning_rate_monitor],
        logger=logger,
        max_epochs=10000,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_iterations,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=6,
    )
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D UNet network using the Copick dataset for segmentation.")
    parser.add_argument("--copick_config_path", type=str, required=True, help="Path to the Copick configuration file")
    parser.add_argument("--train_run_names", type=str, required=True, help="Names of the runs in the Copick project for training")
    parser.add_argument("--val_run_names", type=str, required=True, help="Names of the runs in the Copick project for validation")
    parser.add_argument("--tomo_type", type=str, required=True, help="Tomogram type in the Copick project")
    parser.add_argument("--user_id", type=str, required=True, help="User ID for the Copick project")
    parser.add_argument("--session_id", type=str, required=True, help="Session ID for the Copick project")
    parser.add_argument("--segmentation_type", type=str, required=True, help="Segmentation type in the Copick project")
    parser.add_argument("--voxel_spacing", type=float, required=True, help="Voxel spacing for the Copick project")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the UNet training")
    parser.add_argument("--logdir", type=str, default="checkpoints", help="Output directory name in the current working directory. Default is checkpoints")
    
    args = parser.parse_args()
    
    train_unet_copick(
        copick_config_path=args.copick_config_path,
        train_run_names=args.train_run_names,
        val_run_names=args.val_run_names,
        tomo_type=args.tomo_type,
        user_id=args.user_id,
        session_id=args.session_id,
        segmentation_type=args.segmentation_type,
        voxel_spacing=args.voxel_spacing,
        lr=args.lr,
        logdir=args.logdir
    )
