import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from nuscenes.nuscenes import NuScenes
import numpy as np
from src.utils import vox_utils, data_utils, geom_utils
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from src.datamodules.nuscenes_dataset import NuscData, VizData

class NuScenesDataModule(pl.LightningDataModule):
    def __init__(self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig):
        super().__init__()
        self.cfg_loaders = loaders
        self.dataroot = datasets.dataroot
        self.version = datasets.version
        self.data_aug_conf = datasets.data_aug_conf
        self.nsweeps = datasets.nsweeps
        self.centroid = np.array(datasets.centroid).reshape([1, 3]) if datasets.centroid else None
        self.bounds = tuple(datasets.bounds) if datasets.bounds else None
        self.res_3d = tuple(datasets.res_3d) if datasets.res_3d else None
        self.seqlen = datasets.seqlen
        self.refcam_id = datasets.refcam_id
        self.get_tids = datasets.get_tids
        self.temporal_aug = datasets.temporal_aug
        self.do_shuffle_cams = datasets.do_shuffle_cams
    def prepare_data(self):
        # Load NuScenes once and cache if necessary
        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)

    def setup(self, stage=None):
        # Define training and validation datasets


        if stage == 'fit' or stage is None:
            self.train_dataset = VizData(
                self.nusc, self.dataroot, is_train=True, data_aug_conf=self.data_aug_conf,
                nsweeps=self.nsweeps, centroid=self.centroid, bounds=self.bounds, 
                res_3d=self.res_3d, seqlen=self.seqlen, refcam_id=self.refcam_id, 
                get_tids=self.get_tids, temporal_aug=self.temporal_aug, 
                use_radar_filters=False, do_shuffle_cams=self.do_shuffle_cams
            )

            self.val_dataset = VizData(
                self.nusc, self.dataroot, is_train=False, data_aug_conf=self.data_aug_conf,
                nsweeps=self.nsweeps, centroid=self.centroid, bounds=self.bounds, 
                res_3d=self.res_3d, seqlen=self.seqlen, refcam_id=self.refcam_id, 
                get_tids=self.get_tids, temporal_aug=False, use_radar_filters=False, 
                do_shuffle_cams=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.cfg_loaders.train.batch_size, shuffle=self.cfg_loaders.train.shuffle,
            num_workers=self.cfg_loaders.train.num_workers, drop_last=self.cfg_loaders.train.drop_last, pin_memory=self.cfg_loaders.train.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.cfg_loaders.val.batch_size, shuffle=self.cfg_loaders.val.shuffle,
            num_workers=self.cfg_loaders.val.num_workers, drop_last=self.cfg_loaders.val.drop_last, pin_memory=self.cfg_loaders.val.pin_memory
        )

    def test_dataloader(self):
        # Define a test dataset if needed
        pass


@hydra.main(config_path="/data/karthik/bev_perception/configs/datamodule", config_name="nuscenes")
def main(cfg: DictConfig):
    # Initialize the NuScenesDataModule with Hydra config
    data_module = NuScenesDataModule(
        datasets=cfg.datasets, 
        transforms=cfg.transforms, 
        loaders=cfg.loaders
    )

    # Prepare data and setup the datamodule
    data_module.prepare_data()
    data_module.setup(stage='fit')

    # Get the train dataloader
    train_loader = data_module.train_dataloader()

    # Get a batch from the dataloader
    batch = next(iter(train_loader))

    # Assuming the batch contains images as the first element
    images = batch[0].squeeze(1)  # Adjust this depending on how the data is returned by the dataloader

    # Plot 6 multiple view images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(6):
        img = images[0, i].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()