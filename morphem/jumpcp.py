from typing import List, Union

import numpy as np
import pandas as pd
import torch
import os
from omegaconf import ListConfig
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from typing import Union
import cv2
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Any, Optional
import h5py
from torchvision import transforms
from torch import nn

def load_meta_data(base_path: str):
    PLATE_TO_ID = {"BR00116991": 0}
    FIELD_TO_ID = dict(zip([str(i) for i in range(1, 10)], range(9)))
    WELL_TO_ID = {}
    for i in range(16):
        for j in range(1, 25):
            well_loc = f"{chr(ord('A') + i)}{j:02d}"
            WELL_TO_ID[well_loc] = len(WELL_TO_ID)

    WELL_TO_LBL = {}

    PLATE_MAP = {
        "compound": f"{base_path}/JUMP-Target-1_compound_platemap.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_platemap.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_platemap.tsv",
    }
    META_DATA = {
        "compound": f"{base_path}/JUMP-Target-1_compound_metadata.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_metadata.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_metadata.tsv",
    }

    for perturbation in PLATE_MAP.keys():
        df_platemap = pd.read_parquet(PLATE_MAP[perturbation])
        df_metadata = pd.read_parquet(META_DATA[perturbation])
        df = df_metadata.merge(df_platemap, how="inner", on="broad_sample")

        if perturbation == "compound":
            target_name = "target"
        else:
            target_name = "gene"

        codes, uniques = pd.factorize(df[target_name])
        codes += 1  # set none (neg control) to id 0
        assert min(codes) == 0
        # print(f"...{target_name} has {len(uniques)} unique values")
        WELL_TO_LBL[perturbation] = dict(zip(df["well_position"], codes))

    return PLATE_TO_ID, FIELD_TO_ID, WELL_TO_ID, WELL_TO_LBL


class JUMPCP(Dataset):
    """JUMPCP dataset"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None
    NUM_TOTAL_CHANNELS = 8

    def __init__(
        self,
        path: str,
        split: str,  # train, valid or test
        transform,
        channels: Union[List[int], None],
        use_hdf5: bool = False,
        channel_mask: bool = False,
        scale: float = 1,
        perturbation_list: ListConfig[str] = ["compound"],
        cyto_mask_path_list: ListConfig[str] = None,
    ) -> None:
        """Initialize the dataset."""
        self.root_dir = path + "/" if path[-1] != "/" else path
        self.use_hdf5 = use_hdf5

        if cyto_mask_path_list is None:
            cyto_mask_path_list = [os.path.join(self.root_dir, "jumpcp/BR00116991.pq")]
        # read the cyto mask df
        df = pd.concat([pd.read_parquet(path) for path in cyto_mask_path_list], ignore_index=True)
        df = self.get_split(df, split)

        self.data_path = list(df["path"])
        self.data_id = list(df["ID"])
        self.well_loc = list(df["well_loc"])

        #self.file = h5py.File(os.path.join(self.root_dir, f"jumpcp/jumpcp_{split}.h5"), "r")
        assert len(perturbation_list) == 1
        self.perturbation_type = perturbation_list[0]

        if type(channels[0]) is str:
            # channel is separated by hyphen
            self.channels = torch.tensor([int(c) for c in channels[0].split("-")])
        else:
            self.channels = torch.tensor([c for c in channels])
        if scale is None and channel_mask:
            self.scale = float(self.NUM_TOTAL_CHANNELS) / len(self.channels)
        else:
            self.scale = scale  # scale the input to compensate for input channel masking

        if self.scale != 1:
            print(f"------ Scaling the input to compensate for channel masking, scale={self.scale} ------")

        # print(f"------ channels: {self.channels.numpy()} ------")

        self.transform = transform

        meta_data_path = os.path.join(self.root_dir, "jumpcp/platemap_and_metadata")
        self.plate2id, self.field2id, self.well2id, self.well2lbl = load_meta_data(meta_data_path)

        self.channel_mask = channel_mask

    def get_split(self, df, split_name, seed=0):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(0.6 * m)
        validate_end = int(0.2 * m) + train_end

        if split_name == "train":
            return df.iloc[perm[:train_end]]
        elif split_name == "valid":
            return df.iloc[perm[train_end:validate_end]]
        elif split_name == "test":
            return df.iloc[perm[validate_end:]]
        else:
            raise ValueError("Unknown split")

    def __getitem__(self, index):
        if self.well_loc[index] not in self.well2lbl[self.perturbation_type]:
            # this well is not labeled
            return None
        ## EDIT: use local img
        img_path = self.data_path[index].replace("s3://insitro-research-2023-context-vit/", self.root_dir)

        ## read npy img
        if self.use_hdf5:  ## one big file storing all images, to reduce number of files
            img_name = os.path.basename(img_path)
            img_chw = np.array(self.file[img_name])
        else:  ## each image is stored in a separate numpy file
            img_chw = np.load(img_path)

        if img_chw is None:
            return None

        img_hwc = img_chw.transpose(1, 2, 0)
        img_chw = self.transform(img_hwc)

        channels = self.channels.numpy()

        assert type(img_chw) is not list, "Only support jumpcp for supervised training"

        if self.scale != 1:
            # scale the image pixels to compensate for the masked channels
            # used in inference
            img_chw *= self.scale

        # mask out channels
        if self.channel_mask:
            # mask out unselected channels by setting their pixel values to 0
            unselected = [c for c in range(len(img_chw)) if c not in channels]
            img_chw[unselected] = 0
        else:
            img_chw = img_chw[channels]

        return {
            "image": img_chw,
            # "channels": channels,
            "label": self.well2lbl[self.perturbation_type][self.well_loc[index]],
        }

    def __len__(self) -> int:
        return len(self.data_path)

    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)

class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        """
        Initialize the SaturationNoiseInjector module.
        
        Parameters:
            low (int): Lower bound for uniform noise values.
            high (int): Upper bound for uniform noise values.
        """
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply high-intensity noise injection to saturated pixels in a single-channel image.
        The function expects the input tensor to have the shape (1, H, W) with pixel intensities in the 0-255 range.

        Process:
          - Convert the input tensor to float32.
          - Generate noise drawn uniformly from [low, high] for each pixel.
          - Create a mask for saturated pixels (where the pixel value equals 255).
          - Zero-out saturated pixels and add the masked noise.

        Parameters:
            x (torch.Tensor): Input tensor of shape (1, H, W).
        
        Returns:
            torch.Tensor: The processed tensor with noise injected.
        """
        # Ensure input is in floating point for correct arithmetic
        # Since x has one channel, extract the channel as a 2D tensor (H, W)
        channel = x[0]
        
        # Generate noise with values uniformly drawn between self.low and self.high
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        
        # Create a mask of pixels that are saturated (value == 255)
        mask = (channel == 255).float()
        
        # Apply the mask to the noise to affect only the saturated pixels
        noise_masked = noise * mask
        
        # Remove the saturated pixels by setting them to zero
        channel[channel == 255] = 0
        
        # Add the masked noise to the channel
        channel = channel + noise_masked
        
        # Update the tensor with the modified channel
        x[0] = channel
        
        return x


class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        # We initialize with num_features=1, but we’ll replace it on-the-fly if needed.
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,             # Temporary placeholder
            affine=False,               # No learnable parameters
            track_running_stats=False,  # Use per-forward stats (no running mean)
            eps=self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, C, H, W)
        We'll ensure that our instance_norm has the correct number of channels (C).
        """
        # If your input has a dynamic channel size, we need to re-initialize:
        C, _, _ = x.shape
        if self.instance_norm.num_features != C:
            self.instance_norm = nn.InstanceNorm2d(
                num_features=C,
                affine=False,
                track_running_stats=False,
                eps=self.eps
            )

        # Now we can pass x through our InstanceNorm2d layer
        return self.instance_norm(x)

def normalize_scale_for_test(im):
    sizes = {160:160, 238:238, 512:512}
    t = transforms.functional.center_crop(im, sizes[im.shape[-2]])
    t = transforms.functional.resize(t, (224,224))
    return t


def get_jumpcp_dataloaders(
    root_dir: str,
    image_size: Tuple[int, int],
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    channels: Dict[str, List[int]],
    normalization: Dict[str, List[float]] | None,
    use_hdf5: bool = False,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if normalization is None:
        mean_data = None
        std_data = None
    else:
        mean_data = normalization["mean"]
        std_data = normalization["std"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        SaturationNoiseInjector(),
        PerImageNormalize()])

    train_set = JUMPCP(path=root_dir, split="train", transform=transform, channels=channels["training"], use_hdf5=use_hdf5)
    valid_set = JUMPCP(path=root_dir, split="valid", transform=transform, channels=channels["validation"], use_hdf5=use_hdf5)
    test_set = JUMPCP(path=root_dir, split="test", transform=transform, channels=channels["test"], use_hdf5=use_hdf5)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        valid_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, val_loader, test_loader

