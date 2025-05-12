import torch
from torch.utils.data import DataLoader
from torch import nn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
from torchvision import transforms
import importlib
from tqdm import tqdm

from torchvision.transforms import v2
from vision_transformer import vit_small, vit_base
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import sys

import argparse
import torch
import sys



import folded_dataset
# reload(folded_dataset)
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


def configure_dataset(root_dir, dataset_name, transform=None):
    df_path = f'{root_dir}/{dataset_name}/enriched_meta.csv'
    df = pd.read_csv(df_path)
    dataset = folded_dataset.SingleCellDataset(csv_file=df_path, root_dir=root_dir, target_labels='train_test_split', transform=transform)
    return dataset

import torch
import timm  # If you used timm to define your ViT

class ViTClass():
    def __init__(self, gpu):
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

        # Create model with in_chans=1 to match training setup
        self.model = vit_small()
        remove_prefixes = ["module.backbone.", "module.", "module.head."]

        # Load model weights
        student_model = torch.load("/scr/vidit/Foundation_Models/model_weights/Dino_Small_75ds_Guided/checkpoint.pth")['student']
        # Remove unwanted prefixes
        cleaned_state_dict = {}
        for k, v in student_model.items():
            new_key = k
            for prefix in remove_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]  # Remove prefix
            if not new_key.startswith("head.mlp") and not new_key.startswith("head.last_layer"):
                cleaned_state_dict[new_key] = v  # Keep only valid keys
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

    def get_model(self):
        return self.model


def create_pad(images, patch_width, patch_height): # new method for vit model
    N, C, H, W = images.shape

    new_width = ((W + patch_width - 1) // patch_width) * patch_width
    pad_width = new_width - W

    # Calculate padding amounts for left and right
    pad_left = pad_right = pad_width // 2
    
    if pad_width % 2 != 0:
        pad_right += 1


    new_height = ((H + patch_height - 1) // patch_height) * patch_height
    pad_height = new_height - H
    
    # Calculate padding amounts for top and bottom
    pad_top = pad_bottom = pad_height // 2
    
    if pad_height % 2 != 0:
        pad_bottom += 1
        

    padded_images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return padded_images


def get_save_features(feature_dir, root_dir, model_check, gpu, batch_size):
    dataset_names = ['Allen', 'CP', 'HPA']
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    

    vit_instance = ViTClass(gpu) 
    vit_model = vit_instance.get_model() 
    feature_file = 'pretrained_vit_features.npy'
        
        
    for dataset_name in dataset_names:
        # Post crops and processing getting the transforms
        transform = transforms.Compose([
            PerImageNormalize()])
        dataset = configure_dataset(root_dir, dataset_name, transform=transform)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_feat = []
        for images, label in tqdm(train_dataloader, total=len(train_dataloader)):
            
            if model_check == "vit": # new line of code
                patch_embed = vit_model.patch_embed

                # Access the Conv2d layer within PatchEmbed
                conv_layer = patch_embed.proj
                
                # Extract kernel size (patch size)
                patch_size = conv_layer.kernel_size
                patch_height, patch_width = patch_size
                images = create_pad(images, patch_width, patch_height)

            
            cloned_images = images.clone()
            batch_feat = []
            for c in range(cloned_images.shape[1]):
                # Copy each channel three times
                single_channel = images[:, c, :, :].unsqueeze(1).to(device)
    
                output = vit_model.forward_features((single_channel).to(device))
                feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()

                batch_feat.append(feat_temp)
                
            batch_feat = np.concatenate(batch_feat, axis=1)
            all_feat.append(batch_feat)
       
        all_feat = np.concatenate(all_feat)

        if all_feat.ndim == 4:
            all_feat = all_feat.squeeze(2).squeeze(2)
        elif all_feat.ndim == 3:
            all_feat = all_feat.squeeze(2)
        elif all_feat.ndim == 2:
            all_feat = all_feat.squeeze()

        
        feature_path = feature_path = f'{feature_dir}/{dataset_name}/{feature_file}'
        np.save(feature_path, all_feat)
        torch.cuda.empty_cache() # new line
        


def get_parser():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="The root directory of the original images", required=True)
    parser.add_argument("--feat_dir", type=str, help="The directory that contains the features", required=True)
    parser.add_argument("--model", type=str, help="The type of model that is being trained and evaluated (convnext, resnet, or vit)", required=True, choices=['convnext', 'resnet', 'vit'])
    parser.add_argument("--gpu", type=int, help="The gpu that is currently available/not in use", required=True)
    parser.add_argument("--batch_size", type=int, default=64, help="Select a batch size that works for your gpu size", required=True)
    
    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    root_dir = args.root_dir
    feat_dir = args.feat_dir
    model = args.model
    gpu = args.gpu
    batch_size = args.batch_size

    get_save_features(feat_dir, root_dir, model, gpu, batch_size)

