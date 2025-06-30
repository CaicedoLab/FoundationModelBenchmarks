import sys
sys.path.append('./morphem/FoundationModels/dinov2/') # for internal import of dinov2 modules to work

import torch
from torch.utils.data import DataLoader
from torch import nn

import pandas as pd
import numpy as np
from torchvision import transforms
from tqdm import tqdm

from vision_transformer import vit_small, vit_base
import torch.nn.functional as F
from models_mae import mae_vit_base_patch16, mae_vit_small_patch16
import folded_dataset

import argparse
import torch
import os
import glob
import re

from omegaconf import OmegaConf
from FoundationModels.dinov2.dinov2.configs import dinov2_default_config
from FoundationModels.dinov2.dinov2.models import build_model_from_cfg
from FoundationModels.dinov2.dinov2.utils.utils import load_pretrained_weights


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
            num_features=1,  # Temporary placeholder
            affine=False,  # No learnable parameters
            track_running_stats=False,  # Use per-forward stats (no running mean)
            eps=self.eps,
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
                num_features=C, affine=False, track_running_stats=False, eps=self.eps
            )

        # Now we can pass x through our InstanceNorm2d layer
        return self.instance_norm(x)


def configure_dataset(root_dir, dataset_name, transform=None):
    df_path = f"{root_dir}/{dataset_name}/enriched_meta.csv"
    df = pd.read_csv(df_path)
    dataset = folded_dataset.SingleCellDataset(
        csv_file=df_path,
        root_dir=root_dir,
        target_labels="train_test_split",
        transform=transform,
    )
    return dataset


def path_expansion(path: str):
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))

class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        # return nn.functional.normalize(self.model(samples), dim=1, p=2)
        return self.model(samples)

class ViTClass:
    def __init__(self, weights_path: str, model_size: str, gpu: int = 0):
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

        # Create model with in_chans=1 to match training setup
        if model_size == "base":
            self.model = vit_small()
        elif model_size == "small":
            self.model = vit_base()
        else:
            raise ValueError(
                f"Models of base and small are supported, not {model_size}"
            )

        remove_prefixes = ["module.backbone.", "module.", "module.head."]

        # Load model weights
        student_model = torch.load(weights_path)["student"]
        # Remove unwanted prefixes
        cleaned_state_dict = {}
        for k, v in student_model.items():
            new_key = k
            for prefix in remove_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]  # Remove prefix
            if not new_key.startswith("head.mlp") and not new_key.startswith(
                "head.last_layer"
            ):
                cleaned_state_dict[new_key] = v  # Keep only valid keys
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

    def get_model(self):
        return self.model


class MAEModel:
    def __init__(self, gpu, weights_path, model_size):
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        if model_size == "small":
            self.model = mae_vit_small_patch16()
        elif model_size == "base":
            self.model = mae_vit_base_patch16()
        else:
            raise ValueError(
                f"Only small and base sized models are supported, not {model_size}"
            )

        state_dict = torch.load(
            weights_path,
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model.eval()
        self.model.to(self.device)

    def get_model(self):
        return self.model


def create_pad(images, patch_width, patch_height):  # new method for vit model
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

    padded_images = F.pad(
        images, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )

    return padded_images


def get_save_features(
    feature_dir, 
    root_dir, 
    model_weights, 
    model_check, 
    model_size, 
    gpu, 
    batch_size,
    checkpoint
):
    
    dataset_names = ["Allen", "CP", "HPA"]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    mae_model, vit_model, dinov2_model = None, None, None
    if model_check == 'dinov2':
        eval_dir = os.path.join(model_weights, 'eval')
        checkpoint_dirs = os.listdir(eval_dir)
        if checkpoint == "auto":
            check_iterations = []
            for check_dir in checkpoint_dirs:
                if "final_model" not in check_dir:
                    check_val = int(check_dir.split('_')[-1])
                    check_iterations.append(check_val)
            
            latest_eval = max(check_iterations)
            checkpoint_path = os.path.join(eval_dir, f"training_{latest_eval}", "teacher_checkpoint.pth")
            
            possible_final_check = ["final_model" in checkpoint_dir for checkpoint_dir in checkpoint_dirs]
            if any(possible_final_check):
                checkpoint_path = os.path.join(eval_dir, "final_model", "teacher_checkpoint.pth")
        else:
            has_checkpoint = any([checkpoint in checkpoint_dir for checkpoint_dir in checkpoint_dirs])
            if has_checkpoint:
                checkpoint_path = os.path.join(eval_dir, f"training_{checkpoint}", "teacher_checkpoint.pth")
            else:
                raise ValueError("Checkpoint not found, please check if it's a valid checkpoint.")
        print(f"Running with model gathered from: {checkpoint_path}")
        
        config_path = os.path.join(model_weights, 'config.yaml')        
        default_cfg = OmegaConf.create(dinov2_default_config)
        with open(config_path, 'r') as f:
            cfg = OmegaConf.load(f)
        cfg = OmegaConf.merge(default_cfg, cfg)
        dinov2_model, _ = build_model_from_cfg(cfg, only_teacher=True) # type: ignore
        load_pretrained_weights(dinov2_model, checkpoint_path, 'teacher')
        dinov2_model.eval()
        dinov2_model.cuda()
        dinov2_model = ModelWithNormalize(dinov2_model).to(device)        
        feature_file = "pretrained_dinov2_vit_features.npy"
    elif model_check == "mae":
        mae_instance = MAEModel(gpu, model_weights, model_size)
        mae_model = mae_instance.get_model()
        feature_file = "pretrained_mae_features.npy"
    elif model_check == 'dinov1':
        vit_instance = ViTClass(model_weights, model_size, gpu)
        vit_model = vit_instance.get_model()
        feature_file = "pretrained_vit_features.npy"
    else:
        raise NotImplementedError(f"Given {model_check} has not been implemented yet. Implement it for evaluation")

    for dataset_name in dataset_names:
        # Post crops and processing getting the transforms
        transform = transforms.Compose([SaturationNoiseInjector(), PerImageNormalize()])
        dataset = configure_dataset(root_dir, dataset_name, transform=transform)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_steps = len(train_dataloader)
        all_feat = []
        for index, (images, label) in tqdm(enumerate(train_dataloader), total=total_steps):
            if vit_model:  
                patch_embed = vit_model.patch_embed
                conv_layer = patch_embed.proj
                patch_size = conv_layer.kernel_size
                patch_height, patch_width = patch_size
                images = create_pad(images, patch_width, patch_height)
            elif mae_model:
                patch_embed = mae_model.patch_embed
                conv_layer = patch_embed.proj
                patch_size = conv_layer.kernel_size
                patch_height, patch_width = patch_size
                images = create_pad(images, patch_width, patch_height)

            cloned_images = images.clone()
            batch_feat = []
            for c in range(cloned_images.shape[1]):
                single_channel = images[:, c, :, :].unsqueeze(1).to(device)

                if vit_model:
                    output = vit_model.forward_features((single_channel).to(device))
                    feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
                elif mae_model:
                    feat_temp = (
                        mae_model.get_features((single_channel).to(device))
                        .cpu()
                        .detach()
                        .numpy()
                    )
                elif dinov2_model:
                    output = dinov2_model(single_channel.to(device))
                    feat_temp = output.float().cpu().detach().numpy()
                else:
                    raise ValueError("Either a vit or mae model must be provided")

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

        feature_path = feature_path = f"{feature_dir}/{dataset_name}/{feature_file}"
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        np.save(feature_path, all_feat)
        torch.cuda.empty_cache()  # new line


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        help="The root directory of the original images",
        required=True,
    )
    parser.add_argument(
        "--feat-dir",
        type=str,
        help="The directory that contains the features",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The type of model that is being trained and evaluated (convnext, resnet, or vit)",
        required=True,
        choices=["mae", "resnet", "dinov1", "dinov2"],
    )
    parser.add_argument(
        "--model-size",
        type=str,
        help="The size of model that is evaluated",
        required=True,
        choices=["small", "base"],
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="The path to the checkpoint that contains the model weights.",
        required=True,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="The gpu that is currently available/not in use",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Select a batch size that works for your gpu size",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="auto", # type latest for the script to calculate it...
        help="What checkpoint should be evaluated for dinov2. This should be the number in training_number.",
    )

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    root_dir = path_expansion(args.root_dir)
    feat_dir = path_expansion(args.feat_dir)
    model_path = path_expansion(args.model_path)
    
    get_save_features(
        feat_dir,
        root_dir,
        model_path,
        args.model,
        args.model_size,
        args.gpu,
        args.batch_size,
        args.checkpoint
    )
