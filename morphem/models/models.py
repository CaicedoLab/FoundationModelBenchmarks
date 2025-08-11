import torch
from torch import nn
from models.vision_transformer import vit_base, vit_small
from models.models_mae import mae_vit_base_patch16, mae_vit_small_patch16
from models.utils import create_pad
import numpy as np
import os
from omegaconf import OmegaConf

from FoundationModels.dinov2 import dinov2 as dinov2
from FoundationModels.dinov2.dinov2.configs.config import Dinov2Config
from FoundationModels.dinov2.dinov2.models import build_model_from_cfg
from FoundationModels.dinov2.dinov2.utils.utils import load_pretrained_weights

import models.channel_vit_dino.vit as channelvit 

class DinoV2Models(torch.nn.Module):
    def __init__(self, model_path, checkpoint, device):
        super().__init__()
        
        self.device = device
        
        eval_dir = os.path.join(model_path, 'eval')
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
                checkpoint_path = os.path.join(eval_dir, f"{checkpoint}", "teacher_checkpoint.pth")
            else:
                raise ValueError("Checkpoint not found, please check if it's a valid checkpoint.")
        print(f"Running with model gathered from: {checkpoint_path}")

        config_path = os.path.join(model_path, 'config.yaml')        
        default_cfg = OmegaConf.create(Dinov2Config())
        with open(config_path, 'r') as f:
            cfg = OmegaConf.load(f)
        cfg = OmegaConf.merge(default_cfg, cfg)
        dinov2_model, _ = build_model_from_cfg(cfg, only_teacher=True) # type: ignore
        load_pretrained_weights(dinov2_model, checkpoint_path, 'teacher')
        dinov2_model.eval()
        dinov2_model.to(device)
        self.feature_file = "pretrained_dinov2_vit_features.npy"

    def forward(self, samples):
        # return nn.functional.normalize(self.model(samples), dim=1, p=2)
        return self.model(samples)

class ViTClass:
    def __init__(self, weights_path: str, model_size: str, device):
        self.device = device
        self.feature_file = "pretrained_vit_features.npy"
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
    
    def __call__(self, images):
        patch_embed = self.model.patch_embed
        conv_layer = patch_embed.proj
        patch_size = conv_layer.kernel_size
        patch_height, patch_width = patch_size
        images = create_pad(images, patch_width, patch_height)

        cloned_images = images.clone()
        batch_feat = []
        for c in range(cloned_images.shape[1]):
            single_channel = images[:, c, :, :].unsqueeze(1).to(self.device)

            output = self.model.forward_features((single_channel).to(self.device))
            feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
            
            batch_feat.append(feat_temp)

        return np.concatenate(batch_feat, axis=1)
        

class MAEModel:
    def __init__(self, device, weights_path, model_size):
        self.device = device
        self.feature_file = "pretrained_mae_features.npy"
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

    def __call__(self, images):
        patch_embed = self.model.patch_embed
        conv_layer = patch_embed.proj
        patch_size = conv_layer.kernel_size
        patch_height, patch_width = patch_size
        images = create_pad(images, patch_width, patch_height)
        
        cloned_images = images.clone()
        batch_feat = []
        for c in range(cloned_images.shape[1]):
            single_channel = images[:, c, :, :].unsqueeze(1).to(self.device)

            feat_temp = (
                self.model.get_features((single_channel).to(self.device))
                .cpu()
                .detach()
                .numpy()
            )
            
            batch_feat.append(feat_temp)
            
        return np.concatenate(batch_feat, axis=1)
    
class ChannelVIT:
    def __init__(self, model_path, model_size, device):
        self.device = device
        self.feature_file = "pretrained_vit_features.npy"
        # Create model with in_chans=1 to match training setup
        if model_size == "base":
            self.model = channelvit.vit_base()
        elif model_size == "small":
            self.model = channelvit.vit_small()
        else:
            raise ValueError(
                f"Models of base and small are supported, not {model_size}"
            )

        remove_prefixes = ["module.backbone.", "module.", "module.head."]

        # Load model weights
        student_model = torch.load(model_path, weights_only=False)["student"]
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
    
    def __call__(self, images):
        return None