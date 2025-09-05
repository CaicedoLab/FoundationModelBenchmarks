import torch
from torch import nn
from models.vision_transformer import vit_base, vit_small
from models.models_mae import mae_vit_base_patch16, mae_vit_small_patch16
from models.utils import create_pad
import numpy as np
import os
import json
from omegaconf import OmegaConf

from models.dinov2.dinov2.configs.config import Dinov2Config
from models.dinov2.dinov2.models import build_model_from_cfg
from models.dinov2.dinov2.utils.utils import load_pretrained_weights

import models.channel_vit_dino.vision_transformer as channelvit 

class DinoV2Models(torch.nn.Module):
    def __init__(self, model_path, checkpoint, model_size, device=None):
        super().__init__()
        
        if model_size == 'ngram':
            self.is_ngram = True
        else:
            self.is_ngram = False 
        
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
            
            possible_final_check = list(filter(lambda x: "final_model" in x, checkpoint_dirs))#["final_model" in checkpoint_dir for checkpoint_dir in checkpoint_dirs]
            if len(possible_final_check) > 0:
                checkpoint_path = os.path.join(eval_dir, possible_final_check[0], "teacher_checkpoint.pth")
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
        self.model, _ = build_model_from_cfg(cfg, only_teacher=True) # type: ignore
        load_pretrained_weights(self.model, checkpoint_path, 'teacher')
        self.model.eval()
        if device is not None:
            self.model.to(device)
        self.feature_file = "pretrained_dinov2_vit_features.npy"

    def boc_ngram(self, samples: torch.Tensor):
        entries = []
        for ch_idx in range(samples.shape[1]):
            entries.append((ch_idx, ch_idx))
            
        to_concat = []
        for entry in entries:
            ngram = torch.stack([samples[:,entry[0],:,:], samples[:,entry[1],:,:]], dim=1)
            to_concat.append(self.model(ngram))
        
        return torch.concat(to_concat, dim=1)
    
    def boc(self, samples: torch.Tensor):
        to_concat = []
        for ch_idx in range(samples.shape[1]):
            to_concat.append(self.model(samples[:,ch_idx,:,:].unsqueeze(dim=1)))
        return torch.concat(to_concat, dim=1)

    def all_cat(self, samples: torch.Tensor):
        entries = []
        for ch_idx in range(samples.shape[1]):
            for ch_idx2 in range(samples.shape[1]):
                entries.append((ch_idx, ch_idx2))
            
        to_concat = []
        for entry in entries:
            ngram = torch.stack([samples[:,entry[0],:,:], samples[:,entry[1],:,:]], dim=1)
            to_concat.append(self.model(ngram).cpu().detach())
        
        return torch.concat(to_concat, dim=1)

    def average_of_diagonal(self, samples: torch.Tensor):
        entries = set()
        for ch_idx in range(samples.shape[1]):
            for ch_idx2 in range(samples.shape[1]):
                if (ch_idx2, ch_idx) in entries:
                    continue
                
                entries.add((ch_idx, ch_idx2))
        
        to_concat = []
        for entry in entries:
            if entry[0] == entry[1]:
                ngram = torch.stack([samples[:,entry[0],:,:], samples[:,entry[1],:,:]], dim=1)
                to_concat.append(self.model(ngram).cpu().detach())
            else:
                ngram = torch.stack([samples[:,entry[0],:,:], samples[:,entry[1],:,:]], dim=1)
                ngram_embed = self.model(ngram).cpu().detach()
                
                ngram_rev = torch.stack([samples[:,entry[1],:,:], samples[:,entry[0],:,:]], dim=1)
                ngram_rev_embed = self.model(ngram_rev).cpu().detach()
                
                to_concat.append((ngram_embed + ngram_rev_embed)/2)
        
        return torch.concat(to_concat, dim=1)

    def forward(self, samples: torch.Tensor):
        # return nn.functional.normalize(self.model(samples), dim=1, p=2
        if self.device:
            samples = samples.to(self.device)
        if not self.is_ngram:
            return self.boc(samples).cpu().detach().numpy()
        else:
            return self.average_of_diagonal(samples).cpu().detach().numpy()

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
        self.dataset_channels = None # will be a list 
        self.model_path = model_path
        
        with open(os.path.join(os.path.dirname(model_path), 'channel_map.json'), 'r') as f:
            channel_map_file = f.read()
        self.channel_map = json.loads(channel_map_file)

        self.feature_file = "pretrained_vit_features.npy"
        # Create model with in_chans=1 to match training setup
        if model_size == "base":
            self.model = channelvit.channelvit_base(in_chans=len(self.channel_map))
        elif model_size == "small":
            self.model = channelvit.channelvit_small(in_chans=len(self.channel_map))
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
    
    def set_dataset(self, dataset_name, model_path):
        if dataset_name == "Allen":
            if '_75ds' in model_path or '_10ds' in model_path:
                self.dataset_channels = ['nucleus', 'cell body', 'protein']
            else:
                self.dataset_channels = ['nucleus', 'membrane', 'protein']
        elif dataset_name == "CP":
            if '_75ds' in model_path or '_10ds' in model_path:
                self.dataset_channels = ['nucleus', 'endoplasmic reticulum', 'RNA', 'golgi body', 'mitochondria']
            else:
                self.dataset_channels = ['nucleus', 'cp2', 'er', 'cp4', 'cp5']
        elif dataset_name == "HPA":
            if '_75ds' in model_path or '_10ds' in model_path:
                self.dataset_channels = ['microtubules', 'protein', 'nucleus', 'endoplasmic reticulum']
            else:
                self.dataset_channels = ['microtubules', 'protein', 'nucleus', 'er']
        else:
            raise ValueError("Dataset name supplied is not supported. This class only supports CHAMMIv1 benchmarking.")
    
    def __call__(self, images):
        channel_ids = [[self.channel_map[chan] if chan in self.dataset_channels else 0 for chan in self.dataset_channels]] * len(images)
        channel_masks = [[True for _ in range(images.shape[1])]]*len(images)
        with torch.no_grad():
            images = images.to(self.device)
            return self.model(images, channel_ids, channel_masks).cpu().detach().numpy()