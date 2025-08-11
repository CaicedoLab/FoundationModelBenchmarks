import sys
sys.path.append('./FoundationModels/dinov2/') # for internal import of dinov2 modules to work

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import argparse
import torch
import os

from models.models import *
from utils import *
from models.utils import SaturationNoiseInjector, PerImageNormalize

def main(
    feature_dir, 
    root_dir, 
    model_path, 
    model_check, 
    model_size, 
    gpu, 
    batch_size,
    checkpoint
):
    dataset_names = ["Allen", "CP", "HPA"]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = get_model(model_path, model_check, model_size, device)
    for dataset_name in dataset_names:
        transform = transforms.Compose([SaturationNoiseInjector(), PerImageNormalize()])
        dataset = configure_dataset(root_dir, dataset_name, transform=transform)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_steps = len(train_dataloader)
        all_feat = []
        for index, (images, label) in tqdm(enumerate(train_dataloader), total=total_steps):
            all_feat.append(model(images))

        all_feat = np.concatenate(all_feat)

        if all_feat.ndim == 4:
            all_feat = all_feat.squeeze(2).squeeze(2)
        elif all_feat.ndim == 3:
            all_feat = all_feat.squeeze(2)
        elif all_feat.ndim == 2:
            all_feat = all_feat.squeeze()

        feature_path = feature_path = f"{feature_dir}/{dataset_name}/{model.feature_file}"
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        np.save(feature_path, all_feat)
        torch.cuda.empty_cache()  # new line

def get_model(model_path, model_check, model_size, device):
    if model_check == 'dinov2':
        return DinoV2Models()
    elif model_check == "mae":
        return MAEModel(model_path, model_size, device)
    elif model_check == 'dinov1':
        return ViTClass(model_path, model_size, device)
    elif model_check == 'channelvit':
        return ChannelVIT(model_path, model_size, device)
    else:
        raise NotImplementedError(f"Given {model_check} has not been implemented yet. Implement it for evaluation")

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
        choices=["mae", "resnet", "dinov1", "dinov2", 'channelvit'],
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
    
    main(
        feat_dir,
        root_dir,
        model_path,
        args.model,
        args.model_size,
        args.gpu,
        args.batch_size,
        args.checkpoint
    )
