import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from multiprocessing import Queue, Pool, Manager
import argparse
import torch
import os

from models.models import *
from utils import *
from models.utils import SaturationNoiseInjector, PerImageNormalize

from dataclasses import dataclass

@dataclass
class ExtractionData:
    dataset_name: str
    model_path: str
    model_check: str
    model_size: str
    feature_dir: str
    root_dir: str
    batch_size: int
    
def process_dataset(gpu_queue:Queue, data: ExtractionData):
    gpu = gpu_queue.get()

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = get_model(data.model_path, data.model_check, data.model_size, device)
    
    if isinstance(model, ChannelVIT):
        model.set_dataset(data.dataset_name)
    transform = transforms.Compose([SaturationNoiseInjector(), PerImageNormalize()])
    dataset = configure_dataset(data.root_dir, data.dataset_name, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=data.batch_size, shuffle=False)

    all_feat = []
    for _, (images, _) in enumerate(train_dataloader):
        all_feat.append(model(images))

    all_feat = np.concatenate(all_feat)

    if all_feat.ndim == 4:
        all_feat = all_feat.squeeze(2).squeeze(2)
    elif all_feat.ndim == 3:
        all_feat = all_feat.squeeze(2)
    elif all_feat.ndim == 2:
        all_feat = all_feat.squeeze()

    feature_path = feature_path = f"{data.feature_dir}/{data.dataset_name}/{model.feature_file}"
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    np.save(feature_path, all_feat)
    torch.cuda.empty_cache()  # new line
    
    gpu_queue.put(gpu)


def main():
    feature_dir, root_dir, model_path, model_check, model_size, gpu, batch_size = parse_args()
        
    dataset_names = ["Allen", "CP", "HPA"]

    extraction_data = []

    for idx, dataset in enumerate(dataset_names):
        dataset_data = ExtractionData(
            feature_dir=feature_dir,
            root_dir=root_dir,
            model_path=model_path,
            model_check=model_check,
            model_size=model_size,
            batch_size=batch_size,
            dataset_name=dataset
        )

        extraction_data.append(dataset_data)
    
    with Manager() as manager:
        q = manager.Queue()
        
        for gpu_id in gpu:
            q.put(int(gpu_id))
        
        print(','.join(gpu), "GPUs being used")
        with Pool(processes=len(gpu)) as p:
            results = []
            for data in extraction_data:
                res = p.apply_async(process_dataset, args=(q, data))
                results.append(res)
                
            for res in tqdm(results, desc="Scoring..."):
                res.get() 
            
            p.close()
            p.join()
        
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

def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    root_dir = path_expansion(args.root_dir)
    feat_dir = path_expansion(args.feat_dir)
    model_path = path_expansion(args.model_path)
    
    return feat_dir, root_dir, model_path, args.model, args.model_size, args.gpu.split(','), args.batch_size

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
        type=str,
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
    main()
