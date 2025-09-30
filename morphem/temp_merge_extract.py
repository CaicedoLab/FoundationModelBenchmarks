import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import torch

from models.models import *
from utils import *
from models.utils import NoiseInjection, self_normalize

from tqdm import tqdm

import zarr
from parcha.models.transformer import small_transformer


def main():
    out_dir, root_dir, model_path, model_check, batch_size = parse_args()
    store = zarr.storage.LocalStore('/scr/jpeters/parcha/648c3cc_ngram_no_dynamic_sampling.zarr', read_only=True)
    root = zarr.open(store=store)
    zarr_map = {}
    for group in root.group_keys():
        map = root.get(group).attrs['image_index_map']
        json_map = json.loads(map)
        zarr_map.update(json_map)
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    
    dataset_names = ["Allen", "CP", "HPA"]
    
    model = get_model(model_path, model_check, None).to(device)
    
    for dataset_name in dataset_names:
        dataset = configure_dataset(root_dir, dataset_name, target_labels="file_path")
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        train_dataloader = tqdm(train_dataloader, desc=f"Processing {dataset_name}", total=len(train_dataloader))
        
        all_feat = []
        for _, file_path in  enumerate(train_dataloader):
            embeddings = []
            for path in file_path:
                zarr_idx, group = zarr_map[path]
                features = root[group][group][zarr_idx]
                features = torch.Tensor(features).unsqueeze(0)
                embeddings.append(features)
            cat_embed = torch.cat(embeddings, dim=0).to(device)
            embeds = model(cat_embed).detach().cpu().numpy()
            all_feat.append(embeds)
        
        all_feat = np.concatenate(all_feat)

        if all_feat.ndim == 4:
            all_feat = all_feat.squeeze(2).squeeze(2)
        elif all_feat.ndim == 3:
            all_feat = all_feat.squeeze(2)
        elif all_feat.ndim == 2:
            all_feat = all_feat.squeeze()

        feature_path = os.path.join(out_dir, dataset_name, 'pretrained_vit_features.npy')
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        np.save(feature_path, all_feat)
        torch.cuda.empty_cache()  # new line
        
def get_model(model_path, model_check, checkpoint):
    if model_check == 'ngram':
        model = small_transformer(max_channels=16)
    else:
        model = small_transformer()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    return model

def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    root_dir = path_expansion(args.root_dir)
    out_dir = path_expansion(args.out_dir)
    model_path = path_expansion(args.model_path)
    
    return out_dir, root_dir, model_path, args.model, args.batch_size

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        help="The root directory of the original images",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="The directory which to store output within",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The type of model that is being trained and evaluated (convnext, resnet, or vit)",
        required=True,
        choices=["dinov2", 'ngram'],
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="The path to the checkpoint that contains the model weights.",
        required=True,
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Select a batch size that works for your gpu size",
        required=True,
    )


    return parser


if __name__ == "__main__":
    main()
