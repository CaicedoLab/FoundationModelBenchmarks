import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import torch

from models.models import *
from utils import *
from models.utils import NoiseInjection, self_normalize

from sklearn.preprocessing import normalize

from lightning import Fabric
import torch.distributed as dist

from tqdm import tqdm

import zarr

def main():
    out_dir, root_dir, model_path, model_check, batch_size = parse_args()
    
    torch.set_float32_matmul_precision('high')
    
    fabric = Fabric()
    
    dataset_names = ["Allen", "CP", "HPA"]

    model = get_model(model_path, model_check, 'auto')
    model.stack_features = True
    model = fabric.setup(model)
        
    if fabric.global_rank == 0:
        store = zarr.storage.LocalStore(out_dir, read_only=False)    
        root = zarr.create_group(store, overwrite=True)

    for dataset_name in dataset_names:
        images_added = {}
        model.dataset_name = dataset_name
        transform = transforms.Compose([transforms.ConvertImageDtype(torch.float32), NoiseInjection(), self_normalize()])
        dataset = configure_dataset(root_dir, dataset_name, transform=transform, target_labels="file_path")
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        train_dataloader = fabric.setup_dataloaders(train_dataloader)
        
        if fabric.global_rank == 0:
            train_dataloader = tqdm(train_dataloader, desc=f"Processing {dataset_name}", total=len(train_dataloader))
        embedding_shape = None
        for batch_idx, (images, file_path) in enumerate(train_dataloader):
            features = model(images)
            dist_features = fabric.all_gather(features)
            batch_paths = [None]*fabric.world_size
            dist.all_gather_object(batch_paths, file_path)

            if fabric.global_rank == 0:
                for gpu in range(dist_features.shape[0]):
                    for batch in range(dist_features.shape[1]):
                        file_name = batch_paths[gpu][batch]
                        if file_name in images_added:
                            continue 
                        else:
                            if not embedding_shape:
                                embedding_shape = dist_features[gpu, batch, :].shape
                            images_added[file_name] = dist_features[gpu, batch, :].detach().cpu().numpy()

        if fabric.is_global_zero:
            dataset_group = root.create_group(dataset_name)
            embeddings_array = dataset_group.create_array(
                dataset_name,
                shape=(len(images_added), *embedding_shape),
                chunks=(1, *embedding_shape), 
                shards=(1000, *embedding_shape),
                dtype='float32',
                compressors=None
            )        
            
            image_index_map = {}
            tensor_arrs = []
            for i, (image_name, tensor) in enumerate(images_added.items()):
                tensor_arrs.append(tensor)
                # embeddings_array[i] = tensor
                image_index_map[image_name] = (i, dataset_name)
            dataset_tensor = np.stack(tensor_arrs)
            print(f"Saving dataset {dataset_name }")
            embeddings_array[:, :, :] = dataset_tensor
            dataset_group.attrs['image_index_map'] = json.dumps(image_index_map)
            print("Finished saving.")
        fabric.barrier()
        
    if fabric.global_rank == 0:
        store.close()
        
def get_model(model_path, model_check, checkpoint):
    if model_check == 'dinov2' or model_check == 'ngram':
        return DinoV2Models(model_path, checkpoint, model_check)
    else:
        raise NotImplementedError(f"Given {model_check} has not been implemented yet. Implement it for evaluation")

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
