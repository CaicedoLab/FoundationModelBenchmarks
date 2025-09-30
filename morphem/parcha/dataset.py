import zarr
from torch.utils.data import get_worker_info, Dataset
import polars as pl
import os
from torch import Tensor
from torch.nn.functional import one_hot
from torch import int64

class ZarrChammiFeatures(Dataset):
    def __init__(self, data_path, metadata_path, mode: str = 'train', embed_dim=384):
        """Dataset for Zarr features

        Args:
            data_path (str): Path to zarr zip of features
            metadata_path (str): Path to .csv for chammi dataset
            mode (str, optional): train or test split. Defaults to 'train'.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.data_path = os.path.abspath(os.path.expanduser(data_path))
        self.metadata_path = os.path.abspath(os.path.expanduser(metadata_path)) 
        self.store = zarr.storage.ZipStore(self.data_path, mode='r')
        self.root = zarr.open(store=self.store)
        metadata = pl.read_csv(self.metadata_path).filter(pl.col('train_test_split').eq(mode))
        self.file_names = metadata['file_path'].to_list()
        classes = metadata['Label'].cast(pl.Categorical).to_physical()
        self.num_classes = classes.max() + 1 # +1 for class label 0
        self.classes = classes.to_physical()
        
    def load_zarr(self):
        self.store = zarr.storage.ZipStore(self.data_path, mode='r')
        self.root = zarr.open(store=self.store)
    
    def collate_fn(self, batch: list[tuple[Tensor, int]]):
        labels = []
        samples = []
        for features, label in batch:
            num_chns = features.shape[0]//self.embed_dim
            boc_embed = features.view((num_chns,self.embed_dim)).unsqueeze(0)
            samples.append(boc_embed)
            labels.append(label)
            
        masks = [None for _ in range(len(labels))]
                    
        return samples, Tensor(labels).to(int64), masks
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        return Tensor(self.root[self.file_names[idx]][:]), self.classes[idx]

    def worker_init_fn(self, worker_id = None):
        worker_info = get_worker_info()
        dataset: ZarrChammiFeatures = worker_info.dataset
        dataset.load_zarr()
