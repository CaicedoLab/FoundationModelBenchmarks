import os
import pandas as pd
import folded_dataset

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


