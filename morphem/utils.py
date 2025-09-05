import os
import folded_dataset

def configure_dataset(root_dir, dataset_name, transform=None, target_labels="train_test_split"):
    df_path = f"{root_dir}/{dataset_name}/enriched_meta.csv"
    dataset = folded_dataset.SingleCellDataset(
        csv_file=df_path,
        root_dir=root_dir,
        target_labels=target_labels,
        transform=transform,
    )
    return dataset


def path_expansion(path: str):
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))

