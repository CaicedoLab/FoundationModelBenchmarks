import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
from torchvision import transforms
import time
from sklearn.metrics import accuracy_score, classification_report

# Import our custom modules
from jumpcp import JUMPCP  # Using the JUMPCP dataset from folded_dataset.py
from vision_transformer import vit_small, vit_base  # Import the ViT model

class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1] if x.dim() == 4 else x.shape[0]
        if self.instance_norm.num_features != C:
            self.instance_norm = nn.InstanceNorm2d(
                num_features=C,
                affine=False,
                track_running_stats=False,
                eps=self.eps
            )
        return self.instance_norm(x)

class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # Batch of images
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    channel = x[i, j]
                    noise = torch.empty_like(channel).uniform_(self.low, self.high)
                    mask = (channel == 255).float()
                    noise_masked = noise * mask
                    channel[channel == 255] = 0
                    x[i, j] = channel + noise_masked
        else:  # Single image
            for j in range(x.shape[0]):
                channel = x[j]
                noise = torch.empty_like(channel).uniform_(self.low, self.high)
                mask = (channel == 255).float()
                noise_masked = noise * mask
                channel[channel == 255] = 0
                x[j] = channel + noise_masked
        return x

class ViTExtractor():
    def __init__(self, gpu):
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        # Create model with in_chans=1 to match training setup
        self.model = vit_small()
        remove_prefixes = ["module.backbone.", "module.", "module.head."]

        # Load model weights
        student_model = torch.load("/scr/vidit/Foundation_Models/model_weights/10ds_multiscale_guided/checkpoint.pth")['student']
        # Remove unwanted prefixes
        cleaned_state_dict = {}
        for k, v in student_model.items():
            new_key = k
            for prefix in remove_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]  # Remove prefix
            if not new_key.startswith("head.mlp") and not new_key.startswith("head.last_layer"):
                cleaned_state_dict[new_key] = v  # Keep only valid keys
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def get_model(self):
        return self.model
    
    def extract_features(self, x, return_layers=False):
        """
        Extract features from the model, optionally returning intermediate layer outputs
        for the last 4 layers as described in Caron et al. (2021)
        """
        if return_layers:
            return self.model.forward_features(x, return_layers=True)
        else:
            return self.model.forward_features(x)


# Replace the MLPClassifier with a linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

def create_pad(images, patch_width, patch_height): # new method for vit model
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
        

    padded_images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return padded_images

def extract_features_from_dataset(dataloader, vit_model, gpu, feature_file=None, batch_size=32, return_last_layers=True):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    all_features = []
    all_labels = []
    
    # Get patch size for padding
    patch_embed = vit_model.patch_embed
    conv_layer = patch_embed.proj
    patch_size = conv_layer.kernel_size
    patch_height, patch_width = patch_size
    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            images = batch["image"]
            labels = batch["label"]
            
            # Apply padding to make dimensions compatible with patch size
            images = create_pad(images, patch_width, patch_height)
            
            # Extract features from each channel
            batch_channel_features = []
            for c in range(images.shape[1]):
                # Get single channel and process it
                single_channel = images[:, c, :, :].unsqueeze(1).to(device)
                
                # Get features from the model
                output = vit_model.forward_features(single_channel)
                features = output["x_norm_clstoken"].cpu().detach()
                
                batch_channel_features.append(features)
            
            # Concatenate features from all channels
            batch_features = torch.cat(batch_channel_features, dim=1)
            
            all_features.append(batch_features)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Save features and labels if feature_file is provided
    if feature_file:
        os.makedirs(os.path.dirname(feature_file), exist_ok=True)
        torch.save(all_features, f"{feature_file}_features.pt")
        torch.save(all_labels, f"{feature_file}_labels.pt")
    
    return all_features, all_labels


def train_linear_classifier(train_features, train_labels, val_features=None, val_labels=None, 
                           num_classes=161, embed_dim=None, lr=0.00001, weight_decay=1e-4, 
                           epochs=100, batch_size=256, device='cuda:0', momentum=0.9):
    """Train a linear classifier on the extracted features using SGD with momentum
    Args:
        train_features: Features from the training set
        train_labels: Labels from the training set
        val_features: Features from the validation set
        val_labels: Labels from the validation set
        num_classes: Number of classes for classification
        embed_dim: Embedding dimension (not used, kept for compatibility)
        lr: Initial learning rate
        weight_decay: Weight decay coefficient 
        epochs: Total number of training epochs
        batch_size: Batch size for training
        device: Device to train on ('cuda:0', 'cuda:1', etc.)
        momentum: Momentum for SGD optimizer
    """
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    import math
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score
    
    print(f"Training linear classifier with SGD...")
    print(f"- Initial learning rate: {lr}")
    print(f"- Momentum: {momentum}")
    print(f"- Weight decay: {weight_decay}")
    print(f"- Total epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    
    # Determine input dimension from the feature tensor
    if len(train_features.shape) > 2:
        input_dim = train_features.shape[1] * train_features.shape[2]
        train_features = train_features.reshape(train_features.shape[0], -1)
        if val_features is not None:
            val_features = val_features.reshape(val_features.shape[0], -1)
    else:
        input_dim = train_features.shape[1]
    
    # Create the linear classifier
    model = LinearClassifier(input_dim, num_classes).to(device)
    
    # Define loss and optimizer (SGD with momentum as per Caron et al. 2021)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs
    )
    
    # Convert to PyTorch datasets for batching
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    if val_features is not None and val_labels is not None:
        val_dataset = TensorDataset(val_features, val_labels)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    else:
        val_loader = None
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Collect predictions for accuracy calculation
            _, preds = torch.max(outputs, 1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
        
        # Calculate train accuracy
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation
        if val_loader:
            val_acc, val_loss = evaluate_linear_classifier(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                print(f"New best model with validation accuracy: {best_val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}")
    
    # Load best model if validation was performed
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model


def evaluate_linear_classifier(model, data_loader, criterion=None, device='cuda:0'):
    """Evaluate linear classifier on provided data"""
    model.eval()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    return accuracy, avg_loss


def process_plate(args, plate_id):
    """Process a single plate with feature extraction and linear classifier training"""
    print(f"Processing plate: {plate_id}")
    
    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    # Set up the transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        SaturationNoiseInjector(),
        PerImageNormalize()
    ])
    
    # Configure channels for the datasets
    channels = {
        "training": args.channels,
        "validation": args.channels,
        "test": args.channels,
    }
    
    # Create feature directory for this plate
    plate_feature_dir = os.path.join(args.feature_dir, plate_id)
    os.makedirs(plate_feature_dir, exist_ok=True)
    
    # Paths for saving features
    train_feature_path = os.path.join(plate_feature_dir, "train")
    val_feature_path = os.path.join(plate_feature_dir, "val")
    test_feature_path = os.path.join(plate_feature_dir, "test")
    
    # Check if features already exist
    train_features_exist = os.path.exists(f"{train_feature_path}_features.pt")
    val_features_exist = os.path.exists(f"{val_feature_path}_features.pt")
    test_features_exist = os.path.exists(f"{test_feature_path}_features.pt")
    
    # Initialize the ViT model for feature extraction if needed
    if not (train_features_exist and val_features_exist and test_features_exist):
        print("Initializing ViT model for feature extraction...")
        vit_instance = ViTExtractor(args.gpu)
        vit_model = vit_instance.get_model()
    else:
        vit_model = None
    
    # Load or extract features
    if train_features_exist:
        print("Loading pre-extracted training features...")
        train_features = torch.load(f"{train_feature_path}_features.pt")
        train_labels = torch.load(f"{train_feature_path}_labels.pt")
    else:
        # Create train dataset
        print("Creating training dataset...")
        train_set = JUMPCP(
            path=args.root_dir, 
            split="train", 
            transform=transform, 
            channels=channels["training"], 
            use_hdf5=args.use_hdf5,
            perturbation_list=[args.perturbation],
            cyto_mask_path_list=[os.path.join(args.root_dir, f"jumpcp/{plate_id}.pq")]
        )
        
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,  # No need to shuffle for feature extraction
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=JUMPCP.collate_fn,
        )
        
        print("Extracting features for training set...")
        train_features, train_labels = extract_features_from_dataset(
            train_loader, vit_model, args.gpu, train_feature_path, args.batch_size, 
            return_last_layers=True  # Extract features from last 4 layers
        )
    
    # Similar code for validation and test features...
    # (Code for validation and test feature extraction would be updated similarly)
    
    if val_features_exist:
        print("Loading pre-extracted validation features...")
        val_features = torch.load(f"{val_feature_path}_features.pt")
        val_labels = torch.load(f"{val_feature_path}_labels.pt")
    else:
        # Create validation dataset
        print("Creating validation dataset...")
        val_set = JUMPCP(
            path=args.root_dir, 
            split="valid", 
            transform=transform, 
            channels=channels["validation"], 
            use_hdf5=args.use_hdf5,
            perturbation_list=[args.perturbation],
            cyto_mask_path_list=[os.path.join(args.root_dir, f"jumpcp/{plate_id}.pq")]
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=JUMPCP.collate_fn,
        )
        
        print("Extracting features for validation set...")
        val_features, val_labels = extract_features_from_dataset(
            val_loader, vit_model, args.gpu, val_feature_path, args.batch_size,
            return_last_layers=True  # Extract features from last 4 layers
        )
    
    if test_features_exist:
        print("Loading pre-extracted test features...")
        test_features = torch.load(f"{test_feature_path}_features.pt")
        test_labels = torch.load(f"{test_feature_path}_labels.pt")
    else:
        # Create test dataset
        print("Creating test dataset...")
        test_set = JUMPCP(
            path=args.root_dir, 
            split="test", 
            transform=transform, 
            channels=channels["test"], 
            use_hdf5=args.use_hdf5,
            perturbation_list=[args.perturbation],
            cyto_mask_path_list=[os.path.join(args.root_dir, f"jumpcp/{plate_id}.pq")]
        )
        
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=JUMPCP.collate_fn,
        )
        
        print("Extracting features for test set...")
        test_features, test_labels = extract_features_from_dataset(
            test_loader, vit_model, args.gpu, test_feature_path, args.batch_size,
            return_last_layers=True  # Extract features from last 4 layers
        )
    
    # Reshape features if needed
    if len(train_features.shape) > 2:
        input_dim = train_features.shape[1] * train_features.shape[2]
        train_features_flat = train_features.reshape(train_features.shape[0], -1)
        val_features_flat = val_features.reshape(val_features.shape[0], -1)
        test_features_flat = test_features.reshape(test_features.shape[0], -1)
    else:
        input_dim = train_features.shape[1]
        train_features_flat = train_features
        val_features_flat = val_features
        test_features_flat = test_features
    
    # Create datasets for training the linear classifier
    train_dataset = torch.utils.data.TensorDataset(train_features_flat, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_features_flat, val_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features_flat, test_labels)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Train linear classifier
    print(f"Training linear classifier for plate {plate_id}...")
    
    # Update the training function to use SGD with momentum
    linear_clf = train_linear_classifier(
        train_features_flat, 
        train_labels,
        val_features_flat, 
        val_labels,
        num_classes=args.num_classes,
        lr=args.learning_rate,  # LR from Caron et al.
        momentum=args.momentum,  # Momentum from Caron et al.
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device
    )
    
    # Save the model
    model_path = os.path.join(plate_feature_dir, "linear_model.pt")
    torch.save({
        'model_state_dict': linear_clf.state_dict(),
        'input_dim': input_dim,
        'num_classes': args.num_classes
    }, model_path)
    
    # Evaluate on test set
    test_acc, _ = evaluate_linear_classifier(
        linear_clf, 
        test_loader, 
        nn.CrossEntropyLoss(), 
        device
    )
    
    # Generate classification report
    linear_clf.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = linear_clf(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds)
    
    # Save results
    results_path = os.path.join(plate_feature_dir, "results_linear.txt")
    with open(results_path, 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:")
    print(report)
    
    return test_acc, report

def main(args):
    start_time = time.time()
    
    # Process each plate
    results = {}
    for plate_id in args.plates:
        plate_acc, plate_report = process_plate(args, plate_id)
        results[plate_id] = {
            'accuracy': plate_acc,
            'report': plate_report
        }
    
    # Calculate average accuracy across all plates
    avg_acc = sum(results[plate]['accuracy'] for plate in args.plates) / len(args.plates)
    
    # Save overall results
    overall_results_path = os.path.join(args.feature_dir, "overall_results_linear.txt")
    with open(overall_results_path, 'w') as f:
        f.write(f"Average Linear Classifier Accuracy Across All Plates: {avg_acc:.4f}\n\n")
        
        for plate_id in args.plates:
            f.write(f"Plate {plate_id} Accuracy: {results[plate_id]['accuracy']:.4f}\n")
            f.write(f"Plate {plate_id} Classification Report:\n")
            f.write(results[plate_id]['report'])
            f.write("\n\n")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nProcessing completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average Linear Classifier Accuracy Across All Plates: {avg_acc:.4f}")
    print(f"Overall results saved to {overall_results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features and train linear classifier for JUMPCP")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory for the dataset")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory to save features")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for feature extraction and training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2, 3, 4], 
                        help="Channels to use from the images")
    parser.add_argument("--use_hdf5", action="store_true", help="Use HDF5 dataset format")
    parser.add_argument("--perturbation", type=str, default="compound", 
                        choices=["compound", "crispr", "orf"], 
                        help="Perturbation type to use")
    parser.add_argument("--num_classes", type=int, default=161, 
                        help="Number of classes for classification")
    parser.add_argument("--use_last_layers", action="store_true", default=True,
                        help="Use features from the last 4 layers as per Caron et al. (2021)")
    parser.add_argument("--learning_rate", type=float, default=0.0005, 
                        help="Learning rate for linear classifier training")
    parser.add_argument("--momentum", type=float, default=0.9, 
                        help="Momentum for SGD optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                        help="Weight decay for linear classifier training")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs for linear classifier training")
    parser.add_argument("--plates", type=str, nargs="+", 
                        default=["BR00116991"], 
                        help="List of plates to process")
    
    args = parser.parse_args()
    main(args)