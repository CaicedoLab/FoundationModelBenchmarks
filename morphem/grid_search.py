def visualize_grid_search_results(results, output_dir):
    """Create visualizations of the grid search results"""
    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check if we have enough data to create meaningful plots
    if len(df) < 2:
        print("Not enough data points to create visualizations")
        return
    
    # Create top-level summary figure
    plt.figure(figsize=(12, 10))
    plt.scatter(df['val_accuracy'], df['test_accuracy'], alpha=0.7, s=80)
    
    # Annotate each point with its combination ID
    for i, row in df.iterrows():
        if 'combination_id' in row:
            plt.annotate(row['combination_id'], 
                      (row['val_accuracy'], row['test_accuracy']),
                      xytext=(5, 5), textcoords='offset points')
    
    # Add a reference line (perfect correlation between val and test)
    min_val = min(df['val_accuracy'].min(), df['test_accuracy'].min()) - 0.02
    max_val = max(df['val_accuracy'].max(), df['test_accuracy'].max()) + 0.02
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Test Accuracy')
    plt.title('Relationship Between Validation and Test Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'val_vs_test_accuracy.png'))
    
    # Create separate figures for each hyperparameter's effect on validation accuracy
    if 'overfitting_gap' in df.columns:
        # Create a plot of overfitting gap
        plt.figure(figsize=(12, 8))
        plt.scatter(df['val_accuracy'], df['overfitting_gap'], alpha=0.7, s=80)
        plt.xlabel('Validation Accuracy')
        plt.ylabel('Overfitting Gap (Train Acc - Test Acc)')
        plt.title('Overfitting Gap vs Validation Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(df) > 2:
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df['val_accuracy'], df['overfitting_gap']
                )
                x = np.linspace(df['val_accuracy'].min(), df['val_accuracy'].max(), 100)
                y = slope * x + intercept
                plt.plot(x, y, 'r-', alpha=0.7)
                plt.text(0.05, 0.95, f'R² = {r_value**2:.4f}', transform=plt.gca().transAxes,
                       verticalalignment='top')
            except:
                pass  # Skip if error in regression
        
        plt.savefig(os.path.join(plots_dir, 'overfitting_gap.png'))
    
    # Try to analyze the impact of different hyperparameters
    hyperparams = [col for col in df.columns if col not in [
        'combination_id', 'val_accuracy', 'test_accuracy', 'test_precision', 
        'test_recall', 'test_f1', 'train_loss', 'val_loss', 'train_accuracy',
        'epochs_trained', 'early_stopped', 'overfitting_gap'
    ]]
    
    # Create individual plots for each hyperparameter if we have enough data points
    if len(df) >= 5:
        for param in hyperparams:
            try:
                plt.figure(figsize=(12, 6))
                
                # For numeric hyperparameters, use scatter plot
                if df[param].dtype in [np.float64, np.int64, float, int]:
                    plt.scatter(df[param], df['test_accuracy'], alpha=0.8, s=80)
                    plt.grid(True, alpha=0.3)
                    
                    # Add trend line for numeric params if we have enough dataimport torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
import time
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from itertools import product
import json
from torch.utils.tensorboard import SummaryWriter
import random

# Linear classifier with regularization options
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.0, use_bn=False, l1_regularization=0.0):
        super().__init__()
        
        layers = []
        
        # Add batch normalization if requested
        if use_bn:
            layers.append(nn.BatchNorm1d(input_dim))
        
        # Add dropout if requested
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Add linear layer
        self.linear = nn.Linear(input_dim, num_classes)
        layers.append(self.linear)
        
        self.classifier = nn.Sequential(*layers)
        self.l1_regularization = l1_regularization
    
    def forward(self, x):
        return self.classifier(x)
    
    def get_l1_loss(self):
        """Calculate L1 regularization loss for the weights of the linear layer"""
        if self.l1_regularization > 0:
            return self.l1_regularization * torch.norm(self.linear.weight, p=1)
        return 0

def train_linear_classifier_with_early_stopping(
        train_features, train_labels, 
        val_features, val_labels, 
        num_classes=161, dropout_rate=0.0, use_bn=False, 
        lr=0.001, weight_decay=1e-4, l1_regularization=0.0,
        optimizer_type='sgd', adam_beta1=0.9, adam_beta2=0.999,
        epochs=100, batch_size=256, device='cuda:0', 
        momentum=0.9, patience=10, tensorboard_dir=None,
        use_mixup=False, mixup_alpha=0.2,
        use_class_weights=False):
    """Train a linear classifier with early stopping to prevent overfitting
    
    Args:
        train_features: Features from the training set
        train_labels: Labels from the training set
        val_features: Features from the validation set
        val_labels: Labels from the validation set
        num_classes: Number of classes for classification
        dropout_rate: Dropout rate for regularization (0 to disable)
        use_bn: Whether to use batch normalization
        lr: Initial learning rate
        weight_decay: Weight decay coefficient (L2 regularization)
        l1_regularization: L1 regularization coefficient
        optimizer_type: Type of optimizer ('sgd' or 'adam')
        adam_beta1: Beta1 parameter for Adam optimizer
        adam_beta2: Beta2 parameter for Adam optimizer
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        device: Device to train on ('cuda:0', 'cuda:1', etc.)
        momentum: Momentum for SGD optimizer
        patience: Number of epochs to wait for improvement before early stopping
        tensorboard_dir: Directory for TensorBoard logs
        use_mixup: Whether to use Mixup data augmentation
        mixup_alpha: Alpha parameter for Mixup (controls strength of interpolation)
        use_class_weights: Whether to use class weights to handle imbalanced data
    """
    print(f"Training linear classifier with:")
    print(f"- Dropout rate: {dropout_rate}")
    print(f"- Batch normalization: {use_bn}")
    print(f"- L1 regularization: {l1_regularization}")
    print(f"- Optimizer: {optimizer_type.upper()}")
    print(f"- Initial learning rate: {lr}")
    if optimizer_type == 'sgd':
        print(f"- Momentum: {momentum}")
    else:
        print(f"- Adam beta1: {adam_beta1}, beta2: {adam_beta2}")
    print(f"- Weight decay (L2): {weight_decay}")
    print(f"- Mixup: {use_mixup} (alpha={mixup_alpha if use_mixup else 'N/A'})")
    print(f"- Class weights: {use_class_weights}")
    print(f"- Max epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Early stopping patience: {patience}")
    
    # Set up TensorBoard if directory is provided
    if tensorboard_dir:
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
    else:
        writer = None
    
    # Determine input dimension from the feature tensor
    if len(train_features.shape) > 2:
        input_dim = train_features.shape[1] * train_features.shape[2]
        train_features = train_features.reshape(train_features.shape[0], -1)
        val_features = val_features.reshape(val_features.shape[0], -1)
    else:
        input_dim = train_features.shape[1]
    
    # Create the linear classifier with regularization options
    model = LinearClassifier(input_dim, num_classes, dropout_rate, use_bn, l1_regularization).to(device)
    
    # Calculate class weights if requested
    if use_class_weights:
        # Count occurrences of each class
        class_counts = np.bincount(train_labels.numpy())
        # Calculate inverse frequency for each class
        class_weights = 1.0 / class_counts
        # Normalize weights
        class_weights = class_weights / np.sum(class_weights) * len(class_weights)
        # Convert to tensor
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        # Use weighted CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Define optimizer based on type
    if optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    else:  # Adam
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            betas=(adam_beta1, adam_beta2), 
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
    
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Mixup function
    def mixup_data(x, y, alpha=0.2):
        """Applies Mixup augmentation to the batch."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """Calculates loss for Mixup augmentation."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    # Training loop with early stopping
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features, labels = features.to(device), labels.to(device)
            
            # Apply Mixup if enabled
            if use_mixup:
                features, targets_a, targets_b, lam = mixup_data(features, labels, alpha=mixup_alpha)
            
            # Forward pass
            outputs = model(features)
            
            # Calculate loss with Mixup if enabled
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            # Add L1 regularization if enabled
            if l1_regularization > 0:
                l1_loss = model.get_l1_loss()
                loss += l1_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Collect predictions for accuracy calculation (don't evaluate mixup in training)
            if not use_mixup:
                _, preds = torch.max(outputs, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Calculate train accuracy if not using mixup
        if not use_mixup and all_train_preds:
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            train_accuracies.append(train_acc)
        else:
            # If using mixup, we skip training accuracy calculation
            train_acc = 0.0
            train_accuracies.append(0.0)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
        val_acc, val_loss = evaluate_linear_classifier(model, val_loader, criterion, device)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, "
              f"{'Train Acc: '+str(train_acc):.4f if not use_mixup else ''}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Log to TensorBoard if enabled
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            if not use_mixup:
                writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Check if this is the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model with validation accuracy: {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
        # Check if early stopping criteria is met
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Close TensorBoard writer if used
    if writer:
        writer.close()
        
    # Create a plot of training and validation metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies (if not using mixup for training)
    ax2.plot(train_accuracies if not use_mixup else [0] * len(val_accuracies), 
             label='Training Accuracy' if not use_mixup else 'Training Accuracy (Not Available with Mixup)')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot if TensorBoard directory is provided
    if tensorboard_dir:
        plot_path = os.path.join(os.path.dirname(tensorboard_dir), 'training_metrics.png')
        plt.savefig(plot_path)
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model, best_val_acc

.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    
    return accuracy, avg_loss

def hyperparameter_grid_search(train_features, train_labels, val_features, val_labels, test_features, test_labels, args, device):
    """Perform grid search over hyperparameters to find the best model"""
    print("Starting hyperparameter grid search...")
    
    # Define the hyperparameter grid
    param_grid = {
        'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.5],
        'learning_rate': [0.0001, 0.001, 0.005, 0.01],
        'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],
        'l1_regularization': [0.0, 1e-5, 1e-4, 1e-3],
        'batch_size': [64, 128, 256],
        'optimizer_type': ['sgd', 'adam'],
        'use_bn': [False, True],
        'use_mixup': [False, True],
        'use_class_weights': [False, True]
    }
    
    # Create a directory for grid search results
    grid_search_dir = os.path.join(args.output_dir, "grid_search")
    os.makedirs(grid_search_dir, exist_ok=True)
    
    # Track best parameters and performance
    best_val_acc = 0.0
    best_params = None
    best_model = None
    results = []
    
    # If a subset of hyperparameters is specified for testing, modify the grid
    if args.test_mode:
        print("Running in test mode with reduced hyperparameter combinations")
        param_grid = {
            'dropout_rate': [0.0, 0.3],
            'learning_rate': [0.001, 0.01],
            'weight_decay': [0.0, 1e-4],
            'l1_regularization': [0.0],
            'batch_size': [128],
            'optimizer_type': ['sgd'],
            'use_bn': [False],
            'use_mixup': [False],
            'use_class_weights': [False]
        }
    
    # Generate all combinations of hyperparameters for search
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # If specified, run only a subset of randomly selected combinations
    if args.max_combinations > 0 and len(param_combinations) > args.max_combinations:
        print(f"Randomly selecting {args.max_combinations} from {len(param_combinations)} total combinations")
        param_combinations = random.sample(param_combinations, args.max_combinations)
    
    print(f"Total number of hyperparameter combinations to test: {len(param_combinations)}")
    
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
    
    # Store train/val/test shapes for logging
    data_info = {
        "train_features_shape": list(train_features.shape if isinstance(train_features, torch.Tensor) else train_features.shape),
        "val_features_shape": list(val_features.shape if isinstance(val_features, torch.Tensor) else val_features.shape),
        "test_features_shape": list(test_features.shape if isinstance(test_features, torch.Tensor) else test_features.shape),
        "train_labels_shape": list(train_labels.shape if isinstance(train_labels, torch.Tensor) else train_labels.shape),
        "val_labels_shape": list(val_labels.shape if isinstance(val_labels, torch.Tensor) else val_labels.shape),
        "test_labels_shape": list(test_labels.shape if isinstance(test_labels, torch.Tensor) else test_labels.shape),
        "input_dim": input_dim,
        "num_classes": args.num_classes
    }
    
    # Log class distribution (imbalance checking)
    train_class_dist = np.bincount(train_labels.numpy() if isinstance(train_labels, torch.Tensor) else train_labels)
    val_class_dist = np.bincount(val_labels.numpy() if isinstance(val_labels, torch.Tensor) else val_labels)
    test_class_dist = np.bincount(test_labels.numpy() if isinstance(test_labels, torch.Tensor) else test_labels)
    
    data_info["train_class_distribution"] = train_class_dist.tolist()
    data_info["val_class_distribution"] = val_class_dist.tolist()
    data_info["test_class_distribution"] = test_class_dist.tolist()
    
    # Save data info
    with open(os.path.join(grid_search_dir, "data_info.json"), 'w') as f:
        json.dump(data_info, f, indent=2)
    
    # Create a DataFrame to log results as we go
    results_df = pd.DataFrame(columns=[
        'combination_id', 'dropout_rate', 'learning_rate', 'weight_decay', 
        'l1_regularization', 'batch_size', 'optimizer_type', 'use_bn', 
        'use_mixup', 'use_class_weights', 'train_loss', 'val_loss', 
        'val_accuracy', 'test_accuracy', 'test_precision', 'test_recall', 
        'test_f1', 'epochs_trained', 'early_stopped', 'overfitting_gap'
    ])
    
    # Loop through all parameter combinations
    for i, params in enumerate(param_combinations):
        combination_id = f"combination_{i+1}"
        print(f"\n[{i+1}/{len(param_combinations)}] Testing hyperparameters:")
        for param, value in params.items():
            print(f"  - {param}: {value}")
        
        # Set up TensorBoard directory for this run
        run_name = f"comb_{i+1}_do_{params['dropout_rate']}_lr_{params['learning_rate']}_wd_{params['weight_decay']}_l1_{params['l1_regularization']}_bs_{params['batch_size']}_opt_{params['optimizer_type']}_bn_{params['use_bn']}_mx_{params['use_mixup']}_cw_{params['use_class_weights']}"
        tensorboard_dir = os.path.join(grid_search_dir, "tensorboard", run_name)
        
        # Save current parameters to file
        with open(os.path.join(grid_search_dir, f"params_{i+1}.json"), 'w') as f:
            json.dump(params, f, indent=2)
        
        try:
            # Train model with early stopping
            model, val_acc = train_linear_classifier_with_early_stopping(
                train_features_flat, train_labels,
                val_features_flat, val_labels,
                num_classes=args.num_classes,
                dropout_rate=params['dropout_rate'],
                use_bn=params['use_bn'],
                lr=params['learning_rate'],
                weight_decay=params['weight_decay'],
                l1_regularization=params['l1_regularization'],
                optimizer_type=params['optimizer_type'],
                epochs=args.epochs,
                batch_size=params['batch_size'],
                device=device,
                momentum=args.momentum,
                patience=args.patience,
                tensorboard_dir=tensorboard_dir,
                use_mixup=params['use_mixup'],
                mixup_alpha=args.mixup_alpha,
                use_class_weights=params['use_class_weights']
            )
            
            # Evaluate on test set
            test_dataset = TensorDataset(test_features_flat, test_labels)
            test_loader = DataLoader(
                test_dataset, 
                batch_size=params['batch_size'], 
                shuffle=False
            )
            
            test_acc, test_loss = evaluate_linear_classifier(model, test_loader, nn.CrossEntropyLoss(), device)
            print(f"Test accuracy: {test_acc:.4f}")
            
            # Evaluate on train set to check for overfitting
            train_dataset = TensorDataset(train_features_flat, train_labels)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=params['batch_size'],
                shuffle=False
            )
            
            train_acc, train_loss = evaluate_linear_classifier(model, train_loader, nn.CrossEntropyLoss(), device)
            print(f"Train accuracy: {train_acc:.4f}")
            
            # Calculate overfitting gap (train accuracy - test accuracy)
            overfitting_gap = train_acc - test_acc
            print(f"Overfitting gap: {overfitting_gap:.4f}")
            
            # Generate detailed metrics
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate precision, recall, and F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted'
            )
            
            # Store results
            result = {
                'combination_id': combination_id,
                **params,
                'train_loss': train_loss,
                'val_loss': test_loss,  # Using test_loss as val_loss for consistency
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'epochs_trained': -1,  # We don't track this currently but could
                'early_stopped': True,  # Assuming we used early stopping
                'overfitting_gap': overfitting_gap
            }
            results.append(result)
            
            # Add to DataFrame
            results_df.loc[len(results_df)] = {
                'combination_id': combination_id,
                'dropout_rate': params['dropout_rate'],
                'learning_rate': params['learning_rate'],
                'weight_decay': params['weight_decay'],
                'l1_regularization': params['l1_regularization'],
                'batch_size': params['batch_size'],
                'optimizer_type': params['optimizer_type'],
                'use_bn': params['use_bn'],
                'use_mixup': params['use_mixup'],
                'use_class_weights': params['use_class_weights'],
                'train_loss': train_loss,
                'val_loss': test_loss,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'epochs_trained': -1,
                'early_stopped': True,
                'overfitting_gap': overfitting_gap
            }
            
            # Save updated results after each combination
            results_df.to_csv(os.path.join(grid_search_dir, "grid_search_results.csv"), index=False)
            
            # Update best model if this one is better
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params.copy()
                best_model = model
                
                # Save best model so far
                model_path = os.path.join(grid_search_dir, "best_model.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_dim': input_dim,
                    'num_classes': args.num_classes,
                    'params': best_params,
                    'val_accuracy': val_acc,
                    'test_accuracy': test_acc
                }, model_path)
                
                # Save classification report for best model
                best_report = classification_report(all_labels, all_preds, output_dict=True)
                with open(os.path.join(grid_search_dir, "best_model_report.json"), 'w') as f:
                    json.dump(best_report, f, indent=2)
                
                # Save updated best parameters
                with open(os.path.join(grid_search_dir, "best_params.json"), 'w') as f:
                    json.dump({
                        'best_params': best_params,
                        'best_val_accuracy': best_val_acc,
                        'test_accuracy': test_acc,
                        'overfitting_gap': overfitting_gap
                    }, f, indent=2)
        
        except Exception as e:
            print(f"Error with combination {i+1}: {str(e)}")
            # Log the error and continue with next combination
            with open(os.path.join(grid_search_dir, f"error_{i+1}.txt"), 'w') as f:
                f.write(f"Error with combination {i+1}: {str(e)}")
    
    # After all combinations, create visualizations
    if len(results) > 0:
        visualize_grid_search_results(results, grid_search_dir)
    
    # Print best parameters
    if best_params:
        print("\nGrid search completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print("Best hyperparameters:")
        for param, value in best_params.items():
            print(f"  - {param}: {value}")
        
        # Generate classification report for best model
        test_dataset = TensorDataset(test_features_flat, test_labels)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=best_params['batch_size'], 
            shuffle=False
        )
        
        best_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = best_model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Generate and save classification report
        report = classification_report(all_labels, all_preds)
        with open(os.path.join(grid_search_dir, "best_model_classification_report.txt"), 'w') as f:
            f.write(f"Best Model Classification Report:\n")
            f.write(report)
        
        print("\nBest Model Classification Report:")
        print(report)
    else:
        print("No valid models were found during the grid search!")
    
    return best_model, best_params, best_val_acc

def visualize_grid_search_results(results, output_dir):
    """Create visualizations of the grid search results"""
    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Validation accuracy vs. Learning rate, grouped by dropout rate
    plt.figure(figsize=(12, 8))
    for dropout in df['dropout_rate'].unique():
        subset = df[df['dropout_rate'] == dropout]
        plt.plot(subset['learning_rate'], subset['val_accuracy'], 
                 marker='o', linestyle='-', label=f'Dropout={dropout}')
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Learning Rate')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'val_acc_vs_lr.png'))
    
    # Plot 2: Validation accuracy vs. Dropout rate, grouped by weight decay
    plt.figure(figsize=(12, 8))
    for wd in df['weight_decay'].unique():
        subset = df[df['weight_decay'] == wd]
        plt.plot(subset['dropout_rate'], subset['val_accuracy'], 
                 marker='o', linestyle='-', label=f'Weight Decay={wd}')
    
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Dropout Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'val_acc_vs_dropout.png'))
    
    # Plot 3: Training accuracy vs. Test accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(df['val_accuracy'], df['test_accuracy'], alpha=0.7)
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Test Accuracy')
    plt.title('Validation Accuracy vs. Test Accuracy')
    plt.grid(True)
    
    # Add a reference line
    min_val = min(df['val_accuracy'].min(), df['test_accuracy'].min())
    max_val = max(df['val_accuracy'].max(), df['test_accuracy'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.savefig(os.path.join(plots_dir, 'val_vs_test_acc.png'))
    
    # Plot 4: F1 Score vs. different hyperparameters
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1 vs Learning Rate
    axes[0, 0].scatter(df['learning_rate'], df['test_f1'], alpha=0.7)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('F1 Score vs. Learning Rate')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True)
    
    # F1 vs Dropout Rate
    axes[0, 1].scatter(df['dropout_rate'], df['test_f1'], alpha=0.7)
    axes[0, 1].set_xlabel('Dropout Rate')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score vs. Dropout Rate')
    axes[0, 1].grid(True)
    
    # F1 vs Weight Decay
    axes[1, 0].scatter(df['weight_decay'], df['test_f1'], alpha=0.7)
    axes[1, 0].set_xlabel('Weight Decay')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score vs. Weight Decay')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True)
    
    # F1 vs Batch Size
    axes[1, 1].scatter(df['batch_size'], df['test_f1'], alpha=0.7)
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score vs. Batch Size')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'f1_vs_hyperparams.png'))
    
    # Plot 5: Heatmap of validation accuracy for different combinations of learning rate and weight decay
    pivot_df = df.pivot_table(
        values='val_accuracy', 
        index='learning_rate', 
        columns='weight_decay', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 10))