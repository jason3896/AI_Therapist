import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
import torch.nn.functional as F
import math
from models import resnet
from models import EfficientFace
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

# Set up device for Windows
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/model.pth.tar')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/model_best.pth.tar')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
parser.add_argument('--factor', default=0.1, type=float, metavar='FT')
parser.add_argument('--af', '--adjust-freq', default=30, type=int, metavar='N', help='adjust learning rate frequency')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
parser.add_argument('--gpu', type=str, default="0")

# Add arguments for class balanced loss and focal loss
parser.add_argument('--use-class-balanced', action='store_true', help='Use class-balanced loss')
parser.add_argument('--use-focal-loss', action='store_true', help='Use focal loss')
parser.add_argument('--beta', type=float, default=0.99, help='Beta parameter for class-balanced loss')
parser.add_argument('--gamma', type=float, default=3.5, help='Gamma parameter for focal loss (increased from 2.5)')
# Add argument for two-stage approach
parser.add_argument('--two-stage', action='store_true', help='Use two-stage approach for difficult emotions')

# Add label smoothing argument
parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing parameter')
parser.add_argument('--use-label-smoothing', action='store_true', help='Use label smoothing loss')
# Add temperature scaling argument
parser.add_argument('--use-temperature-scaling', action='store_true', help='Use temperature scaling for calibration')
parser.add_argument('--class-specific-temp', action='store_true', help='Use class-specific temperature scaling')
# Add argument for error analysis
parser.add_argument('--save-error-samples', action='store_true', help='Save misclassified samples for analysis')
parser.add_argument('--error-samples-dir', type=str, default='./error_samples', help='Directory to save error samples')
# Add ensemble arguments
parser.add_argument('--use-ensemble', action='store_true', help='Use model ensemble')
parser.add_argument('--num-ensemble-models', type=int, default=3, help='Number of models in ensemble')
# Add gradient clipping argument
parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping norm')
# Add visualization arguments
parser.add_argument('--save-tsne', action='store_true', help='Save t-SNE visualization of features')
parser.add_argument('--save-confidence-hist', action='store_true', help='Save confidence histograms')

args = parser.parse_args()


# ======================================
# ENHANCED ATTENTION MODULE
# ======================================
class MultiHeadAttentionModule(nn.Module):
    def __init__(self, in_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(num_heads)
        ])
        
        # Fix: Make sure the input dimension for combine matches the output dimension
        # We're not changing the feature dimension, just applying attention
        self.combine = nn.Linear(in_features, in_features)
        
    def forward(self, x):
        # Store the original input shape
        batch_size = x.size(0)
        
        # Apply each attention head
        attention_outputs = []
        for head in self.attention_heads:
            weights = torch.sigmoid(head(x))
            attention_outputs.append(x * weights)
        
        # Average the attention outputs instead of concatenating
        # This preserves the input dimensions
        combined = torch.stack(attention_outputs).mean(dim=0)
        
        # Apply final linear transformation
        return self.combine(combined)


class AttentionModule(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        weights = torch.sigmoid(self.attention(x))
        return x * weights


# ======================================
# ENHANCED LOSS FUNCTIONS
# ======================================
# Enhanced Focal Loss with higher gamma
class EnhancedFocalLoss(nn.Module):
    def __init__(self, gamma=3.5, alpha=None, size_average=True):  # Increased gamma for harder examples
        super(EnhancedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', weight=alpha)

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


# Class-Balanced Focal Loss
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.99, gamma=3.5):  # Increased gamma
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.samples_per_class = samples_per_class
        
        # Compute effective number of samples
        effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        # Compute weights
        self.weights = (1.0 - self.beta) / np.array(effective_num)
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights) * len(self.weights)
        self.weights = torch.tensor(self.weights, dtype=torch.float).to(device)
        
    def forward(self, logits, targets):
        # Focal loss component
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class balancing weights
        class_weights = self.weights[targets]
        
        # Final loss
        loss = focal_weight * class_weights * ce_loss
        return loss.mean()


# ======================================
# ENHANCED LABEL SMOOTHING LOSS
# ======================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.weight = weight
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        loss = torch.sum(-true_dist * pred, dim=-1)
        
        # Apply class weights if provided
        if self.weight is not None:
            weights = self.weight[target]
            loss = loss * weights
            
        return loss.mean()


# ======================================
# ENHANCED TEMPERATURE SCALING
# ======================================
class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def forward(self, logits):
        # Apply temperature scaling
        return logits / self.temperature

    def calibrate(self, val_loader, model, criterion):
        """
        Tune the temperature parameter using validation data
        """
        original_nll = 0.0
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images, target = images.to(device), target.to(device)
                output = model(images)
                original_nll += criterion(output, target).item()
        
        original_nll /= len(val_loader)
        print(f"Original NLL: {original_nll:.6f}")
        
        # Use L-BFGS to optimize the temperature parameter
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            nll = 0.0
            for i, (images, target) in enumerate(val_loader):
                images, target = images.to(device), target.to(device)
                output = model(images)
                scaled_output = self.forward(output)
                nll += criterion(scaled_output, target).item()
            
            nll /= len(val_loader)
            loss = torch.tensor(nll, requires_grad=True)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        calibrated_nll = 0.0
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                images, target = images.to(device), target.to(device)
                output = model(images)
                scaled_output = self.forward(output)
                calibrated_nll += criterion(scaled_output, target).item()
        
        calibrated_nll /= len(val_loader)
        print(f"Temperature: {self.temperature.item():.6f}")
        print(f"Calibrated NLL: {calibrated_nll:.6f}")


# ======================================
# CLASS-SPECIFIC TEMPERATURE SCALING
# ======================================
class ClassSpecificTemperatureScaling(nn.Module):
    def __init__(self, num_classes, initial_temp=1.0):
        super(ClassSpecificTemperatureScaling, self).__init__()
        # One temperature parameter per class
        self.temperatures = nn.Parameter(torch.ones(num_classes) * initial_temp)
        
    def forward(self, logits):
        # Scale each class logit by its own temperature
        scaled_logits = logits.clone()
        for c in range(logits.size(1)):
            scaled_logits[:, c] = logits[:, c] / self.temperatures[c]
        return scaled_logits
    
    def calibrate(self, val_loader, model, criterion):
        """Tune temperature parameters using validation data"""
        optimizer = torch.optim.LBFGS(
            [self.temperatures], 
            lr=0.01, 
            max_iter=50
        )
        
        def eval_fn():
            optimizer.zero_grad()
            total_loss = 0.0
            num_samples = 0
            
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                scaled_outputs = self.forward(outputs)
                loss = criterion(scaled_outputs, targets)
                total_loss += loss.item() * images.size(0)
                num_samples += images.size(0)
            
            avg_loss = total_loss / num_samples
            loss_tensor = torch.tensor(avg_loss, requires_grad=True, device=device)
            loss_tensor.backward()
            
            print(f"Current temperatures: {self.temperatures.data}")
            print(f"Loss: {avg_loss:.6f}")
            return loss_tensor
        
        optimizer.step(eval_fn)
        
        # Print final temperatures
        print("\nFinal temperature values:")
        for i, temp in enumerate(self.temperatures.data):
            print(f"  Class {i}: {temp.item():.4f}")


# ======================================
# ENSEMBLE MODEL
# ======================================
class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        # Initialize uniform weights if not provided
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            # Normalize weights
            self.weights = torch.tensor(weights) / sum(weights)
    
    def forward(self, x):
        # Get predictions from all models
        outputs = []
        for i, model in enumerate(self.models):
            outputs.append(self.weights[i] * model(x))
        
        # Return weighted sum of outputs
        return sum(outputs)


def optimize_ensemble_weights(ensemble, val_loader):
    """Optimize weights for combining ensemble models"""
    # Initialize weights as parameters for optimization
    weights = nn.Parameter(torch.ones(len(ensemble.models)))
    optimizer = torch.optim.Adam([weights], lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few epochs to optimize weights
    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with each model
            outputs = []
            for i, model in enumerate(ensemble.models):
                outputs.append(torch.softmax(model(images), dim=1) * torch.softmax(weights[i], dim=0))
            
            # Combine outputs
            combined_output = sum(outputs)
            
            # Calculate loss
            loss = criterion(combined_output, targets)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = combined_output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Normalize weights
        ensemble.weights = torch.softmax(weights, dim=0).detach()
        
        # Print statistics
        acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss: {running_loss/len(val_loader):.4f} Acc: {acc:.2f}%')
        print(f'Weights: {ensemble.weights}')
    
    print("Final ensemble weights:")
    for i, w in enumerate(ensemble.weights):
        print(f"  Model {i}: {w.item():.4f}")
    
    return ensemble


def create_ensemble_models(num_models, model_cla, traindir, valdir):
    """Create and train multiple models for ensemble"""
    models = [model_cla]  # Add the base model
    
    # Train additional models with different architectures/initializations
    for i in range(1, num_models):
        print(f"Creating model {i+1} for ensemble...")
        
        # Create new model with different architecture or initialization
        if i == 1:
            # Use ResNet-18 as second model
            model = resnet.resnet18(pretrained=False)
            model.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                MultiHeadAttentionModule(256, num_heads=2),
                nn.Linear(256, 7)
            )
        else:
            # Use another EfficientFace with different initialization
            model = EfficientFace.efficient_face()
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),  # Different dropout
                AttentionModule(512),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 7)
            )
        
        model = model.to(device)
        models.append(model)
    
    return models


def count_images_per_class(data_dir):
    class_counts = {}
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_dir] = count
    return class_counts


# ======================================
# CUSTOM DATASET FOR TARGETED AUGMENTATION
# ======================================
# Use a simplified approach with standard ImageFolder dataset
def get_datasets(traindir, valdir, train_transform, test_transform, class_specific_transforms=None):
    """
    Create datasets with optional class-specific transformations
    """
    # For validation dataset, use standard ImageFolder
    val_dataset = datasets.ImageFolder(valdir, test_transform)
    
    # For training dataset, either use standard ImageFolder or create augmented versions for minority classes
    if class_specific_transforms is None:
        train_dataset = datasets.ImageFolder(traindir, train_transform)
    else:
        # Standard dataset with regular augmentation
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        
        # Create specialized augmentation datasets for minority classes
        augmented_samples = []
        for class_idx, transform in class_specific_transforms.items():
            # Find samples for this class
            class_dir = os.path.join(traindir, str(class_idx))
            if os.path.isdir(class_dir):
                # Oversample this class by creating multiple augmented versions
                samples_for_class = [(os.path.join(class_dir, fname), class_idx) 
                                    for fname in os.listdir(class_dir) 
                                    if os.path.isfile(os.path.join(class_dir, fname))]
                
                # Add these samples to the dataset using regular transform (they're already in the main dataset)
                for _ in range(2):  # Duplicate minority classes 2 times (effectively 3x representation)
                    augmented_samples.extend(samples_for_class)
        
        # Convert augmented samples to a dataset
        if augmented_samples:
            augmented_dataset = datasets.ImageFolder(traindir, train_transform)
            # Replace samples with our augmented list
            augmented_dataset.samples.extend(augmented_samples)
            augmented_dataset.imgs.extend(augmented_samples)
            # Update targets list
            augmented_dataset.targets = [s[1] for s in augmented_dataset.samples]
            train_dataset = augmented_dataset
    
    return train_dataset, val_dataset


# ======================================
# ENHANCED ERROR ANALYSIS FUNCTION
# ======================================
def analyze_errors(model, data_loader, class_names, save_dir=None, model_fine=None, temperature_scaling=None):
    """
    Analyze errors made by the model and save misclassified samples.
    """
    model.eval()
    if model_fine is not None:
        model_fine.eval()
    
    # Create directories to save error samples if needed
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for class_name in class_names:
            os.makedirs(os.path.join(save_dir, class_name), exist_ok=True)
    
    errors = {i: [] for i in range(len(class_names))}
    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    
    # Store features, true labels, and predictions for t-SNE visualization
    all_features = []
    all_true_labels = []
    all_pred_labels = []
    all_confidences = []
    all_correct_flags = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Get predictions using two-stage approach if enabled
            if args.two_stage and model_fine is not None:
                # First get prediction from main model
                outputs = model(images)
                
                # For feature extraction (before temperature scaling)
                # Extract features from the second-to-last layer for t-SNE
                if args.save_tsne:
                    # This assumes the model has a structure that allows access to the features
                    # Modify this based on your actual model structure
                    if hasattr(model, 'get_features'):
                        features = model.get_features(images)
                    else:
                        # Fallback - this is just a placeholder, adjust to your model
                        features = outputs.detach().cpu().numpy()
                    all_features.extend(features)
                
                # Apply temperature scaling if provided
                if temperature_scaling is not None:
                    scaled_outputs = temperature_scaling(outputs)
                    probs = F.softmax(scaled_outputs, dim=1)
                    _, preds = torch.max(scaled_outputs, 1)
                else:
                    probs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                
                # For samples predicted as difficult emotions (4,5,6), use fine model
                fine_mask = torch.isin(preds, torch.tensor([4, 5, 6]).to(device))
                
                # If we have any samples predicted as difficult emotions
                if fine_mask.sum() > 0:
                    # Get those samples
                    fine_images = images[fine_mask]
                    
                    # Get fine model predictions for these samples
                    fine_outputs = model_fine(fine_images)
                    
                    # Convert fine output (0,1,2) back to original classes (4,5,6)
                    fine_preds = torch.argmax(fine_outputs, dim=1) + 4
                    
                    # Update the predictions
                    preds[fine_mask] = fine_preds
            else:
                # Standard approach
                outputs = model(images)
                
                # For feature extraction (before temperature scaling)
                if args.save_tsne:
                    # This assumes the model has a structure that allows access to the features
                    # Modify this based on your actual model structure
                    if hasattr(model, 'get_features'):
                        features = model.get_features(images)
                    else:
                        # Fallback - this is just a placeholder, adjust to your model
                        features = outputs.detach().cpu().numpy()
                    all_features.extend(features)
                
                # Apply temperature scaling if provided
                if temperature_scaling is not None:
                    scaled_outputs = temperature_scaling(outputs)
                    probs = F.softmax(scaled_outputs, dim=1)
                    _, preds = torch.max(scaled_outputs, 1)
                else:
                    probs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
            
            # Store true labels and predictions for t-SNE visualization
            all_true_labels.extend(targets.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())
            
            # Store confidences and correct flags for all samples
            for idx in range(len(targets)):
                pred_class = preds[idx].item()
                confidence = probs[idx][pred_class].item()
                all_confidences.append(confidence)
                all_correct_flags.append((preds[idx] == targets[idx]).item())
            
            # Update confusion matrix
            for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                confusion_matrix[t, p] += 1
            
            # Find misclassified samples
            error_indices = (preds != targets).nonzero(as_tuple=True)[0]
            
            # Process each error
            for idx in error_indices:
                true_class = targets[idx].item()
                pred_class = preds[idx].item()
                confidence = probs[idx][pred_class].item()
                probabilities = probs[idx].cpu().numpy()
                
                # Store error information
                errors[true_class].append({
                    'predicted': pred_class,
                    'confidence': confidence,
                    'probabilities': probabilities
                })
                
                # Save error samples if requested
                if save_dir is not None:
                    # Denormalize and convert to PIL image
                    img = images[idx].cpu().numpy().transpose((1, 2, 0))
                    # Undo normalization
                    mean = np.array([0.57535914, 0.44928582, 0.40079932])
                    std = np.array([0.20735591, 0.18981615, 0.18132027])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    
                    # Create filename with true and predicted class
                    filename = f"true_{class_names[true_class]}_pred_{class_names[pred_class]}_{i}_{idx}.png"
                    img.save(os.path.join(save_dir, class_names[true_class], filename))
    
    # Generate t-SNE visualization if requested
    if args.save_tsne and all_features:
        visualize_tsne(np.array(all_features), np.array(all_true_labels), 
                       np.array(all_pred_labels), class_names, save_dir)
    
    # Generate confidence histograms if requested
    if args.save_confidence_hist:
        plot_class_confidence_histograms(all_confidences, all_correct_flags, 
                                         all_true_labels, class_names, save_dir)
    
    # Save confusion pairs to CSV
    save_confusion_pairs(confusion_matrix, class_names, save_dir)
    
    # Print error analysis per class
    print("\nError Analysis:")
    for i in range(len(class_names)):
        total_errors = len(errors[i])
        total_samples = confusion_matrix[i].sum()
        error_rate = total_errors / total_samples if total_samples > 0 else 0
        
        print(f"\nClass: {class_names[i]}")
        print(f"  Total samples: {total_samples}")
        print(f"  Correct predictions: {confusion_matrix[i, i]}")
        print(f"  Misclassifications: {total_errors}")
        print(f"  Error rate: {error_rate:.2%}")
        
        if total_errors > 0:
            # Count error distributions
            error_counts = {}
            for error in errors[i]:
                pred = error['predicted']
                if pred not in error_counts:
                    error_counts[pred] = 0
                error_counts[pred] += 1
            
            # Print error distribution
            print("  Error distribution:")
            for pred, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_errors * 100
                print(f"    Predicted as {class_names[pred]}: {count} ({percentage:.2f}%)")
            
            # Calculate average confidence for errors
            avg_confidence = sum(error['confidence'] for error in errors[i]) / total_errors
            print(f"  Average confidence of wrong predictions: {avg_confidence:.4f}")
    
    # Generate per-class metrics
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(len(class_names)):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        print(f"\nDetailed metrics for {class_names[i]}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    
    # Calculate and print macro and weighted metrics
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    
    class_weights = np.array([confusion_matrix[i].sum() for i in range(len(class_names))])
    class_weights = class_weights / class_weights.sum()
    
    weighted_precision = np.sum(np.array(precisions) * class_weights)
    weighted_recall = np.sum(np.array(recalls) * class_weights)
    weighted_f1 = np.sum(np.array(f1_scores) * class_weights)
    
    print("\nOverall Metrics:")
    print(f"  Macro Precision: {macro_precision:.4f}")
    print(f"  Macro Recall: {macro_recall:.4f}")
    print(f"  Macro F1 Score: {macro_f1:.4f}")
    print(f"  Weighted Precision: {weighted_precision:.4f}")
    print(f"  Weighted Recall: {weighted_recall:.4f}")
    print(f"  Weighted F1 Score: {weighted_f1:.4f}")
    
    return errors, confusion_matrix, f1_scores


def visualize_tsne(features, true_labels, pred_labels, class_names, output_dir):
    """Visualize feature space using t-SNE to identify clusters of errors"""
    # Apply t-SNE dimensionality reduction
    print("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Create masks for correct and incorrect predictions
    correct_mask = true_labels == pred_labels
    incorrect_mask = ~correct_mask
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot correct predictions
    for class_idx in range(len(class_names)):
        class_mask = true_labels == class_idx
        correct_class_mask = class_mask & correct_mask
        if np.sum(correct_class_mask) > 0:  # Only plot if we have samples
            plt.scatter(
                features_2d[correct_class_mask, 0],
                features_2d[correct_class_mask, 1],
                alpha=0.5, label=f'{class_names[class_idx]} (correct)'
            )
    
    # Plot incorrect predictions with a different marker
    for class_idx in range(len(class_names)):
        class_mask = true_labels == class_idx
        incorrect_class_mask = class_mask & incorrect_mask
        if np.sum(incorrect_class_mask) > 0:  # Only plot if we have samples
            plt.scatter(
                features_2d[incorrect_class_mask, 0],
                features_2d[incorrect_class_mask, 1],
                marker='x', alpha=0.8, label=f'{class_names[class_idx]} (incorrect)'
            )
    
    plt.title('t-SNE Visualization of Feature Space')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    plt.close()
    print("t-SNE visualization saved.")


def plot_class_confidence_histograms(all_confidences, all_correct_flags, true_classes, class_names, output_dir):
    """Plot confidence histograms for each class, separate for correct and incorrect predictions"""
    print("Generating confidence histograms...")
    
    # Convert to numpy arrays for easier manipulation
    all_confidences = np.array(all_confidences)
    all_correct_flags = np.array(all_correct_flags)
    true_classes = np.array(true_classes)
    
    # Overall histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_confidences[all_correct_flags], bins=20, alpha=0.7, 
             label=f'Correct Predictions ({np.sum(all_correct_flags)})', color='green')
    plt.hist(all_confidences[~all_correct_flags], bins=20, alpha=0.7, 
             label=f'Incorrect Predictions ({np.sum(~all_correct_flags)})', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Overall Confidence Distribution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'overall_confidence_hist.png'))
    plt.close()
    
    # Per-class histograms
    for class_idx, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 6))
        
        # Get confidences for this class
        class_mask = true_classes == class_idx
        if not np.any(class_mask):
            plt.close()
            continue
            
        class_confidences = all_confidences[class_mask]
        class_correct = all_correct_flags[class_mask]
        
        # Plot histograms
        correct_confidences = class_confidences[class_correct]
        incorrect_confidences = class_confidences[~class_correct]
        
        plt.hist(correct_confidences, bins=20, alpha=0.7, 
                 label=f'Correct ({len(correct_confidences)})', color='green')
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, 
                 label=f'Incorrect ({len(incorrect_confidences)})', color='red')
        
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title(f'Confidence Distribution for {class_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'confidence_hist_{class_name}.png'))
        plt.close()
    
    print("Confidence histograms saved.")


def save_confusion_pairs(confusion_matrix, class_names, output_dir):
    """Save confusion pairs to CSV and generate heatmap"""
    if output_dir is None:
        return
        
    print("Generating confusion pairs analysis...")
    
    # Create confusion pairs dataframe
    confusion_pairs = []
    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            if true_idx != pred_idx and confusion_matrix[true_idx, pred_idx] > 0:
                confusion_pairs.append({
                    'True': class_names[true_idx],
                    'Predicted': class_names[pred_idx],
                    'Count': confusion_matrix[true_idx, pred_idx],
                    'Rate': confusion_matrix[true_idx, pred_idx] / confusion_matrix[true_idx].sum()
                })
    
    # Convert to dataframe and sort
    df_confusion = pd.DataFrame(confusion_pairs)
    df_confusion = df_confusion.sort_values('Count', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'confusion_pairs.csv')
    df_confusion.to_csv(csv_path, index=False)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    conf_matrix_df = pd.DataFrame(confusion_matrix, 
                                  index=class_names, 
                                  columns=class_names)
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Confusion pairs saved to {csv_path}")


def plot_confusion_matrix(conf_matrix, class_names, output_dir):
    """Plot and save confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                 index=class_names, 
                                 columns=class_names)
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def main():
    os.makedirs('./checkpoint', exist_ok=True)
    os.makedirs('./log', exist_ok=True)
    if args.save_error_samples:
        os.makedirs(args.error_samples_dir, exist_ok=True)
    
    best_acc = 0

    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    print(f'Using device: {device}')
    print(f'Advanced loss options:')
    print(f'  - Class-Balanced Loss: {args.use_class_balanced}')
    print(f'  - Focal Loss: {args.use_focal_loss} (gamma={args.gamma})')
    print(f'  - Two-Stage Approach: {args.two_stage}')
    print(f'  - Label Smoothing: {args.use_label_smoothing} (alpha={args.label_smoothing})')
    print(f'  - Temperature Scaling: {args.use_temperature_scaling}')
    print(f'  - Class-Specific Temperature: {args.class_specific_temp}')
    print(f'  - Ensemble: {args.use_ensemble} (models={args.num_ensemble_models})')
    print(f'  - Gradient Clipping: {args.grad_clip}')
    
    if args.use_class_balanced:
        print(f'  - Beta: {args.beta}')

    # Create model
    # EfficientFace model
    print("Loading EfficientFace model...")
    model_cla = EfficientFace.efficient_face()
    try:
        # Try to load pre-trained weights
        checkpoint = torch.load('./chekpoint/Pretrained_EfficientFace.tar', map_location=device)
        pre_trained_dict = checkpoint['state_dict']
        
        # Handle DataParallel wrapping in checkpoint if needed
        new_state_dict = {}
        for k, v in pre_trained_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        
        # Load weights except for final layer
        model_dict = model_cla.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        model_cla.load_state_dict(model_dict)
        print("Pre-trained EfficientFace model loaded successfully!")
    except Exception as e:
        print(f"Error loading pre-trained EfficientFace model: {e}")
        print("Starting with randomly initialized weights...")
    
    # ======================================
    # REDESIGNED CLASSIFIER HEAD WITH MULTI-HEAD ATTENTION
    # ======================================
    in_features = model_cla.fc.in_features
    if args.use_ensemble:
        # Use standard architecture for base model
        model_cla.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            AttentionModule(512),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 7)
        )
    else:
        # Use enhanced architecture with multi-head attention for single model
        model_cla.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            MultiHeadAttentionModule(512, num_heads=4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 7)
        )
    model_cla = model_cla.to(device)
    
    # ======================================
    # TWO-STAGE APPROACH 
    # ======================================
    # If using two-stage approach, create a specialized model for difficult emotions
    if args.two_stage:
        print("Creating specialized model for difficult emotions (Fear, Disgust, Anger)...")
        model_fine = EfficientFace.efficient_face()
        model_fine.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            MultiHeadAttentionModule(256, num_heads=2),  # Use multi-head attention
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 3)  # Just Fear, Disgust, Anger
        )
        model_fine = model_fine.to(device)
    else:
        model_fine = None
    
    # Load LDG model (teacher model)
    print("Loading LDG teacher model...")
    model_dis = resnet.resnet50()
    model_dis.fc = nn.Linear(2048, 7)
    
    try:
        checkpoint = torch.load('./chekpoint/Pretrained_LDG.tar', map_location=device)
        # Handle DataParallel wrapping in checkpoint if needed
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        model_dis.load_state_dict(new_state_dict)
        print("Pre-trained LDG model loaded successfully!")
    except Exception as e:
        print(f"Error loading pre-trained LDG model: {e}")
        print("Starting with randomly initialized weights for LDG model...")
    
    model_dis = model_dis.to(device)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')

    # Print directories structure
    print(f"Train directory: {traindir}")
    print(f"Validation directory: {valdir}")
    
    # Print subdirectories to verify classes
    train_subdirs = [d for d in os.listdir(traindir) if os.path.isdir(os.path.join(traindir, d))]
    val_subdirs = [d for d in os.listdir(valdir) if os.path.isdir(os.path.join(valdir, d))]
    
    print(f"Train classes: {train_subdirs}")
    print(f"Val classes: {val_subdirs}")

    # Count images per class to understand distribution
    class_counts = {}
    for class_dir in os.listdir(traindir):
        class_path = os.path.join(traindir, class_dir)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_dir] = count
    
    # Count images per class for class-balanced loss
    class_counts_list = []
    for i in range(7):  # Assume 7 emotion classes
        class_dir = str(i)
        if class_dir in class_counts:
            class_counts_list.append(class_counts[class_dir])
        else:
            class_counts_list.append(0)  # Just in case a class is missing
    
    print("\nTraining images per class:")
    for class_dir, count in sorted(class_counts.items()):
        print(f"Class {class_dir}: {count} images")

    # Enhanced data augmentation for training
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.57535914, 0.44928582, 0.40079932],
            std=[0.20735591, 0.18981615, 0.18132027]
        )
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.57535914, 0.44928582, 0.40079932],
            std=[0.20735591, 0.18981615, 0.18132027]
        )
    ])

    # ======================================
    # TARGETED AUGMENTATION FOR MINORITY CLASSES
    # ======================================
    # Enhanced transforms for minority classes with more aggressive augmentation
    class_specific_transforms = {
        4: transforms.Compose([  # Fear class
            transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),  # More rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # More intensity
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Add affine transform
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.57535914, 0.44928582, 0.40079932],
                std=[0.20735591, 0.18981615, 0.18132027]
            )
        ]),
        5: transforms.Compose([  # Disgust class
            transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.57535914, 0.44928582, 0.40079932],
                std=[0.20735591, 0.18981615, 0.18132027]
            )
        ]),
        6: transforms.Compose([  # Anger class
            transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.57535914, 0.44928582, 0.40079932],
                std=[0.20735591, 0.18981615, 0.18132027]
            )
        ])
    }

    # Use simplified dataset approach instead of custom dataset
    train_dataset, test_dataset = get_datasets(
        traindir=traindir,
        valdir=valdir,
        train_transform=train_transforms,
        test_transform=test_transforms,
        class_specific_transforms=class_specific_transforms
    )

    print(f"Class to idx mapping: {train_dataset.class_to_idx}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(test_dataset)}")

    # Define class names for error analysis
    emotion_labels = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']

    # ======================================
    # ADJUSTED LOSS FUNCTION WITH ENHANCED WEIGHTS
    # ======================================
    # Adjust class weights more aggressively for difficult classes
    # This puts more emphasis on Fear, Disgust, and Anger classes
    custom_class_weights = torch.ones(7).to(device)
    custom_class_weights[4] = 3.5  # Fear - increased from 3.0
    custom_class_weights[5] = 3.0  # Disgust - increased from 2.5
    custom_class_weights[6] = 2.0  # Anger - increased from 1.5
    
    # Select appropriate loss function based on arguments
    if args.use_label_smoothing:
        print(f"Using Label Smoothing Loss with smoothing={args.label_smoothing}")
        criterion_train = LabelSmoothingLoss(
            classes=7,
            smoothing=args.label_smoothing,
            weight=custom_class_weights if args.use_class_balanced else None
        ).to(device)
    elif args.use_class_balanced and args.use_focal_loss:
        print(f"Using Enhanced Class-Balanced Focal Loss with beta={args.beta}, gamma={args.gamma}")
        criterion_train = ClassBalancedFocalLoss(
            samples_per_class=class_counts_list, 
            beta=args.beta, 
            gamma=args.gamma
        ).to(device)
    elif args.use_focal_loss:
        # Enhanced focal loss with stronger weighting for minority classes
        print(f"Using Enhanced Focal Loss with gamma={args.gamma}")
        print(f"Class weights: {custom_class_weights.cpu().numpy()}")
        criterion_train = EnhancedFocalLoss(gamma=args.gamma, alpha=custom_class_weights).to(device)
    elif args.use_class_balanced:
        print(f"Using Enhanced Class-Balanced Loss with beta={args.beta}")
        # Compute effective number of samples
        effective_num = 1.0 - np.power(args.beta, class_counts_list)
        # Compute weights
        weights = (1.0 - args.beta) / np.array(effective_num)
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
        # Further adjust weights for difficult classes
        weights[4] *= 1.7  # Fear - increased from 1.5
        weights[5] *= 1.5  # Disgust - increased from 1.3
        weights[6] *= 1.3  # Anger - increased from 1.2
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
        criterion_train = nn.CrossEntropyLoss(weight=class_weights).to(device)
    else:
        # Default to standard cross entropy
        criterion_train = cross_entropy
    
    criterion_val = nn.CrossEntropyLoss().to(device)
    
    # ======================================
    # TEMPERATURE SCALING
    # ======================================
    if args.use_temperature_scaling:
        if args.class_specific_temp:
            # Initialize class-specific temperature scaling model
            print("Using class-specific temperature scaling")
            temperature_scaling = ClassSpecificTemperatureScaling(num_classes=7, initial_temp=1.5).to(device)
        else:
            # Initialize standard temperature scaling model
            print("Using standard temperature scaling")
            temperature_scaling = TemperatureScaling(temperature=1.5).to(device)
    else:
        temperature_scaling = None
    
    # If using two-stage approach, create a criterion for fine model
    if args.two_stage:
        # Enhanced loss for specialized model
        if args.use_label_smoothing:
            criterion_fine = LabelSmoothingLoss(
                classes=3, 
                smoothing=args.label_smoothing,
                weight=torch.tensor([1.5, 2.0, 1.0], dtype=torch.float).to(device)
            ).to(device)
        else:
            criterion_fine = nn.CrossEntropyLoss(
                weight=torch.tensor([1.5, 2.0, 1.0], dtype=torch.float).to(device)
            ).to(device)
    else:
        criterion_fine = None
    
    # Create ensemble if requested
    if args.use_ensemble:
        print(f"Creating ensemble with {args.num_ensemble_models} models...")
        ensemble_models = create_ensemble_models(args.num_ensemble_models, model_cla, traindir, valdir)
        # Optimize weights only after training all models
    else:
        ensemble_models = None

    # Use AdamW optimizer for better convergence
    optimizer = torch.optim.AdamW(
        model_cla.parameters(),
        args.lr,
        weight_decay=args.weight_decay
    )
    
    # If using two-stage approach, create an optimizer for fine model
    if args.two_stage:
        optimizer_fine = torch.optim.AdamW(
            model_fine.parameters(),
            args.lr * 2,  # Higher learning rate for specialized model
            weight_decay=args.weight_decay
        )
    else:
        optimizer_fine = None
    
    # Create optimizers for ensemble models if using ensemble
    if args.use_ensemble:
        ensemble_optimizers = []
        for i, model in enumerate(ensemble_models):
            if i > 0:  # Skip the first model (it's already optimized by optimizer)
                ensemble_optimizers.append(
                    torch.optim.AdamW(
                        model.parameters(),
                        args.lr,
                        weight_decay=args.weight_decay
                    )
                )
    
    recorder = RecorderMeter(args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            model_cla.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        # If evaluating with error analysis
        if args.save_error_samples:
            emotion_labels = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
            errors, conf_matrix, f1_scores = analyze_errors(
                model_cla, val_loader, emotion_labels, 
                save_dir=args.error_samples_dir,
                model_fine=model_fine,
                temperature_scaling=temperature_scaling
            )
            return
        else:
            validate(val_loader, model_cla, criterion_val, args, model_fine=model_fine, temperature_scaling=temperature_scaling)
            return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # ======================================
        # IMPROVED LEARNING RATE SCHEDULE WITH WARMUP
        # ======================================
        current_learning_rate = cosine_annealing_lr(optimizer, epoch, args)
        print('Current learning rate: ', current_learning_rate)
        
        # Update learning rate for fine model if using two-stage approach
        if args.two_stage:
            fine_lr = cosine_annealing_lr(optimizer_fine, epoch, args, multiplier=2.0)
            print('Current fine model learning rate: ', fine_lr)
            
        # Update learning rate for ensemble models if using ensemble
        if args.use_ensemble:
            for i, opt in enumerate(ensemble_optimizers):
                ensemble_lr = cosine_annealing_lr(opt, epoch, args, multiplier=1.0)
                print(f'Current ensemble model {i+2} learning rate: {ensemble_lr}')
            
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')
            if args.two_stage:
                f.write('Current fine model learning rate: ' + str(fine_lr) + '\n')

        # train for one epoch
        train_acc, train_los = train(
            train_loader, model_cla, model_dis, criterion_train, optimizer, epoch, args, 
            model_fine=model_fine, 
            optimizer_fine=optimizer_fine if args.two_stage else None,
            criterion_fine=criterion_fine if args.two_stage else None,
            temperature_scaling=temperature_scaling,
            ensemble_models=ensemble_models if args.use_ensemble else None,
            ensemble_optimizers=ensemble_optimizers if args.use_ensemble else None
        )

        # evaluate on validation set
        val_acc, val_los = validate(
            val_loader, model_cla, criterion_val, args, 
            model_fine=model_fine, 
            temperature_scaling=temperature_scaling,
            ensemble_models=ensemble_models if args.use_ensemble else None
        )

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + 'log.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        best_acc = torch.tensor(0.0)

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc if isinstance(best_acc, float) else best_acc.item())
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc if isinstance(best_acc, float) else best_acc.item()) + '\n')

        # ======================================
        # CALIBRATE TEMPERATURE SCALING
        # ======================================
        # Calibrate temperature scaling after training
        if args.use_temperature_scaling and epoch > 0 and (epoch + 1) % 10 == 0:
            print("Calibrating temperature scaling...")
            # Create a standard cross entropy loss for calibration
            nll_criterion = nn.CrossEntropyLoss().to(device)
            temperature_scaling.calibrate(val_loader, model_cla, nll_criterion)
            
            # Save calibrated temperature value
            with open(txt_name, 'a') as f:
                if args.class_specific_temp:
                    f.write(f'Calibrated temperatures: {temperature_scaling.temperatures.data.cpu().numpy()}\n')
                else:
                    f.write(f'Calibrated temperature: {temperature_scaling.temperature.item()}\n')

        # Optimize ensemble weights if using ensemble
        if args.use_ensemble and epoch > 0 and (epoch + 1) % 10 == 0:
            print("Optimizing ensemble weights...")
            ensemble_model = EnsembleModel(ensemble_models)
            ensemble_model = optimize_ensemble_weights(ensemble_model, val_loader)
            
            # Save ensemble weights
            with open(txt_name, 'a') as f:
                f.write(f'Ensemble weights: {ensemble_model.weights.cpu().numpy()}\n')

        # Save main model checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_cla.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'recorder': recorder,
            'temperature': temperature_scaling.temperature.item() if temperature_scaling is not None and not args.class_specific_temp else 1.0,
            'temperatures': temperature_scaling.temperatures.data.cpu().numpy() if temperature_scaling is not None and args.class_specific_temp else None
        }, is_best, args)
        
        # Save fine model checkpoint if using two-stage approach
        if args.two_stage:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model_fine.state_dict(),
                'optimizer': optimizer_fine.state_dict(),
            }, './checkpoint/fine_model.pth.tar')
        
        # Save ensemble models if using ensemble
        if args.use_ensemble:
            for i, model in enumerate(ensemble_models):
                if i > 0:  # Skip the first model (it's already saved as model_cla)
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                    }, './checkpoint/' + f'ensemble_model_{i+1}.pth.tar')
        
        # ======================================
        # ERROR ANALYSIS EVERY 20 EPOCHS
        # ======================================
        if args.save_error_samples and (epoch + 1) % 20 == 0:
            print("Performing error analysis...")
            epoch_save_dir = os.path.join(args.error_samples_dir, f'epoch_{epoch+1}')
            
            emotion_labels = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
            errors, conf_matrix, f1_scores = analyze_errors(
                model_cla, val_loader, emotion_labels, 
                save_dir=epoch_save_dir,
                model_fine=model_fine,
                temperature_scaling=temperature_scaling
            )
            
            # Log confusion matrix and F1 scores
            with open(txt_name, 'a') as f:
                f.write(f"\nConfusion Matrix (Epoch {epoch+1}):\n")
                f.write(str(conf_matrix) + '\n')
                f.write(f"\nF1 Scores (Epoch {epoch+1}):\n")
                for i, label in enumerate(emotion_labels):
                    f.write(f"{label}: {f1_scores[i]:.4f}\n")
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')


def train(train_loader, model_cla, model_dis, criterion, optimizer, epoch, args, 
          model_fine=None, optimizer_fine=None, criterion_fine=None, temperature_scaling=None,
          ensemble_models=None, ensemble_optimizers=None):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))
    soft_max = nn.Softmax(dim=1)

    # switch mode
    model_cla.train()
    model_dis.eval()
    
    # Set fine model to train mode if using two-stage approach
    if model_fine is not None:
        model_fine.train()
    
    # Set ensemble models to train mode if using ensemble
    if ensemble_models is not None:
        for i, model in enumerate(ensemble_models):
            if i > 0:  # Skip the first model (it's already set to train mode as model_cla)
                model.train()
    
    # Debug class distribution in first epoch
    if epoch == 0:
        labels_count = [0] * 7
        for _, target in train_loader:
            for t in target:
                if t.item() < 7:  # Ensure index is valid
                    labels_count[t.item()] += 1
        print(f"Class distribution in first batch: {labels_count}")

    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model_cla(images)
        
        # Apply temperature scaling if enabled
        if temperature_scaling is not None and not args.use_class_balanced and not args.use_focal_loss:
            # Only apply in KD mode (not with focal/class-balanced loss)
            scaled_output = temperature_scaling(output)
            output_prob = soft_max(scaled_output)
        else:
            output_prob = soft_max(output)
        
        # Determine loss type based on arguments
        if args.use_focal_loss or args.use_class_balanced or args.use_label_smoothing:
            # Use the selected loss function directly
            loss = criterion(output, target)
        else:
            # Using teacher model (LDG approach)
            # compute label distribution from teacher model
            with torch.no_grad():
                soft_label = model_dis(images)
                soft_label_prob = soft_max(soft_label)
            # compute loss using soft labels from teacher model
            loss = criterion(output_prob, soft_label_prob)

        # Train fine model if using two-stage approach
        if model_fine is not None and optimizer_fine is not None and criterion_fine is not None:
            # Filter only samples from difficult classes (4: Fear, 5: Disgust, 6: Anger)
            difficult_mask = torch.isin(target, torch.tensor([4, 5, 6]).to(device))
            difficult_count = difficult_mask.sum().item()
            
            # Only train if we have MORE THAN ONE difficult sample (for batch norm)
            if difficult_count > 1:  # Require at least 2 samples for batch norm
                difficult_images = images[difficult_mask]
                # Map classes 4,5,6 to 0,1,2 for fine model
                difficult_targets = target[difficult_mask] - 4
                
                # Forward pass through fine model
                fine_output = model_fine(difficult_images)
                fine_loss = criterion_fine(fine_output, difficult_targets)
                
                # Backward and optimize for fine model
                optimizer_fine.zero_grad()
                fine_loss.backward()
                
                # Apply gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_fine.parameters(), args.grad_clip)
                
                optimizer_fine.step()
            elif difficult_count == 1:
                # If we have only one sample, log a warning but don't try to train
                print(f"Warning: Only one difficult emotion sample in batch {i}, skipping fine model update")

        # Train ensemble models if using ensemble
        if ensemble_models is not None and ensemble_optimizers is not None:
            for idx, (model, opt) in enumerate(zip(ensemble_models[1:], ensemble_optimizers)):
                # Forward pass
                ensemble_output = model(images)
                
                # Compute loss
                if args.use_focal_loss or args.use_class_balanced or args.use_label_smoothing:
                    ensemble_loss = criterion(ensemble_output, target)
                else:
                    ensemble_prob = soft_max(ensemble_output)
                    ensemble_loss = criterion(ensemble_prob, soft_label_prob)
                
                # Backward and optimize
                opt.zero_grad()
                ensemble_loss.backward()
                
                # Apply gradient clipping
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                opt.step()

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_cla.parameters(), args.grad_clip)
        
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, model_fine=None, temperature_scaling=None, ensemble_models=None):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # Track per-class accuracy
    class_correct = [0] * 7
    class_total = [0] * 7
    
    # Create confusion matrix
    confusion_matrix = np.zeros((7, 7), dtype=int)

    # switch to evaluate mode
    model.eval()
    if model_fine is not None:
        model_fine.eval()
    if temperature_scaling is not None:
        temperature_scaling.eval()
    
    # Set ensemble models to eval mode if using ensemble
    if ensemble_models is not None:
        for i, model in enumerate(ensemble_models):
            if i > 0:  # Skip the first model (it's already set to eval mode as model)
                model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # Use ensemble prediction if enabled
            if ensemble_models is not None and len(ensemble_models) > 1:
                # Get predictions from all models
                outputs = []
                for model_idx, ensemble_model in enumerate(ensemble_models):
                    model_output = ensemble_model(images)
                    
                    # Apply temperature scaling to each model if provided
                    if temperature_scaling is not None:
                        model_output = temperature_scaling(model_output)
                    
                    outputs.append(F.softmax(model_output, dim=1))
                
                # Average the predictions
                # For now, use simple averaging; in a more advanced implementation,
                # you would use the weights optimized in optimize_ensemble_weights
                ensemble_output = torch.stack(outputs).mean(dim=0)
                
                # Get predictions
                _, pred = torch.max(ensemble_output, 1)
                
                # Calculate loss using the first model's raw logits for monitoring
                output = ensemble_models[0](images)
                loss = criterion(output, target)
                
                # Update accuracy
                correct = (pred == target).float()
                acc1 = correct.sum() * 100.0 / target.size(0)
                
            # compute output - use two-stage approach if enabled
            elif args.two_stage and model_fine is not None:
                # First get prediction from main model
                output = model(images)
                
                # Apply temperature scaling if enabled
                if temperature_scaling is not None:
                    scaled_output = temperature_scaling(output)
                    # Get predictions from scaled output
                    _, pred = torch.max(scaled_output, 1)
                else:
                    # Get predictions from raw output
                    _, pred = torch.max(output, 1)
                
                # For samples predicted as difficult emotions (4,5,6), use fine model
                fine_mask = torch.isin(pred, torch.tensor([4, 5, 6]).to(device))
                
                # If we have any samples predicted as difficult emotions
                if fine_mask.sum() > 0:
                    # Get those samples
                    fine_images = images[fine_mask]
                    
                    # Get fine model predictions for these samples
                    fine_output = model_fine(fine_images)
                    
                    # Convert fine output (0,1,2) back to original classes (4,5,6)
                    fine_pred = torch.argmax(fine_output, dim=1) + 4
                    
                    # Update the predictions for these samples
                    pred[fine_mask] = fine_pred
                    
                # Use updated predictions for accuracy calculation
                correct = (pred == target).float()
                acc1 = correct.sum() * 100.0 / target.size(0)
                
                # Still compute loss on original output for monitoring
                loss = criterion(output, target)
            else:
                # Standard approach
                output = model(images)
                
                # Apply temperature scaling if enabled
                if temperature_scaling is not None:
                    scaled_output = temperature_scaling(output)
                    loss = criterion(scaled_output, target)
                    _, pred = torch.max(scaled_output, 1)
                else:
                    loss = criterion(output, target)
                    _, pred = torch.max(output, 1)
                
                correct = (pred == target).float()
                acc1 = torch.tensor(100.0 * correct.sum() / target.size(0))
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item() if isinstance(acc1, torch.Tensor) else acc1, images.size(0))
            
            # Update confusion matrix
            for t, p in zip(target.cpu().numpy(), pred.cpu().numpy()):
                confusion_matrix[t, p] += 1
                
            # Update per-class metrics
            for j in range(target.size(0)):
                label = target[j].item()
                class_correct[label] += correct[j].item()
                class_total[label] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        # ======================================
        # ADD F1 SCORE CALCULATION
        # ======================================
        # Calculate per-class F1 scores
        f1_scores = compute_f1_scores(confusion_matrix)
        
        # Print per-class metrics
        print("\nPer-class accuracy and F1 scores:")
        emotion_labels = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
        for i in range(7):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"{emotion_labels[i]}: Acc={class_acc:.2f}%, F1={f1_scores[i]:.4f} ({int(class_correct[i])}/{int(class_total[i])})")
            else:
                print(f"{emotion_labels[i]}: No samples")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        
        # Calculate macro and weighted F1
        macro_f1 = np.mean(f1_scores)
        class_weights = np.array(class_total) / sum(class_total)
        weighted_f1 = np.sum(f1_scores * class_weights)
        
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"Weighted F1 Score: {weighted_f1:.4f}")
        
        # Monitor activations - average prediction probabilities
        avg_probs = torch.zeros(7).to(device)
        count = 0
        for i, (images, _) in enumerate(val_loader):
            if i >= 10:  # Just use 10 batches to save time
                break
            images = images.to(device)
            output = model(images)
            
            # Apply temperature scaling if enabled
            if temperature_scaling is not None:
                output = temperature_scaling(output)
                
            probs = torch.nn.functional.softmax(output, dim=1)
            avg_probs += probs.sum(dim=0)
            count += images.size(0)
        
        avg_probs /= count
        print("\nAverage prediction probabilities per class:")
        for i, label in enumerate(emotion_labels):
            print(f"  {label}: {avg_probs[i].item():.6f}")
        
        # Check the distribution of predictions
        _, preds = torch.max(output, dim=1)
        pred_counts = torch.zeros(7).to(device)
        for i in range(7):
            pred_counts[i] = (preds == i).float().sum()
        pred_dist = pred_counts / pred_counts.sum()
        print("Distribution of model predictions in recent batch:")
        for i, label in enumerate(emotion_labels):
            print(f"  {label}: {pred_dist[i].item()*100:.2f}%")

        print(' *** Overall Accuracy {top1.avg:.3f}%  *** '.format(top1=top1))
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}%'.format(top1=top1) + '\n')
            f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
            f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
            
            for i in range(7):
                if class_total[i] > 0:
                    class_acc = 100 * class_correct[i] / class_total[i]
                    f.write(f"{emotion_labels[i]} Accuracy: {class_acc:.2f}%, F1={f1_scores[i]:.4f}\n")
            
            f.write(f"\nConfusion Matrix:\n{confusion_matrix}\n")
            
            f.write("\nAverage prediction probabilities per class:\n")
            for i, label in enumerate(emotion_labels):
                f.write(f"  {label}: {avg_probs[i].item():.6f}\n")
            
            f.write("Distribution of model predictions in recent batch:\n")
            for i, label in enumerate(emotion_labels):
                f.write(f"  {label}: {pred_dist[i].item()*100:.2f}%\n")
    
    return top1.avg, losses.avg


def compute_f1_scores(confusion_matrix):
    """
    Compute F1 score for each class from confusion matrix
    """
    f1_scores = []
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return f1_scores


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)


def cross_entropy(predict_label, true_label):
    """Cross entropy between two probability distributions"""
    # Add small epsilon to avoid log(0)
    predict_label = torch.clamp(predict_label, min=1e-7, max=1.0)
    return torch.mean(- true_label * torch.log(predict_label))


# ======================================
# COSINE ANNEALING LR SCHEDULE WITH WARMUP
# ======================================
def cosine_annealing_lr(optimizer, epoch, args, multiplier=1.0):
    """Cosine annealing learning rate schedule with warmup"""
    if epoch < 5:
        # Warmup for first 5 epochs
        lr = args.lr * multiplier * (epoch + 1) / 5
    else:
        # Cosine annealing after warmup
        lr = args.lr * multiplier * 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()