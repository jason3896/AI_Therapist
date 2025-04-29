import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from models import EfficientFace
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# RecorderMeter class needed for checkpoint loading
class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        pass  # We don't need the plotting functionality here

# Add the AttentionModule class needed for model architecture
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

# Add needed classes to safe globals
from torch.serialization import add_safe_globals
add_safe_globals([RecorderMeter, AttentionModule])
try:
    import numpy
    from numpy.core.multiarray import _reconstruct
    add_safe_globals([_reconstruct])
except:
    print("Warning: Could not add numpy _reconstruct to safe globals")

# Set up argument parser
parser = argparse.ArgumentParser(description='Error analysis for emotion recognition model')
parser.add_argument('--data', type=str, required=True, help='Path to validation data directory')
parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
parser.add_argument('--output-dir', type=str, default='./error_analysis', help='Directory to save error analysis')
parser.add_argument('--fine-model', type=str, default=None, help='Path to specialized fine model (optional)')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda, mps, cpu)')
parser.add_argument('--visualize', action='store_true', help='Save visualizations of misclassified samples')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling logits')
args = parser.parse_args()

# Set device
if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    if args.device != 'cpu':
        print(f"Warning: {args.device} is not available, falling back to CPU")
print(f"Using device: {device}")

# Emotion class names
EMOTION_LABELS = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']

class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def forward(self, logits):
        return logits / self.temperature

def analyze_errors(model, data_loader, output_dir, fine_model=None, temperature_scaling=None):
    """Analyze errors made by the model and save misclassified samples."""
    model.eval()
    if fine_model is not None:
        fine_model.eval()
    
    # Create directories to save error samples
    os.makedirs(output_dir, exist_ok=True)
    for label in EMOTION_LABELS:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    
    # Storage for error analysis
    errors = {i: [] for i in range(len(EMOTION_LABELS))}
    conf_matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)), dtype=int)
    all_targets = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images, targets = images.to(device), targets.to(device)
            all_targets.extend(targets.cpu().numpy())
            
            # Get predictions
            outputs = model(images)
            
            # Apply temperature scaling if provided
            if temperature_scaling is not None:
                outputs = temperature_scaling(outputs)
            
            # Standard prediction
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Use fine model for difficult emotions if provided
            if fine_model is not None:
                # For samples predicted as difficult emotions (4,5,6), use fine model
                fine_mask = torch.isin(preds, torch.tensor([4, 5, 6]).to(device))
                
                # If we have any samples predicted as difficult emotions
                if fine_mask.sum() > 0:
                    # Get those samples
                    fine_images = images[fine_mask]
                    
                    # Get fine model predictions
                    fine_outputs = fine_model(fine_images)
                    
                    # Convert fine output (0,1,2) back to original classes (4,5,6)
                    fine_preds = torch.argmax(fine_outputs, dim=1) + 4
                    
                    # Update the predictions
                    preds[fine_mask] = fine_preds
            
            # Store predictions and probabilities
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update confusion matrix
            for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                conf_matrix[t, p] += 1
            
            # Find misclassified samples
            error_indices = (preds != targets).nonzero(as_tuple=True)[0]
            
            # Process each error
            for idx in error_indices:
                true_class = targets[idx].item()
                pred_class = preds[idx].item()
                confidence = probs[idx][pred_class].item()
                
                # Store error information
                errors[true_class].append({
                    'image_idx': i * data_loader.batch_size + idx.item(),
                    'predicted': pred_class,
                    'confidence': confidence,
                    'probabilities': probs[idx].cpu().numpy(),
                    'image': images[idx].cpu()
                })
                
                # Save error samples if requested
                if args.visualize:
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
                    filename = f"true_{EMOTION_LABELS[true_class]}_pred_{EMOTION_LABELS[pred_class]}_{i}_{idx}.png"
                    img.save(os.path.join(output_dir, EMOTION_LABELS[true_class], filename))
    
    return errors, conf_matrix, all_targets, all_preds, all_probs

def plot_confusion_matrix(conf_matrix, output_dir):
    """Plot and save confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curves(all_targets, all_probs, output_dir):
    """Plot ROC curves for each emotion class"""
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    plt.figure(figsize=(12, 10))
    
    # One-hot encode targets for ROC curve calculation
    n_classes = len(EMOTION_LABELS)
    y_true_one_hot = np.zeros((len(all_targets), n_classes))
    for i, t in enumerate(all_targets):
        y_true_one_hot[i, t] = 1
    
    # Calculate ROC curve and AUC for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label=f'{EMOTION_LABELS[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

def plot_confidence_distributions(errors, output_dir):
    """Plot confidence distributions for correctly and incorrectly classified samples"""
    # Gather confidence values
    correct_confidences = []
    error_confidences = []
    
    for class_idx, class_errors in errors.items():
        for error in class_errors:
            error_confidences.append(error['confidence'])
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(error_confidences, bins=20, alpha=0.7, label='Incorrect predictions')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution for Incorrectly Classified Samples')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'error_confidence_distribution.png'))
    plt.close()

def print_error_statistics(errors, conf_matrix):
    """Print detailed error statistics"""
    print("\nError Analysis:")
    
    # Calculate per-class metrics
    precision = np.zeros(len(EMOTION_LABELS))
    recall = np.zeros(len(EMOTION_LABELS))
    f1_scores = np.zeros(len(EMOTION_LABELS))
    
    for i in range(len(EMOTION_LABELS)):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_scores[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        
        total_samples = np.sum(conf_matrix[i, :])
        total_errors = len(errors[i])
        error_rate = total_errors / total_samples if total_samples > 0 else 0
        
        print(f"\nClass: {EMOTION_LABELS[i]}")
        print(f"  Total samples: {total_samples}")
        print(f"  Correct predictions: {conf_matrix[i, i]}")
        print(f"  Misclassifications: {total_errors}")
        print(f"  Error rate: {error_rate:.2%}")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1 Score: {f1_scores[i]:.4f}")
        
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
                print(f"    Predicted as {EMOTION_LABELS[pred]}: {count} ({percentage:.2f}%)")
            
            # Calculate average confidence for errors
            avg_confidence = sum(error['confidence'] for error in errors[i]) / total_errors
            print(f"  Average confidence of wrong predictions: {avg_confidence:.4f}")
    
    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"  Macro Precision: {np.mean(precision):.4f}")
    print(f"  Macro Recall: {np.mean(recall):.4f}")
    print(f"  Macro F1 Score: {np.mean(f1_scores):.4f}")
    
    # Calculate weighted metrics
    weights = np.array([np.sum(conf_matrix[i, :]) for i in range(len(EMOTION_LABELS))])
    weights = weights / np.sum(weights)
    
    print(f"  Weighted Precision: {np.sum(precision * weights):.4f}")
    print(f"  Weighted Recall: {np.sum(recall * weights):.4f}")
    print(f"  Weighted F1 Score: {np.sum(f1_scores * weights):.4f}")
    
    # Calculate accuracy
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f"  Overall Accuracy: {accuracy:.4f}")

def analyze_high_confidence_errors(errors, output_dir, threshold=0.9):
    """Analyze errors where the model was highly confident but wrong"""
    high_conf_errors = []
    
    for class_idx, class_errors in errors.items():
        for error in class_errors:
            if error['confidence'] >= threshold:
                high_conf_errors.append({
                    'true_class': class_idx,
                    'predicted': error['predicted'],
                    'confidence': error['confidence'],
                    'image': error['image']
                })
    
    if high_conf_errors:
        print(f"\nHigh Confidence Errors (confidence >= {threshold}):")
        print(f"  Found {len(high_conf_errors)} high confidence errors")
        
        # Group by true/predicted class pairs
        pair_counts = {}
        for error in high_conf_errors:
            pair = (error['true_class'], error['predicted'])
            if pair not in pair_counts:
                pair_counts[pair] = 0
            pair_counts[pair] += 1
        
        print("  Distribution of high confidence errors:")
        for (true_idx, pred_idx), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
            true_class = EMOTION_LABELS[true_idx]
            pred_class = EMOTION_LABELS[pred_idx]
            percentage = count / len(high_conf_errors) * 100
            print(f"    True: {true_class}, Predicted: {pred_class}: {count} ({percentage:.2f}%)")
        
        # Save high confidence error examples if visualize is enabled
        if args.visualize:
            high_conf_dir = os.path.join(output_dir, "high_confidence_errors")
            os.makedirs(high_conf_dir, exist_ok=True)
            
            for i, error in enumerate(high_conf_errors[:min(20, len(high_conf_errors))]):
                # Convert tensor to image
                img = error['image'].numpy().transpose((1, 2, 0))
                # Undo normalization
                mean = np.array([0.57535914, 0.44928582, 0.40079932])
                std = np.array([0.20735591, 0.18981615, 0.18132027])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                
                true_class = EMOTION_LABELS[error['true_class']]
                pred_class = EMOTION_LABELS[error['predicted']]
                filename = f"high_conf_{i}_true_{true_class}_pred_{pred_class}_{error['confidence']:.2f}.png"
                img.save(os.path.join(high_conf_dir, filename))

def main():
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up data transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.57535914, 0.44928582, 0.40079932],
            std=[0.20735591, 0.18981615, 0.18132027]
        )
    ])
    
    # Create validation dataset
    val_dir = args.data
    val_dataset = datasets.ImageFolder(val_dir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load the main model - with fallback options
    print(f"Loading model from {args.model}...")
    try:
        # Try direct loading with weights_only=False (most permissive)
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please make sure the checkpoint file exists and is accessible.")
        return
    
    print("Checkpoint loaded successfully.")
    
    # Create model with the correct architecture matching the checkpoint
    model = EfficientFace.efficient_face()
    in_features = model.fc.in_features
    
    # Recreate the same architecture as used in training
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),  # Less dropout (was 0.5)
        
        # Add attention after first layer
        AttentionModule(512),
        
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),  # Less dropout (was 0.5)
        
        # Add another layer for more representation power
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        
        nn.Linear(128, 7)
    )
    
    # Load state dict
    print("Loading state dictionary into model...")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load the fine model if provided
    fine_model = None
    if args.fine_model is not None:
        print(f"Loading fine model from {args.fine_model}...")
        try:
            fine_checkpoint = torch.load(args.fine_model, map_location=device, weights_only=False)
            
            # Create fine model with correct architecture
            fine_model = EfficientFace.efficient_face()
            in_features = fine_model.fc.in_features
            
            # Set up the fine model architecture to match training
            fine_model.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                
                AttentionModule(256),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                
                nn.Linear(64, 3)  # Just Fear, Disgust, Anger
            )
            
            fine_model.load_state_dict(fine_checkpoint['state_dict'])
            fine_model = fine_model.to(device)
            fine_model.eval()
            print("Fine model loaded successfully.")
        except Exception as e:
            print(f"Error loading fine model: {e}")
            print("Will continue without the fine model.")
            fine_model = None
    
    # Create temperature scaling model if needed
    temperature_scaling = None
    if args.temperature != 1.0:
        print(f"Using temperature scaling with T={args.temperature}")
        temperature_scaling = TemperatureScaling(args.temperature).to(device)
    
    # Perform error analysis
    print("Analyzing model errors...")
    errors, conf_matrix, all_targets, all_preds, all_probs = analyze_errors(
        model, val_loader, args.output_dir, fine_model, temperature_scaling
    )
    
    # Print error statistics
    print_error_statistics(errors, conf_matrix)
    
    # Analyze high confidence errors
    analyze_high_confidence_errors(errors, args.output_dir, threshold=0.9)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_confusion_matrix(conf_matrix, args.output_dir)
    plot_roc_curves(all_targets, all_probs, args.output_dir)
    plot_confidence_distributions(errors, args.output_dir)
    
    # Save detailed classification report
    target_names = EMOTION_LABELS
    report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)
    
    # Convert to DataFrame for better formatting
    import pandas as pd
    df_report = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(df_report)
    
    # Save report to CSV
    df_report.to_csv(os.path.join(args.output_dir, 'classification_report.csv'))
    
    print(f"\nError analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()