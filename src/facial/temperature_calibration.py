import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os
from models import EfficientFace

# Set up argument parser
parser = argparse.ArgumentParser(description='Temperature scaling calibration for emotion recognition model')
parser.add_argument('--data', type=str, required=True, help='Path to validation data directory')
parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
parser.add_argument('--output', type=str, default='./calibrated_model.pth.tar', help='Path to save calibrated model')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for calibration')
parser.add_argument('--initial-temp', type=float, default=1.0, help='Initial temperature value')
parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda, mps, cpu)')
args = parser.parse_args()

# RecorderMeter class needed for checkpoint loading
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

# Set device
if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    if args.device != 'cpu':
        print(f"Warning: {args.device} is not available, falling back to CPU")
print(f"Using device: {device}")

# Temperature scaling module
class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def forward(self, logits):
        return logits / self.temperature

def evaluate_calibration(model, temperature_model, val_loader, criterion):
    """Evaluate the calibration metrics of the model"""
    model.eval()
    temperature_model.eval()
    
    losses = []
    accuracies = []
    confidences = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            scaled_outputs = temperature_model(outputs)
            
            # Calculate NLL loss
            loss = criterion(scaled_outputs, targets)
            losses.append(loss.item())
            
            # Calculate accuracy
            _, preds = torch.max(scaled_outputs, 1)
            acc = (preds == targets).float().mean().item()
            accuracies.append(acc)
            
            # Calculate confidence
            probs = torch.softmax(scaled_outputs, dim=1)
            confidence = probs.max(dim=1)[0].mean().item()
            confidences.append(confidence)
    
    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)
    avg_conf = np.mean(confidences)
    
    print(f"Evaluation Results:")
    print(f"  NLL Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  Average Confidence: {avg_conf:.4f}")
    print(f"  Confidence Gap: {avg_conf - avg_acc:.4f} (lower is better)")
    
    return avg_loss

def main():
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
    
    # Load the model - with fallback options
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
    
    # Create temperature scaling model
    temperature_model = TemperatureScaling(args.initial_temp).to(device)
    
    # Use NLL loss for calibration
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Evaluate before calibration
    print("Before calibration:")
    evaluate_calibration(model, temperature_model, val_loader, criterion)
    
    # Optimize temperature parameter
    print("\nOptimizing temperature parameter...")
    optimizer = torch.optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)
    
    def eval_fn():
        optimizer.zero_grad()
        total_loss = 0.0
        num_samples = 0
        
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            scaled_outputs = temperature_model(outputs)
            loss = criterion(scaled_outputs, targets)
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
        
        avg_loss = total_loss / num_samples
        loss_tensor = torch.tensor(avg_loss, requires_grad=True, device=device)
        loss_tensor.backward()
        
        print(f"Current temperature: {temperature_model.temperature.item():.6f}, Loss: {avg_loss:.6f}")
        return loss_tensor
    
    optimizer.step(eval_fn)
    
    # Evaluate after calibration
    print("\nAfter calibration:")
    evaluate_calibration(model, temperature_model, val_loader, criterion)
    
    # Save the calibrated model
    final_temp = temperature_model.temperature.item()
    print(f"\nFinal temperature value: {final_temp:.6f}")
    
    # Prepare new checkpoint with temperature value
    new_checkpoint = {
        'state_dict': model.state_dict(),
        'temperature': final_temp
    }
    
    # Copy over other keys from original checkpoint if they exist
    for key in checkpoint:
        if key != 'state_dict' and key != 'temperature':
            new_checkpoint[key] = checkpoint[key]
    
    # Save the checkpoint
    print(f"Saving calibrated model to {args.output}")
    torch.save(new_checkpoint, args.output, _use_new_zipfile_serialization=True)
    
    print("Temperature calibration complete!")

if __name__ == "__main__":
    main()