import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from models import EfficientFace
import glob
from tqdm import tqdm

# Define necessary classes for model loading
class RecorderMeter(object):
    """Needed for checkpoint loading"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

class AttentionModule(nn.Module):
    """Attention module for the model architecture"""
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

# Add MultiHeadAttentionModule for compatibility with updated model architecture
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

# Temperature scaling for calibrated confidence
class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def forward(self, logits):
        return logits / self.temperature

# Add required classes to safe globals
from torch.serialization import add_safe_globals
add_safe_globals([RecorderMeter, AttentionModule, MultiHeadAttentionModule])
try:
    import numpy
    from numpy.core.multiarray import _reconstruct
    add_safe_globals([_reconstruct])
except:
    print("Warning: Could not add numpy _reconstruct to safe globals")

# Set up argument parser
parser = argparse.ArgumentParser(description='Test emotion recognition model on images')
parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
parser.add_argument('--test-dir', type=str, default='./test_image', help='Directory containing test images')
parser.add_argument('--output-dir', type=str, default='./test_results', help='Directory to save results')
parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda, mps, cpu)')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling logits')
parser.add_argument('--visualize', action='store_true', help='Save visualizations of predictions')
parser.add_argument('--fine-model', type=str, default=None, help='Path to specialized fine model (optional)')
parser.add_argument('--use-multihead', action='store_true', help='Use multi-head attention in model architecture')
parser.add_argument('--fine-use-multihead', action='store_true', help='Use multi-head attention in fine model architecture')
args = parser.parse_args()

# Emotion class names
EMOTION_LABELS = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
EMOTION_COLORS = ['gray', 'yellow', 'blue', 'purple', 'red', 'green', 'orange']

# Set device
if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    if args.device != 'cpu':
        print(f"Warning: {args.device} is not available, falling back to CPU")
print(f"Using device: {device}")

def load_model(model_path):
    """Load the trained model with the correct architecture"""
    print(f"Loading model from {model_path}...")
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model with the correct architecture
        model = EfficientFace.efficient_face()
        in_features = model.fc.in_features
        
        # Recreate the same architecture as used in training
        if args.use_multihead:
            # Use multi-head attention architecture
            model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                
                # Add multi-head attention
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
        else:
            # Use standard attention architecture
            model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                
                # Add attention after first layer
                AttentionModule(512),
                
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                
                # Add another layer for more representation power
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                
                nn.Linear(128, 7)
            )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        
        # Check if model has a temperature value
        temperature = 1.0
        if 'temperature' in checkpoint:
            temperature = checkpoint['temperature']
            print(f"Using temperature value from checkpoint: {temperature}")
        else:
            temperature = args.temperature
            print(f"Using provided temperature value: {temperature}")
        
        return model, temperature
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 1.0

def load_fine_model(model_path):
    """Load the specialized fine model for difficult emotions"""
    if not model_path:
        return None
    
    print(f"Loading fine model from {model_path}...")
    try:
        fine_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create fine model with correct architecture
        fine_model = EfficientFace.efficient_face()
        in_features = fine_model.fc.in_features
        
        # Set up the fine model architecture based on flag
        if args.fine_use_multihead:
            # Use multi-head attention for fine model
            fine_model.fc = nn.Sequential(
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
        else:
            # Use standard attention for fine model
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
        return fine_model
    except Exception as e:
        print(f"Error loading fine model: {e}")
        return None

def preprocess_image(image_path):
    """Preprocess a single image for the model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.57535914, 0.44928582, 0.40079932],
            std=[0.20735591, 0.18981615, 0.18132027]
        )
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        return img_tensor, img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def predict_emotion(model, img_tensor, temperature_scaling=None, fine_model=None):
    """Predict emotion for a single image"""
    with torch.no_grad():
        # Get model output
        output = model(img_tensor)
        
        # Apply temperature scaling if provided
        if temperature_scaling is not None:
            output = temperature_scaling(output)
        
        # Get probabilities and prediction
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(output, dim=1).item()
        
        # Use fine model for difficult emotions if provided
        if fine_model is not None and pred_idx in [4, 5, 6]:
            # Get fine model prediction
            fine_output = fine_model(img_tensor)
            fine_pred = torch.argmax(fine_output, dim=1).item()
            # Convert fine output (0,1,2) back to original classes (4,5,6)
            pred_idx = fine_pred + 4
            
        pred_label = EMOTION_LABELS[pred_idx]
        confidence = probs[0, pred_idx].item()
        
        # Get all class probabilities
        all_probs = probs[0].cpu().numpy()
        
        return pred_idx, pred_label, confidence, all_probs

def visualize_prediction(image, pred_idx, confidence, all_probs, save_path):
    """Create a visualization of the prediction"""
    plt.figure(figsize=(10, 5))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Predicted: {EMOTION_LABELS[pred_idx]}\nConfidence: {confidence:.2f}")
    plt.axis('off')
    
    # Display the probability distribution
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(len(EMOTION_LABELS)), all_probs, color=EMOTION_COLORS)
    plt.xticks(range(len(EMOTION_LABELS)), EMOTION_LABELS, rotation=45)
    plt.ylabel('Probability')
    plt.title('Emotion Probabilities')
    
    # Highlight the predicted class
    bars[pred_idx].set_color('darkred')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    model, temperature = load_model(args.model)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Create temperature scaling layer
    temperature_scaling = TemperatureScaling(temperature).to(device)
    
    # Load fine model if provided
    fine_model = load_fine_model(args.fine_model)
    
    # Get list of image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.test_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.test_dir, '**', ext), recursive=True))
    
    if not image_files:
        print(f"No image files found in {args.test_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create a CSV file for results
    import csv
    csv_path = os.path.join(args.output_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Predicted Emotion', 'Confidence'] + EMOTION_LABELS)
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            # Get image name (for results)
            image_name = os.path.basename(image_path)
            
            # Preprocess the image
            img_tensor, img = preprocess_image(image_path)
            if img_tensor is None:
                continue
            
            # Get prediction
            pred_idx, pred_label, confidence, all_probs = predict_emotion(
                model, img_tensor, temperature_scaling, fine_model
            )
            
            # Save to CSV
            writer.writerow([image_name, pred_label, f"{confidence:.4f}"] + [f"{p:.4f}" for p in all_probs])
            
            # Visualize if requested
            if args.visualize:
                viz_dir = os.path.join(args.output_dir, 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                viz_path = os.path.join(viz_dir, f"{os.path.splitext(image_name)[0]}_pred.png")
                visualize_prediction(img, pred_idx, confidence, all_probs, viz_path)
    
    print(f"Results saved to {csv_path}")
    if args.visualize:
        print(f"Visualizations saved to {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()