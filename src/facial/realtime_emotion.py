import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import time
from PIL import Image
import os
from datetime import datetime
import csv

# Import models - assuming EfficientFace is in a file called models.py
from models import EfficientFace

# Define necessary classes for model loading
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
        
        self.combine = nn.Linear(in_features, in_features)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        attention_outputs = []
        for head in self.attention_heads:
            weights = torch.sigmoid(head(x))
            attention_outputs.append(x * weights)
        
        combined = torch.stack(attention_outputs).mean(dim=0)
        return self.combine(combined)

class RecorderMeter(object):
    """Needed for checkpoint loading"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

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

# Emotion class names and display colors
EMOTION_LABELS = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
EMOTION_COLORS = [
    (255, 255, 255),  # Neutral - White
    (0, 255, 255),    # Happiness - Yellow
    (255, 0, 0),      # Sadness - Blue 
    (255, 0, 255),    # Surprise - Purple
    (0, 0, 255),      # Fear - Red
    (0, 255, 0),      # Disgust - Green
    (0, 128, 255)     # Anger - Orange
]

# Function to load the emotion recognition model
def load_model(model_path, device):
    """Load the trained model with the correct architecture"""
    print(f"Loading model from {model_path}...")
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model with the correct architecture
        model = EfficientFace.efficient_face()
        in_features = model.fc.in_features
        
        # Recreate the same architecture as used in training
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
        
        return model, temperature
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 1.0

# Function to load the fine model for difficult emotions
def load_fine_model(model_path, device):
    """Load the specialized fine model for difficult emotions"""
    if not model_path:
        return None
    
    print(f"Loading fine model from {model_path}...")
    try:
        fine_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create fine model with correct architecture
        fine_model = EfficientFace.efficient_face()
        in_features = fine_model.fc.in_features
        
        # Set up the fine model architecture
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

# Function to preprocess a face image for the model
def preprocess_face(face_img, transform):
    """Preprocess a face image for the model"""
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    img_tensor = transform(pil_img).unsqueeze(0)
    return img_tensor

# Function to predict emotion for a face
def predict_emotion(model, img_tensor, device, temperature_scaling=None, fine_model=None):
    """Predict emotion for a face image"""
    img_tensor = img_tensor.to(device)
    
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

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--fine-model', type=str, default=None, help='Path to specialized fine model (optional)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling logits')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--no-display', action='store_true', help='Start with display turned off')
    parser.add_argument('--face-detection-model', type=str, default='haarcascade',
                        choices=['haarcascade', 'dnn'],
                        help='Face detection model to use')
    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if args.device != 'cpu':
            print(f"Warning: {args.device} is not available, falling back to CPU")
    print(f"Using device: {device}")

    # Load emotion recognition model
    model, temp_value = load_model(args.model, device)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Create temperature scaling layer
    temperature = args.temperature if args.temperature > 0 else temp_value
    temperature_scaling = TemperatureScaling(temperature).to(device)
    
    # Load fine model if provided
    fine_model = load_fine_model(args.fine_model, device) if args.fine_model else None

    # Set up image preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.57535914, 0.44928582, 0.40079932],
            std=[0.20735591, 0.18981615, 0.18132027]
        )
    ])

    # Set up face detection
    if args.face_detection_model == 'haarcascade':
        # Use Haar cascade face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    else:
        # Use DNN-based face detector
        face_net = cv2.dnn.readNetFromCaffe(
            os.path.join(os.path.dirname(__file__), 'deploy.prototxt'),
            os.path.join(os.path.dirname(__file__), 'res10_300x300_ssd_iter_140000.caffemodel')
        )

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    display_on = not args.no_display
    print("Starting real-time emotion detection. Press 'q' to quit.")
    print("Press 'd' to toggle display, 'q' to quit.")
    
    # For FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    # Placeholder window for instructions
    if display_on:
        instructions_window = np.zeros((150, 400, 3), np.uint8)
        cv2.putText(instructions_window, "Press 'd' to toggle display", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(instructions_window, "Press 'q' to quit", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(instructions_window, f"Display: {'ON' if display_on else 'OFF'}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Controls', instructions_window)
        
    # Set up CSV logging
    log_dir = os.path.join(os.path.dirname(__file__), 'data', 'emotion_logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"emotion_log_{timestamp}.csv")

    csv_file = open(log_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Emotion', 'Confidence'])

    last_processed_time = 0
    desired_fps = 2.5  # <-- set to 2 for 2 FPS
    min_frame_interval = 1.0 / desired_fps

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera")
            break
        
        current_time = time.time()
        if current_time - last_processed_time < min_frame_interval:
            continue
        last_processed_time = current_time
    
        # Create a copy for visualization
        display_frame = frame.copy()
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        if args.face_detection_model == 'haarcascade':
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        else:
            # DNN-based face detection
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()
            faces = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                # Filter detections by confidence
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    # Ensure box coordinates are within frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    faces.append((x1, y1, x2-x1, y2-y1))
        
        # Process each detected face
        for face in faces:
            if args.face_detection_model == 'haarcascade':
                # For Haar cascade detector
                x, y, w, h = face
                x2, y2 = x + w, y + h
            else:
                # For DNN detector
                x, y, w, h = face
                x2, y2 = x + w, y + h
            
            # Extract face region
            face_img = frame[y:y2, x:x2]
            
            # Skip if face region is empty
            if face_img.size == 0:
                continue
            
            
            # Preprocess face for model
            try:
                img_tensor = preprocess_face(face_img, transform)
                
                # Predict emotion
                pred_idx, pred_label, confidence, all_probs = predict_emotion(
                    model, img_tensor, device, temperature_scaling, fine_model
                )
                
                # Log to CSV in desired format
                log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([log_time, pred_label, confidence])
                csv_file.flush()  # Force write to disk immediately

                
                if display_on:
                    # Draw bounding box and emotion label
                    color = EMOTION_COLORS[pred_idx]
                    cv2.rectangle(display_frame, (x, y), (x2, y2), color, 2)
                    
                    # Draw emotion label with confidence
                    label_text = f"{pred_label}: {confidence:.2f}"
                    cv2.putText(display_frame, label_text, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw probability bars for each emotion
                    bar_height = 15
                    max_bar_width = 100
                    start_x = x
                    start_y = y2 + 20
                    
                    for i, prob in enumerate(all_probs):
                        # Draw emotion label
                        label = EMOTION_LABELS[i]
                        cv2.putText(display_frame, label, (start_x, start_y + i*bar_height), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Draw probability bar
                        bar_width = int(prob * max_bar_width)
                        cv2.rectangle(display_frame, (start_x + 70, start_y + i*bar_height - 10),
                                    (start_x + 70 + bar_width, start_y + i*bar_height - 2),
                                    EMOTION_COLORS[i], -1)
            
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Calculate and display FPS
        fps_counter += 1
        time_elapsed = time.time() - fps_start_time
        if time_elapsed > 1.0:  # Update FPS every second
            fps = fps_counter / time_elapsed
            fps_counter = 0
            fps_start_time = time.time()
            
        # Show display if it's on
        if display_on:
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Real-time Emotion Detection', display_frame)
            
            # Update the instructions window with current display state
            instructions_window = np.zeros((150, 400, 3), np.uint8)
            cv2.putText(instructions_window, "Press 'd' to toggle display", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(instructions_window, "Press 'q' to quit", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(instructions_window, f"Display: {'ON' if display_on else 'OFF'}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Controls', instructions_window)
        
        # Check for keys
        key = cv2.waitKey(1) & 0xFF
        
        # Toggle display with 'd' key
        if key == ord('d'):
            display_on = not display_on
            print(f"Display turned {'ON' if display_on else 'OFF'}")
            
            if display_on:
                # Create windows if they don't exist
                instructions_window = np.zeros((150, 400, 3), np.uint8)
                cv2.putText(instructions_window, "Press 'd' to toggle display", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(instructions_window, "Press 'q' to quit", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(instructions_window, "Display: ON", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Controls', instructions_window)
                cv2.imshow('Real-time Emotion Detection', display_frame)
            else:
                # Close all windows
                cv2.destroyAllWindows()
        
        # Quit with 'q' key
        elif key == ord('q'):
            break
        
    # Clean up
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()