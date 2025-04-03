import cv2
import time
import os
from facial_emotion import FacialEmotionAnalyzer
from emotion_logger import EmotionLogger

def start_emotion_detection():
    # Initialize webcam - '0' is usually the built-in webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set frame dimensions (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize the emotion analyzer
    analyzer = FacialEmotionAnalyzer()
    
    # Initialize the emotion logger
    logger = EmotionLogger()
    
    # Create export directory for emotion data
    export_dir = "data/emotion_exports"
    os.makedirs(export_dir, exist_ok=True)
    
    # Variables to control analysis frequency
    last_analysis_time = 0
    analysis_interval = 1  # Analyze every 1 second
    
    # Dictionary to store the last valid emotion result
    last_result = {"emotion": "Unknown", "confidence": 0.0, "all_emotions": {}}
    
    print("Starting emotion detection. Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Create a copy of frame for display
        display_frame = frame.copy()
        
        # Get current time
        current_time = time.time()
        
        # Analyze emotion at regular intervals
        if current_time - last_analysis_time >= analysis_interval:
            result = analyzer.analyze_emotion(frame)
            
            if result["success"]:
                last_result = result
                # Log the emotion to CSV
                logger.log_emotion(result)
                
                # Export the emotion for other modules
                export_emotion(result, export_dir)
                
            # Update the time of last analysis
            last_analysis_time = current_time
        
        # Display the emotion result on the frame
        emotion_text = f"Emotion: {last_result['emotion']}"
        confidence_text = f"Confidence: {last_result['confidence']:.2f}"
        
        # Add text to the frame
        cv2.putText(
            display_frame, 
            emotion_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )
        
        cv2.putText(
            display_frame, 
            confidence_text, 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )
        
        # Display the frame
        cv2.imshow('Emotion Detection', display_frame)
        
        # Check for exit (press 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Emotion detection stopped.")

def export_emotion(result, export_dir):
    """Export the latest emotion result to a JSON file for other modules to read"""
    import json
    
    if not result["success"]:
        return
    
    export_file = os.path.join(export_dir, "latest_emotion.json")
    
    with open(export_file, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    start_emotion_detection()