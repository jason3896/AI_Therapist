import csv
import os
import datetime

class EmotionLogger:
    def __init__(self, log_dir="data/emotion_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a new log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"emotion_log_{timestamp}.csv")
        
        # Initialize CSV file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'emotion', 'confidence', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    
    def log_emotion(self, result):
        if not result["success"]:
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emotion = result["emotion"]
        confidence = result["confidence"]
        
        # Get all emotion scores
        all_emotions = result.get("all_emotions", {})
        angry = all_emotions.get("angry", 0)
        disgust = all_emotions.get("disgust", 0)
        fear = all_emotions.get("fear", 0)
        happy = all_emotions.get("happy", 0)
        sad = all_emotions.get("sad", 0)
        surprise = all_emotions.get("surprise", 0)
        neutral = all_emotions.get("neutral", 0)
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, emotion, confidence, angry, disgust, fear, happy, sad, surprise, neutral])