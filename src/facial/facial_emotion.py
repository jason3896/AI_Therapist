from deepface import DeepFace
import cv2
import numpy as np

class FacialEmotionAnalyzer:
    def __init__(self):
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
    def analyze_emotion(self, frame):
        try:
            # Analyze face with DeepFace
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False
            )
            
            emotion = result[0]['dominant_emotion']
            emotion_scores = result[0]['emotion']
            confidence = emotion_scores[emotion] / 100.0
            
            return {
                "emotion": emotion.capitalize(),
                "confidence": confidence,
                "success": True,
                "all_emotions": emotion_scores
            }
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            return {
                "emotion": None,
                "confidence": 0.0,
                "success": False,
                "all_emotions": {}
            }