import json
import os

class EmotionExporter:
    def __init__(self, export_dir="data/emotion_exports"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_latest_emotion(self, result):
        """Export the latest emotion result to a JSON file for other modules to read"""
        if not result["success"]:
            return
        
        export_file = os.path.join(self.export_dir, "latest_emotion.json")
        
        with open(export_file, 'w') as f:
            json.dump(result, f)