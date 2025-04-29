import os
import datetime

class VocalLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"vocal_log_{timestamp}.csv")
        
        with open(self.log_file, 'w') as f:
            f.write(f"[INFO] Vocal Log started at {timestamp}\n")

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)  # still prints to console
        with open(self.log_file, 'a') as f:
            f.write(full_message + "\n")
