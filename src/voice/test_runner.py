import joblib
import parselmouth
import numpy as np
import sys
from colorama import Fore, Style, init as colorama_init

# Init colorama
colorama_init(autoreset=True)

# Load latest model, scaler, and label encoder
model = joblib.load('models/xgb_model_20250406_205834.joblib')
scaler = joblib.load('models/scaler_20250406_205834.joblib')
label_encoder = joblib.load('models/label_encoder_20250406_205834.joblib')

def extract_features(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        mean_pitch = np.mean(pitch_values) if pitch_values.size else 0

        intensity = snd.to_intensity()
        from parselmouth.praat import call
        mean_intensity = call(intensity, "Get mean", 0, 0, "energy")

        duration = snd.get_total_duration()

        print(Fore.CYAN + f"[INFO] Extracted features - Pitch: {mean_pitch:.2f} Hz, Intensity: {mean_intensity:.2f} dB, Duration: {duration:.2f} sec" + Style.RESET_ALL)
        return np.array([mean_pitch, mean_intensity, duration]).reshape(1, -1)

    except Exception as e:
        print(Fore.RED + f"[ERROR] Failed to extract features: {e}" + Style.RESET_ALL)
        return None

def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is not None:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        print(Fore.GREEN + f"[RESULT] Predicted Emotion: {predicted_label.upper()}" + Style.RESET_ALL)
    else:
        print(Fore.RED + "[ERROR] Could not process the file for prediction." + Style.RESET_ALL)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(Fore.YELLOW + "Usage: python test_runner.py <path_to_audio.wav>" + Style.RESET_ALL)
        sys.exit(1)

    audio_file = sys.argv[1]
    predict_emotion(audio_file)
