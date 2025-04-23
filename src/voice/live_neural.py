import os
import time
import joblib
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from datetime import datetime

"""
Real-Time Emotion Predictor (Mic Input)

This script continuously records 1-second audio snippets from your microphone,
extracts audio features (MFCC, chroma, RMS, etc.), and predicts the speaker's
emotion using a pretrained deep learning model.

- Uses sounddevice for audio capture
- Extracts features with librosa
- Loads Keras model, scaler, and label encoder
- Displays emotion predictions and confidence in real-time
"""

# Config
SAMPLE_RATE = 48000
DURATION = 3  # seconds
N_MFCC = 13
INPUT_DEVICE = 7  # Confirmed from your tests

# Load models and encoders
print("[INFO] Loading models and encoders...")
model = tf.keras.models.load_model('./models/emotion_model.keras')
label_encoder = joblib.load('./models/label_encoder.pkl')
scaler = joblib.load('./models/scaler.pkl')

expected_features = scaler.mean_.shape[0]
print(f"[INFO] Model expects {expected_features} features.")
print(f"[INFO] Using sample rate: {SAMPLE_RATE}")

def is_silent(audio, threshold=0.01):
    energy = np.sqrt(np.mean(audio**2))
    return energy < threshold


def interpolate_to_length(y, target_length):
    current_length = len(y)
    if current_length >= target_length:
        return y

    x_old = np.linspace(0, 1, num=current_length)
    x_new = np.linspace(0, 1, num=target_length)
    y_interp = np.interp(x_new, x_old, y)
    return y_interp


def extract_features(y):
    if is_silent(y) or len(y) < 1024:
        return None

    try:
        sr = SAMPLE_RATE
        n_fft = 1024  # fixed size, consistent with training

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y, frame_length=n_fft)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(zcr),
            np.mean(rms),
            np.mean(spec_contrast, axis=1),
            np.mean(tonnetz, axis=1)
        ])
        return features
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None

    
def predict(audio):
    features = extract_features(audio)
    if features is None:
        return None

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction)

    return predicted_label[0], confidence

# Live prediction loop
print("[INFO] Starting live prediction. Press Ctrl+C to stop.")
try:
    while True:
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, device=INPUT_DEVICE, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        if is_silent(audio):
            print("[INFO] Silence detected — skipping prediction.")
            continue

        result = predict(audio)
        if result:
            label, conf = result
            print(f"[PREDICTION] {label} (Confidence: {conf:.2f})")

except KeyboardInterrupt:
    print("[INFO] Live prediction stopped.")
