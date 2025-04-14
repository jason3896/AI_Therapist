import os
import time
import joblib
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from datetime import datetime

# Config
SAMPLE_RATE = 48000
DURATION = 1  # seconds
N_MFCC = 13
INPUT_DEVICE = 7  # Confirmed from your tests

# Load models and encoders
print("[INFO] Loading models and encoders...")
model = tf.keras.models.load_model('./models/emotion_model.pkl')
label_encoder = joblib.load('./models/label_encoder.pkl')
scaler = joblib.load('./models/scaler.pkl')

expected_features = scaler.mean_.shape[0]
print(f"[INFO] Model expects {expected_features} features.")
print(f"[INFO] Using sample rate: {SAMPLE_RATE}")

def extract_features(y):
    if len(y) < 2048:
        return None

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=SAMPLE_RATE)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(zcr),
            np.mean(rms),
            np.mean(spec_contrast, axis=1),
            np.mean(tonnetz, axis=1),
        ])

        if features.shape[0] != expected_features:
            print(f"[WARN] Feature size mismatch: {features.shape[0]} instead of {expected_features}")
            return None

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

        result = predict(audio)
        if result:
            label, conf = result
            print(f"[PREDICTION] {label} (Confidence: {conf:.2f})")

except KeyboardInterrupt:
    print("[INFO] Live prediction stopped.")
