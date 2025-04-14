import sounddevice as sd
import numpy as np
import joblib
import librosa
import parselmouth
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load models
print("[INFO] Loading models and encoders...")
model = joblib.load('./models/emotion_model.pkl')
scaler = joblib.load('./models/scaler.pkl')
label_encoder = joblib.load('./models/label_encoder.pkl')
print(f"[INFO] Model expects {model.named_steps['clf'].n_features_in_} features.")

# Audio settings
INPUT_DEVICE = 7  # Confirmed from your list
SAMPLE_RATE = 48000
DURATION = 1  # seconds

print("[INFO] Using sample rate:", SAMPLE_RATE)

# === Extract features (exact copy from train_full.py) ===
def extract_features_live(y, sr):
    try:
        if len(y) < 512:
            return None

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft, axis=1)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))

        pitch = parselmouth.Sound(y, sampling_frequency=sr).to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_mean = np.nanmean(pitch_values[pitch_values > 0]) if np.any(pitch_values > 0) else 0

        features = np.concatenate([
            mfccs_mean,
            chroma_stft_mean,
            spectral_contrast_mean,
            tonnetz_mean,
            [zero_crossing_rate, rms, pitch_mean]
        ])

        return features

    except Exception:
        return None

print("[INFO] Starting live prediction. Press Ctrl+C to stop.")

try:
    while True:
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE,
                       channels=1, device=INPUT_DEVICE, dtype='float32')
        sd.wait()

        y = audio.flatten()
        features = extract_features_live(y, SAMPLE_RATE)

        if features is None:
            continue

        if features.shape[0] != model.named_steps['clf'].n_features_in_:
            print(f"[WARN] Feature size mismatch: {features.shape[0]} instead of {model.named_steps['clf'].n_features_in_}")
            continue

        X_scaled = scaler.transform([features])
        pred = model.predict(X_scaled)
        label = label_encoder.inverse_transform(pred)[0]
        print(f"[PREDICTION] {label}")

except KeyboardInterrupt:
    print("[INFO] Live prediction stopped.")
