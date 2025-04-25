import os
import time
import joblib
import numpy as np
import sounddevice as sd
import librosa
import soundfile as sf
import tensorflow as tf
from datetime import datetime
from faster_whisper import WhisperModel
from transformers import BertTokenizer, TFBertModel

# === Config ===
SAMPLE_RATE = 48000
DURATION = 3  # seconds
N_MFCC = 13
INPUT_DEVICE = 7  # Your system's input device ID

# === Load Models ===
print("[INFO] Loading models and encoders...")
model = tf.keras.models.load_model('./models/meld_multimodal_model.keras')
label_encoder = joblib.load('./models/label_encoder.pkl')
scaler = joblib.load('./models/audio_scaler.pkl')

print(f"[INFO] Model expects {scaler.mean_.shape[0]} audio features.")

# === Load Whisper and BERT ===
whisper = WhisperModel("base", device="cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")

# === Utility Functions ===
def is_silent(audio, threshold=0.01):
    energy = np.sqrt(np.mean(audio**2))
    return energy < threshold

def extract_features(y):
    if is_silent(y) or len(y) < 1024:
        return None
    try:
        n_fft = 1024
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_fft=n_fft)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y, frame_length=n_fft)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, n_fft=n_fft)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=SAMPLE_RATE)

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

def transcribe_audio(file_path):
    segments, _ = whisper.transcribe(file_path)
    return " ".join(segment.text for segment in segments)

def embed_text(text):
    try:
        tokens = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=64)
        output = bert(**tokens)
        return output.last_hidden_state[:, 0, :].numpy().squeeze()
    except Exception as e:
        print(f"[ERROR] Text embedding failed: {e}")
        return None

def predict_multimodal(audio, temp_path="temp.wav"):
    features = extract_features(audio)
    if features is None:
        return None

    # Save to temp WAV file
    sf.write(temp_path, audio, SAMPLE_RATE)

    # Transcribe to text
    text = transcribe_audio(temp_path)
    if not text.strip():
        return None

    # Feature transformation
    audio_scaled = scaler.transform([features])
    text_embedded = embed_text(text)

    if text_embedded is None:
        return None

    prediction = model.predict([audio_scaled, text_embedded[np.newaxis, :]])
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction)

    return predicted_label[0], confidence, text

# === Live Prediction Loop ===
print("[INFO] Starting live multimodal prediction. Press Ctrl+C to stop.")
try:
    while True:
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, device=INPUT_DEVICE, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        if is_silent(audio):
            print("[INFO] Silence detected — skipping prediction.")
            continue

        result = predict_multimodal(audio)
        if result:
            label, conf, transcript = result
            print(f"[PREDICTION] {label} (Confidence: {conf:.2f}) | Text: {transcript}")

except KeyboardInterrupt:
    print("[INFO] Live prediction stopped.")