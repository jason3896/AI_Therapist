import os
import time
import joblib
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import csv
from datetime import datetime
from transformers import BertTokenizer, TFBertModel
from faster_whisper import WhisperModel

# === Vocal Logger ===
import csv
import os
from datetime import datetime

class VocalLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"emotion_log_{timestamp}.csv")
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "emotion", "confidence"])

    def log_prediction(self, emotion, confidence, words=""):
        if emotion.lower() == "silent":
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([timestamp, emotion, f"{confidence:.6f}"])

    def log_console(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")



# === Config ===
SAMPLE_RATE = 16000  # Required by YAMNet
DURATION = 5  # seconds
INPUT_DEVICE = 1  # Set to your mic index (check with sd.query_devices())
USE_SMALL_MODELS = True

# === Load Models ===
print("[INFO] Loading models and encoders...")
model = tf.keras.models.load_model('./v2_models/meld_multimodal_model.keras')
label_encoder = joblib.load('./v2_models/label_encoder.pkl')
scaler = joblib.load('./v2_models/audio_scaler.pkl')
print(f"[INFO] Multimodal model loaded. Expects {scaler.mean_.shape[0]} audio features.")

# === Load Whisper and BERT ===
print("[INFO] Loading Whisper model...")
whisper_model_name = "tiny.en" if USE_SMALL_MODELS else "base"
whisper = WhisperModel(whisper_model_name, device="cpu")
print("[INFO] Whisper model loaded.")

print("[INFO] Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")
print("[INFO] BERT model loaded.")

# === Load YAMNet ===
print("[INFO] Loading YAMNet model...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("[INFO] YAMNet model loaded.")

# === Feature Extraction ===
def extract_yamnet_features(y, sr):
    try:
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            y = resampler(torch.tensor(y).unsqueeze(0)).squeeze().numpy()
        elif isinstance(y, torch.Tensor):
            y = y.numpy()
        elif isinstance(y, np.ndarray):
            y = y
        else:
            return None

        scores, embeddings, spectrogram = yamnet_model(y)
        embedding = tf.reduce_mean(embeddings, axis=0)
        return embedding.numpy()
    except Exception as e:
        print(f"[ERROR] YAMNet feature extraction failed: {e}")
        return None

# === Text Processing ===
def transcribe_audio(file_path):
    try:
        segments, _ = whisper.transcribe(file_path)
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return ""

def embed_text(text):
    try:
        tokens = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=64)
        output = bert(**tokens)
        return output.last_hidden_state[:, 0, :].numpy().squeeze()
    except Exception as e:
        print(f"[ERROR] Text embedding failed: {e}")
        return None

# === Prediction ===
def predict_multimodal(audio, temp_path="temp.wav"):
    sf.write(temp_path, audio, SAMPLE_RATE)
    y, sr = torchaudio.load(temp_path)
    y = y.squeeze()

    features = extract_yamnet_features(y, sr)
    if features is None:
        return None

    text = transcribe_audio(temp_path)
    if not text.strip():
        return None

    audio_scaled = scaler.transform([features])
    text_embedded = embed_text(text)
    if text_embedded is None:
        return None

    prediction = model.predict([audio_scaled, text_embedded[np.newaxis, :]])
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction)

    return predicted_label[0], confidence, text

# === Live Prediction Loop ===
logger = VocalLogger(log_dir="logs")
logger.log_console("[INFO] Starting live multimodal prediction. Press Ctrl+C to stop.")

try:
    while True:
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, device=INPUT_DEVICE, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        if np.sqrt(np.mean(audio**2)) < 0.01:
            logger.log_console("Silence detected — skipping prediction.")
            continue  # No need to log silent at all

        result = predict_multimodal(audio)
        if result:
            label, conf, transcript = result
            if len(transcript.split()) < 3:
                logger.log_console("Not enough speech detected — skipping prediction.")
                continue
            logger.log_console(f"[PREDICTION] {label} (Confidence: {conf:.2f}) | Text: {transcript}")
            logger.log_prediction(label, conf)  

except KeyboardInterrupt:
    logger.log_console("[INFO] Live prediction stopped.")