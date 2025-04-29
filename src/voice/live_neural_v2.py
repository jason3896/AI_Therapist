import os
import time
import joblib
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from datetime import datetime
from faster_whisper import WhisperModel
from transformers import BertTokenizer, TFBertModel

# === Vocal Logger ===
class VocalLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"vocal_log_{timestamp}.csv")
        self.index = 0

        with open(self.log_file, 'w') as f:
            f.write("index,timestamp,emotion,confidence\n")

    def log_prediction(self, emotion, confidence):
        self.index += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"{self.index},{timestamp},{emotion},{confidence:.6f}\n")

    def log_console(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

# === Config ===
SAMPLE_RATE = 48000
DURATION = 5  # seconds
INPUT_DEVICE = 7
N_MELS = 40
USE_SMALL_MODELS = True

# === Load Models ===
print("[INFO] Loading models and encoders...")
model = tf.keras.models.load_model('./models/meld_multimodal_model.keras')
label_encoder = joblib.load('./models/label_encoder.pkl')
scaler = joblib.load('./models/audio_scaler.pkl')
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

# === Feature Extraction ===
def zero_crossing_rate(waveform):
    zc = (waveform[:, 1:] * waveform[:, :-1]) < 0
    return zc.float().mean(dim=1).item()

def extract_features_torch(y, sr):
    try:
        if len(y) < 250 or np.sqrt(np.mean(y**2)) < 0.01:
            return None

        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40}
        )(waveform)
        features = mfcc.mean(dim=2).squeeze()
        return features.numpy()
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None

# === Text ===
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
    y = y.squeeze().numpy()

    features = extract_features_torch(y, sr)
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
            logger.log_prediction("silent", 0.0)
            continue

        result = predict_multimodal(audio)
        if result:
            label, conf, transcript = result
            if len(transcript.split()) < 3:
                logger.log_console("Not enough speech detected — skipping prediction.")
                logger.log_prediction("silent", 0.0)
                continue
            logger.log_console(f"[PREDICTION] {label} (Confidence: {conf:.2f}) | Text: {transcript}")
            logger.log_prediction(label, conf)

except KeyboardInterrupt:
    logger.log_console("[INFO] Live prediction stopped.")
