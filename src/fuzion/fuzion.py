import os
import time
import uuid
import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import sounddevice as sd
import scipy.io.wavfile
import soundfile as sf
import tensorflow as tf
import torchaudio
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel
from faster_whisper import WhisperModel
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel
from elevenlabs import ElevenLabs, play
import chromadb
from chromadb.utils import embedding_functions
import customtkinter as ctk
import joblib
import threading
import queue
from collections import defaultdict

# === DEVICE SELECTION ===
def select_input_device():
    devices = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
    return 1 if len(devices) > 1 else 0

# === INITIALIZE SERVICES ===
init(project="ai-therapist-456121", location="us-central1")
model = GenerativeModel("publishers/meta/models/llama-3.3-70b-instruct-maas")
client = ElevenLabs(api_key="sk_9f9ea317311bc0c0f0da8e16a6127ad17fb34fe8d21b3bd6")
hf_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(name="user_chat_memory")

# === CONFIG ===
FACIAL_LOGS_DIR = "src/facial/data/emotion_logs"
SAMPLE_RATE = 16000
USE_SMALL_MODELS = True
RECORDING_DEVICE = 1
EXPECTED_AUDIO_FEATURES = 1024

# === Globals for recording ===
audio_buffer = []
recording_stream = None

# === MODELS ===
whisper = WhisperModel("tiny.en" if USE_SMALL_MODELS else "base", device="cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
emotion_model = tf.keras.models.load_model("./src/voice/v2_models/meld_multimodal_model.keras")
label_encoder = joblib.load("./src/voice/v2_models/label_encoder.pkl")
scaler = joblib.load("./src/voice/v2_models/audio_scaler.pkl")

# === HELPERS ===
def get_latest_log_file(log_dir):
    files = glob.glob(f"{log_dir}/emotion_log_*.csv")
    return Path(max(files, key=os.path.getctime)) if files else None

def parse_timestamp(ts):
    return pd.to_datetime(ts)

def extract_range_from_csv(csv_path, start_time, end_time):
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    return pd.DataFrame()

def extract_primary_secondary_emotions(df):
    if df.empty or 'emotion' not in df or 'confidence' not in df:
        return ("none", 0.0), ("none", 0.0)
    grouped = df.groupby('emotion')['confidence'].mean()
    sorted_emotions = grouped.sort_values(ascending=False).head(2)
    top = list(sorted_emotions.items()) + [("none", 0.0)] * (2 - len(sorted_emotions))
    return top[0], top[1]

def extract_yamnet_features(y, sr):
    if sr != SAMPLE_RATE:
        y = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(torch.tensor(y).unsqueeze(0)).squeeze().numpy()
    scores, embeddings, _ = yamnet_model(y)
    mean_embed = tf.reduce_mean(embeddings, axis=0).numpy()
    if mean_embed.shape[0] != EXPECTED_AUDIO_FEATURES:
        print(f"[ERROR] Feature dimension mismatch: got {mean_embed.shape[0]}") 
        return None
    return mean_embed

def embed_text(text):
    tokens = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=64)
    return bert(**tokens).last_hidden_state[:, 0, :].numpy().squeeze()

def get_words_for_chunk(segments, chunk_start, chunk_end):
    words = []
    for seg in segments:
        center = (seg.start + seg.end) / 2
        if chunk_start <= center < chunk_end:
            words.append(seg.text)
    return " ".join(words)


def run_multimodal_analysis(audio_path, segments):
    y, sr = torchaudio.load(audio_path)
    y = y.squeeze().numpy()
    duration = len(y) / SAMPLE_RATE

    emotions = []
    for i in range(int(np.ceil(duration / 5))):
        chunk = y[int(i * 5 * SAMPLE_RATE):int(min((i + 1) * 5 * SAMPLE_RATE, len(y)))]

        rms = np.sqrt(np.mean(chunk**2))
        if rms < 0.001:
            print(f"[SKIPPED] Chunk {i} is too silent (RMS={rms:.4f})")
            continue

        sf.write("chunk_temp.wav", chunk, SAMPLE_RATE)
        features = extract_yamnet_features(chunk, SAMPLE_RATE)
        if features is None:
            continue 

        audio_scaled = scaler.transform([features])
        text_chunk = get_words_for_chunk(segments, i * 5, (i + 1) * 5)
        print(f"[CHUNK {i}] Words in chunk: \"{text_chunk}\"")

        text_embedded = embed_text(text_chunk)
        if text_embedded.shape[0] != 768:
            print(f"[ERROR] Text embedding shape mismatch: {text_embedded.shape}")
            continue

        prediction = emotion_model.predict([audio_scaled, text_embedded[np.newaxis, :]])
        label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)

        print(f"[DEBUG] Prediction: {label} ({confidence:.2f})")
        emotions.append((label, confidence))

    return emotions

def speak(text):
    play(client.generate(text=text, voice="Rosie", model="eleven_monolingual_v1"))

# === Audio Recording Callbacks ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[WARN] Audio status: {status}")
    audio_buffer.append(indata.copy())

# === Updated Therapist Session ===
def run_therapist_session(audio_array, start_time, end_time, facial_log):
    duration = (end_time - start_time).total_seconds()
    audio = np.concatenate(audio_array, axis=0)
    audio = audio[:int(SAMPLE_RATE * duration)]
    scipy.io.wavfile.write("temp_audio.wav", SAMPLE_RATE, audio)

    result = whisper.transcribe("temp_audio.wav")
    segments = list(result[0])  # Get actual segments Figure out why this works later
        
    transcript = " ".join([s.text for s in segments]).strip()
    if not transcript:
        return transcript, "[ERROR] No speech detected."

    emotions = run_multimodal_analysis("temp_audio.wav", segments)
    emotion_groups = defaultdict(list)
    for label, conf in emotions:
        emotion_groups[label].append(conf)

    # Step 2: Average the confidences
    avg_emotions = [(label, np.mean(confs)) for label, confs in emotion_groups.items()]

    # Step 3: Sort by average confidence descending
    avg_emotions.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Get top 2 (pad if needed)
    voice_primary, voice_secondary = (avg_emotions + [("none", 0.0)] * 2)[:2]

    face_df = extract_range_from_csv(facial_log, start_time, end_time)
    face_primary, face_secondary = extract_primary_secondary_emotions(face_df)

    emotion_context = (
        "You are an empathetic therapist. The system has analyzed the user's emotional state from their voice and facial expressions, "
        "but you should not mention any scores, numbers, or analysis methods in your reply. "
        "Instead, respond compassionately as if you sensed these emotions naturally through the conversation.\n\n"
        f"Facial emotions suggest the user is primarily feeling {face_primary[0]} and secondarily {face_secondary[0]}. "
        f"Voice emotions suggest the user is primarily feeling {voice_primary[0]} and secondarily {voice_secondary[0]}."
    )

    retrieved_docs = "\n".join(chroma_collection.query(query_texts=[transcript], n_results=2)["documents"][0])

    prompt = (
        f"{emotion_context}\n\nRelevant past thoughts:\n{retrieved_docs}\n\n"
        f"User (voice transcription): {transcript}\nTherapist:"
    )
    
    print("\n" + "="*20 + " PROMPT " + "="*20)
    print(prompt)
    print("="*49 + "\n")

    response = model.generate_content(prompt)
    chroma_collection.add(
        documents=[transcript, response.text],
        ids=[f"user-{uuid.uuid4()}", f"therapist-{uuid.uuid4()}"],
        embeddings=hf_embeddings([transcript, response.text])
    )

    # speak(response.text)
    return transcript, response.text

# === GUI ===
class TherapistApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Therapist Interface")
        self.geometry("900x700")
        self.start_time = None
        self.end_time = None
        self.audio_thread = None

        self.chat_display = ctk.CTkTextbox(self, wrap="word")
        self.chat_display.pack(padx=20, pady=(10, 5), fill="both", expand=True)
        self.chat_display.insert("end", "Therapist: Hello. I'm here when you're ready to talk.\n\n")
        self.chat_display.configure(state="disabled")

        self.prompt_preview = ctk.CTkTextbox(self, height=80)
        self.prompt_preview.pack(fill="x", padx=20, pady=(5, 5))

        self.controls = ctk.CTkFrame(self)
        self.controls.pack(pady=10)
        self.mic_button = ctk.CTkButton(self.controls, text="🎤", width=50, command=self.toggle_recording)
        self.mic_button.grid(row=0, column=0, padx=10)

    def toggle_recording(self):
        global audio_buffer, recording_stream
        if self.start_time is None:
            audio_buffer = []
            self.start_time = datetime.now()
            recording_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=audio_callback,
                device=RECORDING_DEVICE
            )
            recording_stream.start()
            self.mic_button.configure(text="⏹️", fg_color="red")
        else:
            print("\n" + "=" * 40 + " STARTING SECTION " + "=" * 40 + "\n")
            self.end_time = datetime.now()
            recording_stream.stop()
            recording_stream.close()
            self.mic_button.configure(text="🎤", fg_color="green")

            def threaded_call():
                facial_log = str(get_latest_log_file(FACIAL_LOGS_DIR))
                transcript, response = run_therapist_session(audio_buffer, self.start_time, self.end_time, facial_log)
                self.after(0, lambda: self.display_response(transcript, response))
                self.start_time = None

            threading.Thread(target=threaded_call, daemon=True).start()

    def display_response(self, transcript, response):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "You:", "you_tag")
        self.chat_display.insert("end", f" {transcript}\n\n")
        self.chat_display.insert("end", "Therapist:", "therapist_tag")
        self.chat_display.insert("end", f" {response}\n\n")
        self.chat_display.tag_config("you_tag", foreground="green")
        self.chat_display.tag_config("therapist_tag", foreground="#6fa8dc")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

if __name__ == "__main__":
    app = TherapistApp()
    app.mainloop()