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

# whisper is used to get text from speech
whisper = WhisperModel("tiny.en" if USE_SMALL_MODELS else "base", device="cpu")

""" 
Bert is a pretrained model developed by Google that is designed to understand context and meaning of
words in a sentence by looking at both the left and right sides of a word simultaneously
For the purpose of this model, BERT turns text into rich, context-aware numerical features that the neural 
network can use to help classify emotions, especially when combined with audio features from YAMNet.
The tokenizer is used for the bert-model to to its job.
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
emotion_model = tf.keras.models.load_model("./src/voice/v2_models/meld_multimodal_model.keras")
label_encoder = joblib.load("./src/voice/v2_models/label_encoder.pkl")
scaler = joblib.load("./src/voice/v2_models/audio_scaler.pkl")

# === HELPERS ===

""" Grab the most recent log file from the facial emotion detector"""
def get_latest_log_file(log_dir):
    files = glob.glob(f"{log_dir}/emotion_log_*.csv")
    return Path(max(files, key=os.path.getctime)) if files else None

""" Convert a timestamp to a pandas datetime object"""
def parse_timestamp(ts):
    return pd.to_datetime(ts)
"""
Loads a facial emotion log CSV (e.g., with timestamps, emotion labels, and confidence scores).
Correctly formats the column names
Converts the given row to pandas datetime object
Return all rows that fall within the time interval
If there is no timestamp in the given time slot, return an empty dataframe
"""
def extract_range_from_csv(csv_path, start_time, end_time):
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    return pd.DataFrame()

"""  Used to extract the two most dominant emotions from the facial detector """
def extract_primary_secondary_emotions(df):
    
    # Make sure the dataframe is not empty and has the correct columns
    if df.empty or 'emotion' not in df or 'confidence' not in df:
        return ("none", 0.0), ("none", 0.0)
    
    # Group the data by the emotion label and get avereage confidence interval
    grouped = df.groupby('emotion')['confidence'].mean()
    
    # Sort the values
    sorted_emotions = grouped.sort_values(ascending=False).head(2)
    
    # create a list of tuples
    top = list(sorted_emotions.items()) + [("none", 0.0)] * (2 - len(sorted_emotions))
    
    # Grab the two most dominant
    return top[0], top[1]

""" Used to extract the audio features to plug into the model """
def extract_yamnet_features(y, sr):
    
    # If the sample rate does not match 16000 Hz, converet to appropriate sample rate
    if sr != SAMPLE_RATE:
        y = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(torch.tensor(y).unsqueeze(0)).squeeze().numpy()
    
    # Grab the data from yammnet
    scores, embeddings, _ = yamnet_model(y)
    
    # Temporal pooling to average into 1024 dimensoinal vector
    mean_embed = tf.reduce_mean(embeddings, axis=0).numpy()
    
    # Make sure it matches 1024 dimensional vector shape
    if mean_embed.shape[0] != EXPECTED_AUDIO_FEATURES:
        print(f"[ERROR] Feature dimension mismatch: got {mean_embed.shape[0]}") 
        return None
    return mean_embed

""" Creates data from Bert """
def embed_text(text):
    
    # Tokenize the text recieved
    tokens = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=64)
    
    # Run it through the bery model and grab the CLS embedding (summary of the whole sentence) 
    return bert(**tokens).last_hidden_state[:, 0, :].numpy().squeeze()

""" Grab all the files associated with an audio chunk from the entire audio file """
def get_words_for_chunk(segments, chunk_start, chunk_end):
    words = []
    for seg in segments:
        center = (seg.start + seg.end) / 2
        if chunk_start <= center < chunk_end:
            words.append(seg.text)
    return " ".join(words)

""" 
Break the audio into 5-second chunks, extract both audio and text features for each chunk, 
and use the trained model to predict the speaker’s emotion in that chunk.
"""
def run_multimodal_analysis(audio_path, segments):

    # Load the audio file 
    y, sr = torchaudio.load(audio_path)
    y = y.squeeze().numpy()
    # determine the length of the file
    duration = len(y) / SAMPLE_RATE

    emotions = []
    # loop over 5 second intervals
    for i in range(int(np.ceil(duration / 5))):
        
        # Extract a chunk as a slice of the entire audio file
        chunk = y[int(i * 5 * SAMPLE_RATE):int(min((i + 1) * 5 * SAMPLE_RATE, len(y)))]

        # Make sure to skip over files that are to quiet
        rms = np.sqrt(np.mean(chunk**2))
        if rms < 0.001:
            print(f"[SKIPPED] Chunk {i} is too silent (RMS={rms:.4f})")
            continue
        
        # Write the wav file to directory
        sf.write("chunk_temp.wav", chunk, SAMPLE_RATE)
        
        # Extract the yamnet features from the audio file
        features = extract_yamnet_features(chunk, SAMPLE_RATE)
        if features is None:
            continue 
        audio_scaled = scaler.transform([features])
        
        # Get the words assocaited with a current chunk
        text_chunk = get_words_for_chunk(segments, i * 5, (i + 1) * 5)
        print(f"[CHUNK {i}] Words in chunk: \"{text_chunk}\"")

        # Run the extracted text through bert
        text_embedded = embed_text(text_chunk)
        
        # Make sure it matches the correct size
        if text_embedded.shape[0] != 768:
            print(f"[ERROR] Text embedding shape mismatch: {text_embedded.shape}")
            continue
        
        # Run a prediction with the pretrained neural network
        prediction = emotion_model.predict([audio_scaled, text_embedded[np.newaxis, :]])
        label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)

        print(f"[DEBUG] Prediction: {label} ({confidence:.2f})")
        emotions.append((label, confidence))

    return emotions

""" For ElevenLabs TTS """
def speak(text):
    play(client.generate(text=text, voice="Rosie", model="eleven_monolingual_v1"))

# === Audio Recording Callbacks ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[WARN] Audio status: {status}")
    
    # Append the audio chunk to the global audio_buffer
    audio_buffer.append(indata.copy())

""" Runs the therpaist session to detect emotions"""
def run_therapist_session(audio_array, start_time, end_time, facial_log):
    
    # Process the recorded audio
    duration = (end_time - start_time).total_seconds()
    audio = np.concatenate(audio_array, axis=0)
    audio = audio[:int(SAMPLE_RATE * duration)]
    scipy.io.wavfile.write("temp_audio.wav", SAMPLE_RATE, audio) # Save the recored audio as a temporary wav file

    # Extract the text
    result = whisper.transcribe("temp_audio.wav")
    segments = list(result[0])  # Get actual segments
    
    # Combined all segments into one string
    transcript = " ".join([s.text for s in segments]).strip()
    
    # Make sure words were actually said
    if not transcript:
        return transcript, "[ERROR] No speech detected."

    # Run multimodal analysis on the entire recording for audio and text
    emotions = run_multimodal_analysis("temp_audio.wav", segments)
    
    # Collect emotion labels and confidence
    emotion_groups = defaultdict(list)
    for label, conf in emotions:
        emotion_groups[label].append(conf)

    # Average the confidences
    avg_emotions = [(label, np.mean(confs)) for label, confs in emotion_groups.items()]

    # Sort by average confidence descending
    avg_emotions.sort(key=lambda x: x[1], reverse=True)

    # Get top 2 emotions
    voice_primary, voice_secondary = (avg_emotions + [("none", 0.0)] * 2)[:2]

    face_df = extract_range_from_csv(facial_log, start_time, end_time)
    face_primary, face_secondary = extract_primary_secondary_emotions(face_df)

    # Emotional context for the query
    emotion_context = (
        "You are an empathetic therapist. The system has analyzed the user's emotional state from their voice and facial expressions, "
        "but you should not mention any scores, numbers, or analysis methods in your reply. "
        "Instead, respond compassionately as if you sensed these emotions naturally through the conversation.\n\n"
        f"Facial emotions suggest the user is primarily feeling {face_primary[0]} and secondarily {face_secondary[0]}. "
        f"Voice emotions suggest the user is primarily feeling {voice_primary[0]} and secondarily {voice_secondary[0]}."
    )

    # Grab the relevenat context from chromadb for this query
    retrieved_docs = "\n".join(chroma_collection.query(query_texts=[transcript], n_results=2)["documents"][0])

    # Construct the prompt
    prompt = (
        f"{emotion_context}\n\nRelevant past thoughts:\n{retrieved_docs}\n\n"
        f"User (voice transcription): {transcript}\nTherapist:"
    )
    
    print("\n" + "="*20 + " PROMPT " + "="*20)
    print(prompt)
    print("="*49 + "\n")

    # Query Llama 3.3
    response = model.generate_content(prompt)
    
    # Add response to the database
    chroma_collection.add(
        documents=[transcript, response.text],
        ids=[f"user-{uuid.uuid4()}", f"therapist-{uuid.uuid4()}"],
        embeddings=hf_embeddings([transcript, response.text])
    )

    # text-to-speech
    speak(response.text)
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
    
