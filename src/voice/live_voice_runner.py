import queue
import threading
import sounddevice as sd
import numpy as np
import joblib
import datetime
import os
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for colored output
colorama_init()

# Configuration
SAMPLERATE = 16000
CHUNK_DURATION = 3  # seconds
CHUNK_SIZE = int(SAMPLERATE * CHUNK_DURATION)

# Load your trained models
timestamp = "20250406_205834"  # 📝 Update this to your model timestamp
model = joblib.load(f'models/xgb_model_{timestamp}.joblib')
scaler = joblib.load(f'models/scaler_{timestamp}.joblib')
label_encoder = joblib.load(f'models/label_encoder_{timestamp}.joblib')

# Prepare logging
log_file = f'output_{timestamp}/live_emotion_log.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

def log_message(message):
    timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp_now}] {message}\n")

def extract_features_from_audio(data, samplerate):
    try:
        import parselmouth
        snd = parselmouth.Sound(data, sampling_frequency=samplerate)

        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        mean_pitch = np.mean(pitch_values) if pitch_values.size else 0

        intensity = snd.to_intensity()
        from parselmouth.praat import call
        mean_intensity = call(intensity, "Get mean", 0, 0, "energy")

        duration = snd.get_total_duration()

        return np.array([[mean_pitch, mean_intensity, duration]])
    except Exception as e:
        print(Fore.RED + f"[ERROR] Feature extraction failed: {e}" + Style.RESET_ALL)
        return None

def predict_emotion(features):
    if features is None:
        return None

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    emotion_label = label_encoder.inverse_transform(prediction)[0]
    return emotion_label

# Queue for audio chunks
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(Fore.YELLOW + f"[WARNING] Audio status: {status}" + Style.RESET_ALL)
    q.put(indata.copy())

def audio_stream():
    with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='float32', callback=audio_callback):
        print(Fore.CYAN + "[INFO] Microphone stream started. Speak into the mic!" + Style.RESET_ALL)
        while True:
            sd.sleep(1000)

def process_audio_stream():
    buffer = np.zeros(0, dtype=np.float32)

    while True:
        data = q.get()
        buffer = np.concatenate((buffer, data.flatten()))

        if len(buffer) >= CHUNK_SIZE:
            audio_chunk = buffer[:CHUNK_SIZE]
            buffer = buffer[CHUNK_SIZE:]

            features = extract_features_from_audio(audio_chunk, SAMPLERATE)
            emotion = predict_emotion(features)

            timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if emotion:
                output = f"[{timestamp_now}] 🎙️ Detected emotion: {Fore.GREEN}{emotion.upper()}{Style.RESET_ALL}"
                print(output)
                log_message(output)
            else:
                print(Fore.MAGENTA + f"[{timestamp_now}] No emotion detected in chunk." + Style.RESET_ALL)

if __name__ == "__main__":
    print(Fore.CYAN + "\n[INFO] Starting real-time voice emotion recognition..." + Style.RESET_ALL)

    try:
        # Start audio and processing threads
        audio_thread = threading.Thread(target=audio_stream, daemon=True)
        process_thread = threading.Thread(target=process_audio_stream, daemon=True)

        audio_thread.start()
        process_thread.start()

        while True:
            pass  # Keep main thread alive

    except KeyboardInterrupt:
        print(Fore.CYAN + "\n[INFO] Stopping voice emotion recognition. Goodbye!" + Style.RESET_ALL)
