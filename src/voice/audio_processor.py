import queue
import threading
import sounddevice as sd
import numpy as np
import parselmouth
from faster_whisper import WhisperModel
import soundfile as sf
import tempfile
from datetime import datetime


# Configuration
DEVICE = "cpu"  # or "cuda" if you have GPU
MODEL_SIZE = "base"  # "small", "medium", etc.
SAMPLERATE = 16000
CHUNK_DURATION = 3  # seconds of audio per chunk

# Initialize Whisper model
print("[INFO] Loading Whisper model...")
model = WhisperModel(MODEL_SIZE, device=DEVICE)

# Audio stream setup
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"[WARNING] Audio status: {status}")
    q.put(indata.copy())

def extract_voice_features(audio_data, sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, audio_data, sample_rate)
        snd = parselmouth.Sound(temp_audio_file.name)

    # Extract features
    pitch = snd.to_pitch().mean()
    intensity = snd.to_intensity().mean()
    jitter = snd.to_jitter().local_jitter()
    shimmer = snd.to_shimmer().local_shimmer()

    return pitch, intensity, jitter, shimmer

def classify_voice_emotion(pitch, intensity, jitter, shimmer):
    if pitch is None or np.isnan(pitch):
        pitch = 0
    if jitter is None or np.isnan(jitter):
        jitter = 0
    if intensity is None or np.isnan(intensity):
        intensity = 0

    # Basic rule-based confidence
    if pitch > 220 and jitter > 0.01:
        return {"emotion": "Anxious", "confidence": 0.8}
    elif intensity > 60 and jitter < 0.005:
        return {"emotion": "Angry", "confidence": 0.7}
    elif pitch < 150 and intensity < 50:
        return {"emotion": "Sad", "confidence": 0.75}
    else:
        return {"emotion": "Neutral", "confidence": 0.6}


def audio_stream():
    with sd.InputStream(samplerate=SAMPLERATE, channels=1, dtype='float32',
                        callback=audio_callback):
        print("[INFO] Microphone stream started. Speak into the mic!")
        while True:
            sd.sleep(1000)

def transcribe_stream():
    buffer = np.zeros(0, dtype=np.float32)
    chunk_size = int(SAMPLERATE * CHUNK_DURATION)

    while True:
        data = q.get()
        buffer = np.concatenate((buffer, data.flatten()))

        if len(buffer) >= chunk_size:
            audio_chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]

            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Voice feature extraction
            pitch, intensity, jitter, shimmer = extract_voice_features(audio_chunk, SAMPLERATE)
            voice_emotion_result = classify_voice_emotion(pitch, intensity, jitter, shimmer)

            # Transcribe audio chunk
            segments, info = model.transcribe(audio_chunk, language="en", beam_size=5, vad_filter=True)
            text = " ".join(segment.text for segment in segments).strip()

            if text:
                # ✅ Print updated with timestamp and structured emotion output
                print(f"\n[Timestamp]: {timestamp}")
                print(f"[Transcript]: {text}")
                print(f"[Voice Emotion]: {voice_emotion_result['emotion']} (Confidence: {voice_emotion_result['confidence']})")

                # ✅ Log to file
                with open("conversation_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"[Timestamp]: {timestamp}\n")
                    f.write(f"[Transcript]: {text}\n")
                    f.write(f"[Voice Emotion]: {voice_emotion_result['emotion']} (Confidence: {voice_emotion_result['confidence']})\n\n")

                # ✅ Optional: Structured result object for future fusion module
                result = {
                    "timestamp": timestamp,
                    "modality": "voice",
                    "emotion": voice_emotion_result['emotion'],
                    "confidence": voice_emotion_result['confidence'],
                    "transcript": text
                }

                print(f"[Voice Output]: {result}")


if __name__ == "__main__":
    # Start threads
    audio_thread = threading.Thread(target=audio_stream, daemon=True)
    transcribe_thread = threading.Thread(target=transcribe_stream, daemon=True)

    audio_thread.start()
    transcribe_thread.start()

    print("[INFO] System is live! Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n[INFO] Stopping system. Goodbye!")
