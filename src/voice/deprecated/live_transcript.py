from faster_whisper import WhisperModel

"""
Audio Transcriber with Faster-Whisper

This script loads a small Whisper model and defines a helper function
`transcribe_audio(file_path)` to return the full text transcription
from an audio file.

- Uses Faster-Whisper for fast CPU inference
- Supports most audio formats (e.g., .wav, .mp3)
- Returns a single string with the joined transcript
"""

model = WhisperModel("base", device="cpu")  # Use "medium" or "small" for speed

def transcribe_audio(file_path):
    segments, info = model.transcribe(file_path)
    text = " ".join(segment.text for segment in segments)
    return text
