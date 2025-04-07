from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu")  # Use "medium" or "small" for speed

def transcribe_audio(file_path):
    segments, info = model.transcribe(file_path)
    text = " ".join(segment.text for segment in segments)
    return text
