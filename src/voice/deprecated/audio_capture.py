import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

"""
Real-Time Audio Recorder

This script records a short audio snippet from the default microphone using 
the sounddevice library and saves it as a 16-bit PCM WAV file called 'realtime_audio.wav'.

- Default duration: 1 second
- Sample rate: 16,000 Hz (suitable for speech/emotion recognition)
- Output format: mono, 16-bit WAV
"""

def record_audio(duration=1, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    audio = np.squeeze(audio)
    wav.write("realtime_audio.wav", fs, (audio * 32767).astype(np.int16))
    print("Saved audio chunk.")
