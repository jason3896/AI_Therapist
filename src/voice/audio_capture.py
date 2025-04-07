import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

def record_audio(duration=1, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    audio = np.squeeze(audio)
    wav.write("realtime_audio.wav", fs, (audio * 32767).astype(np.int16))
    print("Saved audio chunk.")
