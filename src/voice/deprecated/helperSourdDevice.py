import sounddevice as sd

"""
Microphone Sample Rate Compatibility Checker

This script tests if a given input device (microphone) supports common sample rates.
Useful for verifying compatibility before running real-time audio models (e.g., Whisper, emotion detection).

Steps:
1. Set your mic's device index manually.
2. It will print device info.
3. It checks several common sample rates (8000-48000 Hz) and reports which ones work.

"""

device_index = 7  # your mic index

info = sd.query_devices(device_index, 'input')
print(f"Input device info: {info}")

for rate in [8000, 11025, 16000, 22050, 32000, 44100, 48000]:
    try:
        sd.check_input_settings(device=device_index, samplerate=rate)
        print(f"✅ Sample rate {rate} Hz is supported")
    except Exception as e:
        print(f"❌ Sample rate {rate} Hz not supported: {e}")
