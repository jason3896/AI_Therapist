import sounddevice as sd

device_index = 7  # your mic index

info = sd.query_devices(device_index, 'input')
print(f"Input device info: {info}")

for rate in [8000, 11025, 16000, 22050, 32000, 44100, 48000]:
    try:
        sd.check_input_settings(device=device_index, samplerate=rate)
        print(f"✅ Sample rate {rate} Hz is supported")
    except Exception as e:
        print(f"❌ Sample rate {rate} Hz not supported: {e}")
