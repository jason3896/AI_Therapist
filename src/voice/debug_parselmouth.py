import parselmouth
import numpy as np

file_path = "./crema-d/1091_WSI_DIS_XX.wav"  # <-- Update path if needed!

try:
    snd = parselmouth.Sound(file_path)

    # Pitch
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]

    if len(pitch_values) == 0:
        print("❌ No pitch detected.")
    else:
        print(f"✅ Mean pitch: {np.mean(pitch_values):.2f} Hz")

    # Intensity
    intensity = snd.to_intensity()
    intensity_values = intensity.values
    if len(intensity_values) == 0:
        print("❌ No intensity detected.")
    else:
        print(f"✅ Mean intensity: {np.mean(intensity_values):.2f} dB")

    # Voice object (required for jitter/shimmer)
    voice = parselmouth.praat.call(snd, "To Voice (pitch)", 75, 500, "yes")
    print("✅ Voice object created.")

    jitter = parselmouth.praat.call(voice, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call(voice, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    print(f"✅ Jitter: {jitter}")
    print(f"✅ Shimmer: {shimmer}")

except Exception as e:
    print(f"❌ Error: {e}")
