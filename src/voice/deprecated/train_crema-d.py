import os
import numpy as np
import parselmouth
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR = "crema-d/"

# Map filename parts to emotions
emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

def extract_features(file_path):
    try:
        snd = parselmouth.Sound(file_path)

        # Convert to mono if stereo
        if snd.n_channels > 1:
            snd = snd.convert_to_mono()

        # Pitch
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        if len(pitch_values) == 0:
            raise ValueError("No pitch detected in audio.")
        pitch_mean = np.mean(pitch_values)

        # Intensity
        intensity = snd.to_intensity()
        intensity_values = intensity.values
        intensity_mean = np.mean(intensity_values) if len(intensity_values) > 0 else 0

        # Point process for jitter/shimmer
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        return [pitch_mean, intensity_mean, jitter, shimmer]

    except Exception as e:
        print(f"[WARNING] Skipped {file_path}: {e}")
        return [None, None, None, None]


features = []
labels = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".wav"):
        emotion_code = file.split('_')[2]
        label = emotion_map.get(emotion_code)

        if label:
            file_path = os.path.join(DATA_DIR, file)
            feature = extract_features(file_path)
            if None not in feature:
                features.append(feature)
                labels.append(label)

print(f"[INFO] Extracted features from {len(features)} samples.")

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Train classifier
clf = SVC(probability=True)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and label encoder
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/cremad_voice_svm.pkl")
joblib.dump(le, "models/cremad_label_encoder.pkl")

print("\n[INFO] CREMA-D voice emotion model trained and saved!")
