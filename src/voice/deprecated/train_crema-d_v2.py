import os
import numpy as np
import parselmouth
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

# ======== Config ========
DATA_DIR = "./crema-d/"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Map filename parts to emotions
emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# ======== Feature Extraction ========
def extract_features(file_path):
    try:
        snd = parselmouth.Sound(file_path)

        # Convert stereo to mono
        if snd.n_channels > 1:
            snd = snd.convert_to_mono()

        # Raw audio
        y = snd.values[0]
        sr = int(snd.sampling_frequency)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Pitch
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        if len(pitch_values) == 0:
            raise ValueError("No pitch detected.")
        pitch_mean = np.mean(pitch_values)

        # Intensity
        intensity = snd.to_intensity()
        intensity_values = intensity.values
        intensity_mean = np.mean(intensity_values) if len(intensity_values) > 0 else 0

        # Jitter & Shimmer from PointProcess
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # Combine all features
        features = [pitch_mean, intensity_mean, jitter, shimmer]
        features.extend(mfccs_mean.tolist())

        return features

    except Exception as e:
        print(f"[WARNING] Skipped {file_path}: {e}")
        return None  # Cleaner handling

# ======== Dataset Preparation ========
features, labels = [], []
processed, success = 0, 0

for file in os.listdir(DATA_DIR):
    if file.endswith(".wav"):
        emotion_code = file.split('_')[2]
        label = emotion_map.get(emotion_code)
        if label:
            file_path = os.path.join(DATA_DIR, file)
            feature = extract_features(file_path)
            processed += 1
            if feature is not None:
                features.append(feature)
                labels.append(label)
                success += 1

print(f"\n[INFO] Processed {processed} files.")
print(f"[INFO] Successfully extracted features from {success} files.")
print(f"[INFO] Success rate: {success / processed * 100:.2f}%\n")

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Check class distribution
print(f"[INFO] Training set class distribution: {Counter(y_train)}")

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"[INFO] After SMOTE class distribution: {Counter(y_train_resampled)}")

# ======== Model Training ========
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# ======== Evaluation ========
y_pred = clf.predict(X_test)
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ======== Save Model ========
joblib.dump(clf, os.path.join(MODEL_DIR, "cremad_voice_rf.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "cremad_label_encoder.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "cremad_scaler.pkl"))

print(f"\n[INFO] Model and encoder saved to '{MODEL_DIR}/'")
