import os
import time
import joblib
import numpy as np
import pandas as pd
import librosa
import parselmouth
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.combine import SMOTETomek
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ===============================
# SETTINGS
# ===============================
OUTPUT_DIR = './output'
MODEL_DIR = './models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Auto timestamp for outputs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_output_dir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
os.makedirs(run_output_dir, exist_ok=True)

# Data directories (relative to script)
data_dirs = {
    "crema-d": "data/crema-d",
    "RAVDESS": "data/RAVDESS",
    "EmoDB": "data/EmoDB",
    "TESS": "data/TESS"
}

# ===============================
# Feature Extraction Function
# ===============================
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # === Robust length check ===
        if len(y) < 512:
            return None  # Skip very short files

        # Audio features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft, axis=1)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))

        pitch = parselmouth.Sound(y, sampling_frequency=sr).to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_mean = np.nanmean(pitch_values[pitch_values > 0]) if np.any(pitch_values > 0) else 0

        features = np.concatenate([
            mfccs_mean,
            chroma_stft_mean,
            spectral_contrast_mean,
            tonnetz_mean,
            [zero_crossing_rate, rms, pitch_mean]
        ])

        return features

    except Exception as e:
        return None  # skip problematic files

# ===============================
# Dataset Parsing
# ===============================
def parse_dataset():
    all_features, all_labels = [], []
    summary = {}

    for dataset_name, dir_path in data_dirs.items():
        print(f"\n[INFO] Processing {dataset_name}...")
        valid_samples = 0
        features = []
        labels = []

        # Collect files
        file_paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    file_paths.append(os.path.join(root, file))

        for file_path in tqdm(file_paths, desc=f"{dataset_name} Feature Extraction"):
            feats = extract_features(file_path)
            if feats is not None:
                features.append(feats)
                labels.append(dataset_name)  # label = dataset name
                valid_samples += 1

        if features:
            all_features.extend(features)
            all_labels.extend(labels)

        summary[dataset_name] = (valid_samples, len(file_paths))

    # Log summary
    print("\n[SUMMARY] Dataset processing complete:")
    for name, (valid, total) in summary.items():
        print(f"- {name}: {valid}/{total} valid samples")

    return np.array(all_features), np.array(all_labels)

# ===============================
# Model Training
# ===============================
def train_model(X, y):
    print(f"\n[INFO] Total samples: {len(X)}")
    print(f"[INFO] Feature size per sample: {X.shape[1]}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance classes
    print("[INFO] Applying SMOTE + Tomek Links...")
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_scaled, y_encoded)
    print(f"[INFO] After resampling: {Counter(y_res)}")

    # Model pipeline
    pipeline = Pipeline([
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0))
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__max_depth': [6, 9]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)
    grid.fit(X_res, y_res)

    print(f"\n[INFO] Best parameters: {grid.best_params_}")

    # Evaluate
    y_pred = grid.predict(X_scaled)
    report = classification_report(y_encoded, y_pred, target_names=le.classes_)
    print("\n[Classification Report]\n")
    print(report)

    # Save report
    report_path = os.path.join(run_output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Save outputs
    joblib.dump(grid.best_estimator_, os.path.join(MODEL_DIR, 'emotion_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    print(f"\n[INFO] Training log saved to {run_output_dir}")
    print(f"[INFO] Models saved to {MODEL_DIR}")
    print(f"[✅] Training complete!")

# ===============================
# Main
# ===============================
def main():
    X, y = parse_dataset()

    if X.size == 0 or y.size == 0:
        print("[ERROR] No valid features extracted. Exiting.")
        return

    train_model(X, y)

if __name__ == "__main__":
    main()
