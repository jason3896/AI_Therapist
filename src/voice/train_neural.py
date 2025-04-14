import os
import warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
from datetime import datetime
import joblib
import shap

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths
DATA_DIR = './data'
MODEL_DIR = './models'
OUTPUT_DIR = './output'
LOG_DIR = './logs'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Parameters
SAMPLE_RATE = 48000
N_MFCC = 13
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 0.001

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_log = os.path.join(OUTPUT_DIR, f'training_log_{timestamp}.csv')

print(f"[{timestamp}] [INFO] Training started...")

# Helper functions
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # If too short, skip
        if len(y) < 2048:
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(zcr),
            np.mean(rms),
            np.mean(spec_contrast, axis=1),
            np.mean(tonnetz, axis=1),
        ])

        return features
    except Exception:
        return None

def parse_dataset():
    features, labels = [], []

    print(f"[INFO] Processing dataset...")
    datasets = os.listdir(DATA_DIR)
    for dataset in datasets:
        dataset_path = os.path.join(DATA_DIR, dataset)
        if not os.path.isdir(dataset_path):
            continue

        print(f"[INFO] Processing {dataset}...")
        for root, _, files in os.walk(dataset_path):
            for file in tqdm(files, desc=f"{dataset} Feature Extraction"):
                if not file.endswith('.wav'):
                    continue

                file_path = os.path.join(root, file)
                feats = extract_features(file_path)
                if feats is not None:
                    features.append(feats)

                    # Dataset-specific label extraction
                    if dataset == 'crema-d':
                        parts = file.split('_')
                        labels.append(parts[2])  # Example: 1031_IOM_HAP_XX.wav -> HAP
                    elif dataset == 'ravdess':
                        parts = file.split('-')
                        labels.append(parts[2])  # Example: 03-01-05-01-01-01-01.wav -> 05
                    elif dataset == 'TESS':
                        labels.append(os.path.basename(root))
                    elif dataset == 'emodb':
                        labels.append(file[5])  # Example: 03a01Fa.wav -> 'F'

    if not features:
        print("[ERROR] No valid features extracted. Exiting.")
        exit()

    X = np.array(features)
    y = np.array(labels)

    print(f"[SUMMARY] Dataset processing complete.")
    print(f"- Total samples: {len(y)}")
    print(f"- Total features per sample: {X.shape[1]}")

    return X, y

def balance_dataset(X, y):
    print("[INFO] Balancing dataset with SMOTE + Tomek links...")
    smote_tomek = SMOTETomek(random_state=42, n_jobs=-1)
    X_res, y_res = smote_tomek.fit_resample(X, y)
    return X_res, y_res

def build_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_feature_importance(model, X_train, output_dir):
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    shap.save_html(os.path.join(output_dir, 'shap_summary.html'), shap_values)

def main():
    X, y = parse_dataset()

    # Clean bad values
    print("[INFO] Cleaning dataset...")
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance dataset
    X_bal, y_bal = balance_dataset(X_scaled, y_encoded)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

    # Build model
    model = build_model(input_dim=X_train.shape[1], num_classes=len(np.unique(y_bal)))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'emotion_model.pkl'), save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.CSVLogger(output_log),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, timestamp))
    ]

    # Training
    history = model.fit(
        X_train, tf.keras.utils.to_categorical(y_train),
        validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save encoders and scalers
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    print("\n[Classification Report]\n")
    print(classification_report(y_test, y_pred_labels, target_names=le.classes_))

    # Optional SHAP importance
    print("[INFO] Generating feature importance report (SHAP)...")
    plot_feature_importance(model, X_train, OUTPUT_DIR)

    print(f"[INFO] Training complete! Logs saved to: {output_log}")

if __name__ == "__main__":
    main()
