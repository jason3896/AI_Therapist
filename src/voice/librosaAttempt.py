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
from joblib import Parallel, delayed

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

MIN_LEN = 1024 

# Parameters
SAMPLE_RATE = 48000
N_MFCC = 13
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 25
LEARNING_RATE = 0.001

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_log = os.path.join(OUTPUT_DIR, f'training_log_{timestamp}.csv')

print(f"[{timestamp}] [INFO] Training started...")

# ---- Dataset paths and emotion mappings ----
data_dirs = {
    "crema-d": "data/crema-d",
    "ravdess": "data/RAVDESS",
    "TESS": "data/TESS",
}

emotion_map = {
    'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 'ANG': 'angry', 'FEA': 'fearful', 'DIS': 'disgust',
    
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',
    
    'neutral': 'neutral', 'happy': 'happy', 'sad': 'sad', 'angry': 'angry', 'fear': 'fearful', 'disgust': 'disgust', 'ps': 'surprised',
}

def is_silent(y, threshold=0.01):
    energy = np.sqrt(np.mean(y**2))
    return energy < threshold

# Feature extraction function
def extract_features(file_path):
    try:
        if os.path.getsize(file_path) < 4000:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Drop short or silent clips
            if is_silent(y) or len(y) < 1024:
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
            np.mean(tonnetz, axis=1)
        ])
        return features

    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None


# Dataset parsing

from joblib import Parallel, delayed

def parse_dataset():
    cache_path = os.path.join(OUTPUT_DIR, 'features_cache.npz')

    if os.path.exists(cache_path):
        print("[INFO] Loading cached features from disk...")
        cache = np.load(cache_path, allow_pickle=True)
        return cache['X'], cache['y']

    entries = []

    for dataset, dataset_path in data_dirs.items():
        print(f"[INFO] Scanning files in {dataset}...")
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if not file.endswith('.wav'):
                    continue
                file_path = os.path.join(root, file)
                entries.append((dataset, root, file, file_path))

    def process_entry(dataset, root, file, file_path):
        feats = extract_features(file_path)
        if feats is None:
            return None
        
        label = None  
        
        if dataset == 'crema-d':
            label = emotion_map.get(file.split('_')[2])
        elif dataset == 'ravdess':
            label = emotion_map.get(file.split('-')[2])
        elif dataset == 'TESS':
            filename = os.path.splitext(file)[0].lower()
            for emotion_key in emotion_map:
                if filename.endswith(emotion_key):
                    label = emotion_map[emotion_key]
                    break
        else:
            label = None

        if label:
            return (feats, label)
        else:
            print(f"[WARN] Unknown label code in file: {file}")
            return None

    print("[INFO] Extracting features in parallel...")
    results = Parallel(n_jobs=-1)(delayed(process_entry)(*entry) for entry in tqdm(entries))

    results = [r for r in results if r is not None]
    features, labels = zip(*results)

    X = np.array(features)
    y = np.array(labels)

    np.savez_compressed(cache_path, X=X, y=y)

    print(f"[SUMMARY] Dataset processing complete. Cached to: {cache_path}")
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
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X, y = parse_dataset()
    print("[INFO] Cleaning dataset...")
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_bal, y_bal = balance_dataset(X_scaled, y_encoded)
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)
    checkpoint_path = os.path.join(MODEL_DIR, 'emotion_model.keras')
    if os.path.exists(checkpoint_path):
        print("[INFO] Resuming from saved checkpoint...")
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        print("[INFO] No checkpoint found. Building a new model...")
        model = build_model(input_dim=X_train.shape[1], num_classes=len(np.unique(y_bal)))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=20),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.95**epoch),
        tf.keras.callbacks.CSVLogger(output_log),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, timestamp))
    ]
    
    print(f"[DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG] y_train unique values: {np.unique(y_train)}")
    history = model.fit(
        X_train, tf.keras.utils.to_categorical(y_train),
        validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1
    )
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    print("\n[Classification Report]\n")
    print(classification_report(y_test, y_pred_labels, target_names=le.classes_))
    print(f"[INFO] Training complete! Logs saved to: {output_log}")

if __name__ == "__main__":
    main()
