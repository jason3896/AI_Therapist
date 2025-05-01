import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from multiprocessing import Pool
import librosa
import torchaudio

# === Environment Setup ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === Paths ===
DATA_DIR = "./data"
MODEL_DIR = "./v2_models"
OUTPUT_DIR = "./v2_output"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Configuration ===
SAMPLE_RATE = 16000  # YAMNet expects 16kHz
TEXT_EMBEDDING_DIM = 768
BATCH_SIZE = 32
EPOCHS = 50

# === Load Models ===
print("[INFO] Loading YAMNet and BERT...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
bert_model.trainable = True  # Allow BERT to fine-tune

# === Feature Extraction Functions ===
def extract_yamnet_features(file_path):
    try:
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.squeeze()  # <<< Make sure 1D

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform.unsqueeze(0)).squeeze()

        waveform = waveform.numpy()  # Convert to numpy

        scores, embeddings, spectrogram = yamnet_model(waveform)
        embedding = tf.reduce_mean(embeddings, axis=0)  # Average across time
        return embedding.numpy()

    except Exception as e:
        print(f"[ERROR] Failed to extract YAMNet features: {e}")
        return None

def embed_text(text):
    try:
        tokens = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=64)
        output = bert_model(**tokens)
        return output.last_hidden_state[:, 0, :]
    except Exception as e:
        print(f"[ERROR] Failed to embed text: {e}")
        return None

# === Dataset Loader ===
def process_row(row, audio_base_path):
    file_id = f"{row['Dialogue_ID']}_{row['Utterance_ID']}"
    audio_path = os.path.join(audio_base_path, f"{file_id}.wav")
    if not os.path.exists(audio_path):
        return None
    audio_feat = extract_yamnet_features(audio_path)
    text_feat = embed_text(row['Utterance'])
    if audio_feat is not None and text_feat is not None:
        return audio_feat, text_feat.numpy().squeeze(), row['Emotion'].lower()
    return None

def process_row_with_path(args):
    row, audio_base_path = args
    return process_row(row, audio_base_path)

def load_meld_dataset_parallel(meld_csv_path, audio_base_path):
    df = pd.read_csv(meld_csv_path)
    with Pool(processes=os.cpu_count()) as pool:
        rows = [(row, audio_base_path) for _, row in df.iterrows()]
        results = list(tqdm(pool.imap(process_row_with_path, rows), total=len(rows)))

    audio_features, text_features, labels = zip(*[r for r in results if r is not None])
    return np.array(audio_features), np.array(text_features), np.array(labels)

# === Main Training ===
if __name__ == "__main__":
    audio_path = os.path.join(DATA_DIR, "MELD", "train_audio")
    csv_path = os.path.join(DATA_DIR, "MELD", "train_sent_emo.csv")

    X_audio, X_text, y = load_meld_dataset_parallel(csv_path, audio_path)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = tf.keras.utils.to_categorical(y_encoded)

    scaler = StandardScaler()
    X_audio_scaled = scaler.fit_transform(X_audio)

    class_weights = dict(zip(
        np.unique(y_encoded),
        compute_class_weight(class_weight="balanced", classes=np.unique(y_encoded), y=y_encoded)
    ))

    X_audio_train, X_audio_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
        X_audio_scaled, X_text, y_cat, test_size=0.2, stratify=y_encoded, random_state=42)

    print(f"[INFO] Training on {len(y_train)} samples, validating on {len(y_val)} samples")

    # === Build Model ===
    audio_input = tf.keras.Input(shape=(X_audio.shape[1],), name='audio_input')
    text_input = tf.keras.Input(shape=(TEXT_EMBEDDING_DIM,), name='text_input')
    x_audio = tf.keras.layers.Dense(256, activation='relu')(audio_input)
    x_text = tf.keras.layers.Dense(256, activation='relu')(text_input)
    x = tf.keras.layers.Concatenate()([x_audio, x_text])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(y_cat.shape[1], activation='softmax')(x)

    model = tf.keras.Model(inputs=[audio_input, text_input], outputs=out)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)  # Very low LR for finetuning 2e-5
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"), save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "training_log.csv"))
    ]

    history = model.fit(
        [X_audio_train, X_text_train], y_train,
        validation_data=([X_audio_val, X_text_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "meld_multimodal_model.keras"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "audio_scaler.pkl"))

    print("[INFO] Training complete. Models and artifacts saved to './v2_models/'.")