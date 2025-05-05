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

""" 
Bert is a pretrained model developed by Google that is designed to understand context and meaning of
words in a sentence by looking at both the left and right sides of a word simultaneously
For the purpose of this model, BERT turns text into rich, context-aware numerical features that the neural 
network can use to help classify emotions, especially when combined with audio features from YAMNet.
The tokenizer is used for the bert-model to to its job.
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
bert_model.trainable = True  # Allow BERT to fine-tune to our model

# === Feature Extraction Functions ===
def extract_yamnet_features(file_path):
    try:
        
        # Load the audio file 
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.squeeze()  # Make sure 1D

        # Since yamnet expect a sample rate of 16000, must resample it to 16000 Hz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform.unsqueeze(0)).squeeze()
        
        # Since yamnet is a tensorflow model, it expects Numpy inputs
        waveform = waveform.numpy()  # Convert to numpy

        # Feed teh waveform into the yammnet_model
        scores, embeddings, spectrogram = yamnet_model(waveform)
        embedding = tf.reduce_mean(embeddings, axis=0)  # Average across time
        return embedding.numpy()

    except Exception as e:
        print(f"[ERROR] Failed to extract YAMNet features: {e}")
        return None

"""
This function is used to take text and return a context-aware vector embeddings generated
by the BERT model
"""
def embed_text(text):
    try:
        # First, tokenize the input
        tokens = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=64)
        # Feed that text into the bert_model
        output = bert_model(**tokens)
        
        # Caputre the last_hidden_state which is a 768-dimensional vector used for capturing semantic meaning to feed to our model
        return output.last_hidden_state[:, 0, :]
    except Exception as e:
        print(f"[ERROR] Failed to embed text: {e}")
        return None

# === Dataset Loader ===
"""
Extracts audio and text features along with the emotion label from a given dataset row.

This function takes a single row from the MELD dataset and attempts to:
1. Construct the expected audio file path using Dialogue_ID and Utterance_ID.
2. Load the audio file and extract audio features using YAMNet.
3. Convert the corresponding utterance text into a BERT embedding.
4. Return a tuple of (audio features, text features, emotion label) if successful; otherwise, returns None.
"""
def process_row(row, audio_base_path):
    # Create the file
    file_id = f"{row['Dialogue_ID']}_{row['Utterance_ID']}"
    
    # Create the associated audio file
    audio_path = os.path.join(audio_base_path, f"{file_id}.wav")
    
    if not os.path.exists(audio_path):
        return None
    
    # Extract the associated audio features using yamnet
    audio_feat = extract_yamnet_features(audio_path)
    
    # Extract the associated text features using BERT
    text_feat = embed_text(row['Utterance'])
    
    # If bot where succesful return audio_feat: 1024-dim vector from YAMNet, text_feat: 768-dim vector from BERT , row['Emotion']: the target label (e.g., "happy", "angry", etc.)
    if audio_feat is not None and text_feat is not None:
        return audio_feat, text_feat.numpy().squeeze(), row['Emotion'].lower()
    return None

# Wrapper function to unpack arguments for multiprocessing compatibility.
# Accepts a tuple of (row, audio_base_path) and passes them to process_row.
def process_row_with_path(args):
    row, audio_base_path = args
    return process_row(row, audio_base_path)

"""
# Loads the MELD dataset and extracts features in parallel using multiprocessing.
    - Reads the CSV containing utterances and emotion labels.
    - Pairs each row with the audio base path for processing.
    - Uses a process pool to extract audio/text features and labels concurrently.
    - Filters out any failed (None) results.
    - Returns audio features, text features, and emotion labels as NumPy arrays.
"""
def load_meld_dataset_parallel(meld_csv_path, audio_base_path):
    df = pd.read_csv(meld_csv_path)  # Load the MELD metadata (text + emotion)
    
    # Prepare a list of (row, path) tuples for parallel processing
    rows = [(row, audio_base_path) for _, row in df.iterrows()]
    
    # Run process_row_with_path in parallel across CPU cores with a progress bar
    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_row_with_path, rows), total=len(rows)))

    # Filter out failed rows and unpack the successful results
    # So results returned something like "results = [(a1, t1, l1),(a2, t2, l2)]" and this turn it into
    # audio_features = (a1, a2, a3), etc.
    audio_features, text_features, labels = zip(*[r for r in results if r is not None])
    
    return np.array(audio_features), np.array(text_features), np.array(labels)


# === Main Training ===
if __name__ == "__main__":
    
    # Define paths to the data
    audio_path = os.path.join(DATA_DIR, "MELD", "train_audio")
    csv_path = os.path.join(DATA_DIR, "MELD", "train_sent_emo.csv")

    # load the dataset
    # X_audio: 1024-dim YAMNet features
    # X_text: 768-dim BERT features
    # y: emotion labels (strings)
    X_audio, X_text, y = load_meld_dataset_parallel(csv_path, audio_path)

    # Since the original y labels are strings like "happy", "sad", etc, neural networks cant process those, so they need numerical labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y) # Converts labels to integers
    # On hot encoding turns each label into a vector.
    # Ex) Happy -> 0, and we have three emotion, then the output = [1, 0, 0]
    y_cat = tf.keras.utils.to_categorical(y_encoded) # One-hot encode for classification

    # Since yamnet might have wildly varying scales, normalize between 0 and 1
    scaler = StandardScaler()
    X_audio_scaled = scaler.fit_transform(X_audio) # Normalize audio features

    # Compute weights for each emotion class so that the model doesnt get biased by frequent classes
    # For example, if the MELD dataset has hardly any angry emotions, then its will be weighed higher than others
    class_weights = dict(zip(
        np.unique(y_encoded),
        compute_class_weight(class_weight="balanced", classes=np.unique(y_encoded), y=y_encoded)
    ))

    # Split the data into  80% for training and 20% for validation
    X_audio_train, X_audio_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
        X_audio_scaled, X_text, y_cat, test_size=0.2, stratify=y_encoded, random_state=42)

    print(f"[INFO] Training on {len(y_train)} samples, validating on {len(y_val)} samples")

    # === Build Model ===
    # Inputs: audio (1024) and text (768)
    # Hidden layers: 256 → concat → 128
    # Output: softmax with num_classes units
    
    # Create an input layer for audio features (e.g., 1024-dimensional YAMNet vector)
    audio_input = tf.keras.Input(shape=(X_audio.shape[1],), name='audio_input')

    # Create an input layer for text features (768-dimensional BERT [CLS] embedding)
    text_input = tf.keras.Input(shape=(TEXT_EMBEDDING_DIM,), name='text_input')

    # Add a dense layer to process the audio input (compress to 256 units with ReLU activation)
    x_audio = tf.keras.layers.Dense(256, activation='relu')(audio_input)

    # Add a dense layer to process the text input (compress to 256 units with ReLU activation)
    x_text = tf.keras.layers.Dense(256, activation='relu')(text_input)

    # Concatenate the processed audio and text feature vectors into one combined feature vector
    x = tf.keras.layers.Concatenate()([x_audio, x_text])  # Shape: (512,)

    # Add another dense layer to learn joint audio-text representations (128 units, ReLU activation)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    # Final output layer: one neuron per emotion class, with softmax to produce class probabilities
    out = tf.keras.layers.Dense(y_cat.shape[1], activation='softmax')(x)

    # Defines a model that takes in boh audio and text as input and produces a softmax probability vector
    model = tf.keras.Model(inputs=[audio_input, text_input], outputs=out)

    # Create and Adam optimizer with a low learning-rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)  # Very low LR for finetuning
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks to check forsuch as early stopping, checkpointing, and logging the data
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"), save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "training_log.csv"))
    ]

    # Create and generate the model with previousy defined strucutre
    history = model.fit(
        [X_audio_train, X_text_train], y_train,
        validation_data=([X_audio_val, X_text_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save the data to files
    model.save(os.path.join(MODEL_DIR, "meld_multimodal_model.keras"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "audio_scaler.pkl"))

    print("[INFO] Training complete. Models and artifacts saved to './v2_models/'.")
    