import os
import time
import joblib
import parselmouth
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from datetime import datetime
from colorama import Fore, Style, init as colorama_init
from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Initialize colorama
colorama_init(autoreset=True)

# Timestamp for logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'output_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

# Paths
data_dir = './crema-d'
log_csv = os.path.join(output_dir, 'training_log.csv')

# Initialize logging DataFrame
log_df = pd.DataFrame(columns=["params", "mean_test_score", "std_test_score", "rank_test_score"])

# Function to extract features from audio file
def extract_features(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        mean_pitch = np.mean(pitch_values) if pitch_values.size else 0

        intensity = snd.to_intensity()
        from parselmouth.praat import call
        mean_intensity = call(intensity, "Get mean", 0, 0, "energy")

        duration = snd.get_total_duration()

        return np.array([mean_pitch, mean_intensity, duration])
    except Exception as e:
        print(Fore.YELLOW + f"[WARNING] Skipped {file_path}: {e}" + Style.RESET_ALL)
        return None

# Prepare data
features = []
labels = []
files = list(os.listdir(data_dir))
print(Fore.CYAN + "[INFO] Extracting features..." + Style.RESET_ALL)

for file in tqdm(files, desc="Feature Extraction"):
    if file.endswith(".wav"):
        label = file.split('_')[2]  # Emotion label
        mapping = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fearful",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad"
        }
        emotion = mapping.get(label, None)
        if emotion is None:
            continue

        file_path = os.path.join(data_dir, file)
        feature = extract_features(file_path)
        if feature is not None:
            features.append(feature)
            labels.append(emotion)

print(Fore.GREEN + f"\n[INFO] Extracted features from {len(features)} files out of {len(files)} total." + Style.RESET_ALL)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Balance classes with SMOTE + Tomek
print(Fore.CYAN + "[INFO] Applying SMOTE + Tomek..." + Style.RESET_ALL)
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

print(Fore.GREEN + f"[INFO] After SMOTE + Tomek class distribution: {Counter(y_train_resampled)}" + Style.RESET_ALL)

# Define model
model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    verbosity=0,
    use_label_encoder=False
)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}

# Grid search
print(Fore.CYAN + "\n[INFO] Starting Grid Search with progress bar..." + Style.RESET_ALL)
grid = GridSearchCV(model, param_grid, cv=3, verbose=0, n_jobs=-1, scoring='accuracy', return_train_score=True)

with tqdm(total=len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])) as pbar:
    def progress_callback(*args, **kwargs):
        pbar.update(1)

    grid.fit(X_train_resampled, y_train_resampled)
    pbar.close()

# Save grid results to CSV
results = pd.DataFrame(grid.cv_results_)
results.to_csv(log_csv, index=False)
print(Fore.GREEN + f"[INFO] Training log saved to {log_csv}" + Style.RESET_ALL)

# Evaluation
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(Fore.YELLOW + "\n[Classification Report]\n" + report + Style.RESET_ALL)

# Save report to text file
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Save model and preprocessors
joblib.dump(best_model, os.path.join('models', f'xgb_model_{timestamp}.joblib'))
joblib.dump(scaler, os.path.join('models', f'scaler_{timestamp}.joblib'))
joblib.dump(label_encoder, os.path.join('models', f'label_encoder_{timestamp}.joblib'))
print(Fore.GREEN + f"[INFO] Optimized model and encoders saved to 'models/'" + Style.RESET_ALL)
print(Fore.GREEN + f"[INFO] Timestamp for this run: {timestamp}" + Style.RESET_ALL)
