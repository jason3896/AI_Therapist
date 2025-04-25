import os
import joblib
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
from datetime import datetime

# Prepare timestamped output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./output/deep_test_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Load models
print("[INFO] Loading models...")
model = joblib.load("./models/emotion_model.keras")
scaler = joblib.load("./models/scaler.pkl")
label_encoder = joblib.load("./models/label_encoder.pkl")
expected_features = scaler.mean_.shape[0]
print(f"[INFO] Model expects {expected_features} features.")

# Load test dataset
test_dir = "./data/test"
results = []

print("[INFO] Processing test dataset...")
for root, _, files in os.walk(test_dir):
    for file in files:
        if file.endswith(".wav"):
            try:
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)
                
                # Deep extraction
                if len(y) < 2048:
                    y = np.pad(y, (0, 2048 - len(y)))

                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                mel = librosa.feature.melspectrogram(y=y, sr=sr)
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
                pitch = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                pitch_mean = np.nanmean(pitch[0]) if pitch[0] is not None else 0

                features = np.hstack([
                    np.mean(mfcc, axis=1),
                    np.std(mfcc, axis=1),
                    np.mean(chroma, axis=1),
                    np.mean(mel, axis=1)[:13],  # limit mel features
                    np.mean(contrast, axis=1),
                    np.mean(tonnetz, axis=1),
                    pitch_mean
                ])

                # Auto-align features
                if len(features) > expected_features:
                    features = features[:expected_features]
                elif len(features) < expected_features:
                    features = np.pad(features, (0, expected_features - len(features)))

                scaled = scaler.transform([features])
                prediction = model.predict(scaled)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                actual_label = os.path.basename(root)

                results.append({
                    "file": file,
                    "predicted": predicted_label,
                    "actual": actual_label
                })
            except Exception as e:
                print(f"[ERROR] Failed processing {file}: {e}")

# Save predictions
df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, "predictions.csv")
excel_path = os.path.join(output_dir, "predictions.xlsx")
df.to_csv(csv_path, index=False)
df.to_excel(excel_path, index=False)
print(f"[INFO] Predictions saved to {csv_path} and {excel_path}")

# Classification report
y_true = df["actual"]
y_pred = df["predicted"]

report = classification_report(y_true, y_pred, zero_division=0)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)
print("[INFO] Classification report saved.")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()
print("[INFO] Confusion matrix saved.")

# Precision-recall curve
y_true_bin = pd.get_dummies(y_true, columns=label_encoder.classes_)
y_pred_bin = pd.get_dummies(y_pred, columns=label_encoder.classes_).reindex(columns=y_true_bin.columns, fill_value=0)

for label in y_true_bin.columns:
    precision, recall, _ = precision_recall_curve(y_true_bin[label], y_pred_bin[label])
    plt.plot(recall, precision, label=f"{label} (AUC = {auc(recall, precision):.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
plt.close()
print("[INFO] Precision-recall curve saved.")

print(f"[✅] Deep test complete! Results saved to {output_dir}")
