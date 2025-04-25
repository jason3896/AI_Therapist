import os
import subprocess
import pandas as pd
from tqdm import tqdm

# === CONFIGURATION ===
BASE_DIR = "./data/MELD"
SPLITS = {
    "train": "train",
    "dev": "dev",
    "test": "test"
}

SAMPLE_RATE = 48000

def convert_videos_to_wav(split_name, video_subdir):
    video_dir = os.path.join(BASE_DIR, video_subdir)
    csv_path = os.path.join(BASE_DIR, f"{split_name}_sent_emo.csv")
    audio_dir = os.path.join(BASE_DIR, f"{split_name}_audio")

    os.makedirs(audio_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    missing_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{split_name.upper()}] Converting to WAV"):
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        video_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        audio_filename = f"{dialogue_id}_{utterance_id}.wav"
        audio_path = os.path.join(audio_dir, audio_filename)

        if not os.path.exists(video_path):
            print(f"[WARN] Missing video: {video_filename}")
            missing_count += 1
            continue

        cmd = [
            "ffmpeg", "-i", video_path,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-y",
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[{split_name.upper()}] Audio conversion done. Missing: {missing_count} files\n")

# === Run for all defined splits
for split_name, video_subdir in SPLITS.items():
    convert_videos_to_wav(split_name, video_subdir)

print("[DONE] All splits converted to audio.")
