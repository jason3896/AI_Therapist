import os
import csv
from collections import deque
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel
from elevenlabs import ElevenLabs
from elevenlabs import ElevenLabs, play
import time
from pathlib import Path
import glob
import chromadb
from chromadb.utils import embedding_functions
import uuid


'''
Dependencies for elevenLabs text to speech
    1) sk_9f9ea317311bc0c0f0da8e16a6127ad17fb34fe8d21b3bd6
    2) pip install elevenlabs
    3) Also have to install ffmpeg
'''

'''
Install for Chroma for RAG to have history with the user
    1) pip install chromadb
    2) pip install sentence-transformers
'''

# Setting up LLM
init(project="ai-therapist-456121", location="us-central1")
model = GenerativeModel("publishers/meta/models/llama-3.3-70b-instruct-maas")

# ElevenLabs api information
api_key = "sk_9f9ea317311bc0c0f0da8e16a6127ad17fb34fe8d21b3bd6"
client = ElevenLabs(api_key=api_key)

# Setting up chroma for RAG
hf_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    name="user_chat_memory"
)

'''
# Add a collection (where the user querys and LLM responses will be stored)
chroma_collection.add(
    documents=["User said they feel anxious at night", "We discussed mindfulness for sleep"],
    ids=["doc1", "doc2"],
    embedding_function=hf_embeddings
)

# Query
results = chroma_collection.query(
    query_texts=["How do I stop overthinking at night?"],
    n_results=2
)

To do next time: make this work to have some sort of history
'''

# Get the newest emotion log file in the folder
# Must be done this way because the emotion logs file names include the time, so its not just one big file
def get_latest_log_file(log_dir="src/facial/data/emotion_logs"):
    log_files = glob.glob(f"{log_dir}/emotion_log_*.csv")
    if not log_files:
        raise FileNotFoundError("No emotion log files found.")
    latest_file = max(log_files, key=os.path.getctime)
    return Path(latest_file)  


""" Function below gives an output like {
  "angry": 1.04,
  "disgust": 0.01,
  "fear": 7.24,
  "happy": 2.3,
  "sad": 15.7,
  "surprise": 0.02,
  "neutral": 72.9
}
where it averages out the emotions in the emotion window
"""

def average_emotions(window):
    totals = {k: 0 for k in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}
    for entry in window:
        for k in totals:
            totals[k] += entry[k]
    return {k: round(totals[k] / len(window), 3) for k in totals}

# Grab the first and second most dominant emotions to be used in our query
def format_emotion_context(average):
    sorted_emotions = sorted(average.items(), key=lambda x: x[1], reverse=True)
    dominant = sorted_emotions[0]
    second = sorted_emotions[1]

    print(dominant[0])
    print(dominant[1])
    print(second[0])
    print(second[1],"\n")
    
    return (
        f"The user currently appears primarily {dominant[0]} ({dominant[1]}%), "
        f"with some signs of {second[0]} ({second[1]}%). "
        "Generate a response that acknowledges their emotional state and encourages gentle engagement."
    )
    
def speak(text):
    audio = client.generate(
        text=text,
        voice="Rosie",
        model="eleven_monolingual_v1"
    )
    play(audio) 

# Rolling window of the last N frames
# It creates a double-ended queue (a deque) from, with a maximum length of 10.
# where if it adds something and the cap is reached, it enques and deques at the same time
emotion_window = deque(maxlen=10)
last_timestamp = None

def get_latest_emotion(file_path):
    with open(file_path) as f:
        reader = list(csv.reader(f))
        if len(reader) < 2:
            return None
        last_row = reader[-1]
        return {
            "timestamp": last_row[0],
            "dominant": last_row[1],
            "confidence": float(last_row[2]),
            "angry": float(last_row[3]),
            "disgust": float(last_row[4]),
            "fear": float(last_row[5]),
            "happy": float(last_row[6]),
            "sad": float(last_row[7]),
            "surprise": float(last_row[8]),
            "neutral": float(last_row[9]),
        }

# Get the latest file
emotion_file = get_latest_log_file()  
print(f"File being used: {emotion_file}")


while True:
    
    # Add the newest emotions
    latest = get_latest_emotion(emotion_file)
    if latest and latest["timestamp"] != last_timestamp:
        last_timestamp = latest["timestamp"]
        emotion_window.append(latest)

    # Make sure have atleast 10 previous emotions in the queue to reference
    if len(emotion_window) >= 5:
        user_input = input("What do you want to say? ")

        if user_input.strip():  # Only respond if input isn't empty
            emotion_context = format_emotion_context(average_emotions(emotion_window))
            
            results = chroma_collection.query(
                query_texts=[user_input],
                n_results=2
            )
            
            retrieved_docs = "\n".join(results["documents"][0]) if results["documents"] else ""
            
            prompt = (
                f"{emotion_context}\n\n"
                f"Relevant past thoughts:\n{retrieved_docs}\n\n"
                f"User: {user_input}\n"
                "Therapist:"
            )
            
            response = model.generate_content(prompt)
            
            
            embeddings = hf_embeddings([user_input, response.text])
            chroma_collection.add(
                documents=[user_input, response.text],
                ids=[f"user-{uuid.uuid4()}", f"therapist-{uuid.uuid4()}"],
                embeddings=embeddings
            )
            
            print(f"Therapist: {response.text}")
            speak(response.text)

    time.sleep(1) # Sleep for a second to give some time for other processess




