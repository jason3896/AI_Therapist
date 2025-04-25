import os
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel

# 1. Set credentials (only needed if not using gcloud CLI auth)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/home/raff/Documents/tamu/CSCE420/Project/AI_Therapist/GCKey/ai-therapist-456121-4ef8e7281226.json"

# 2. Init Vertex AI
init(
    project="ai-therapist-456121",
    location="us-central1"
)

# 3. Load the model
model = GenerativeModel("publishers/meta/models/llama-3.3-70b-instruct-maas")

# 4. Send a prompt
response = model.generate_content("How can I manage stress during finals week?")
print(response.text)
