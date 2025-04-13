import os
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel

# 1. Set credentials (only needed if not using gcloud CLI auth)

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
