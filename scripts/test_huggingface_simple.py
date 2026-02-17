"""
Simple test to verify HuggingFace API works with different models.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    print("ERROR: HUGGINGFACE_API_KEY not set")
    exit(1)

print(f"Testing HuggingFace API with key: {api_key[:5]}...{api_key[-5:]}")
print()

# Try different models
test_models = [
    "mistralai/Mistral-7B-Instruct-v0.1",  # Older version
    "mistralai/Mistral-7B-Instruct-v0.2",  # Your specified model
    "meta-llama/Llama-2-7b-chat-hf",  # Alternative
    "google/flan-t5-large",  # Smaller, definitely available
]

import requests

for model in test_models:
    print(f"Testing model: {model}")
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "inputs": "What is COPD? Answer in one sentence.",
            "parameters": {"max_new_tokens": 50}
        }
        
        url = f"https://api-inference.huggingface.co/models/{model}"
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                text = result[0].get("generated_text", "")
            else:
                text = result.get("generated_text", "")
            print(f"  [SUCCESS] Response: {text[:100]}...")
            print(f"\n✅ Working model found: {model}")
            print(f"   Update your .env file:")
            print(f"   HUGGINGFACE_MODEL={model}")
            break
        elif response.status_code == 503:
            print(f"  [INFO] Model is loading (503) - this model exists but needs to be loaded")
        elif response.status_code == 410:
            print(f"  [FAIL] Endpoint deprecated (410)")
        else:
            print(f"  [FAIL] Error: {response.text[:200]}")
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
    
    print()

print("\nIf all models fail, the issue might be:")
print("1. API key doesn't have access to these models")
print("2. Models need to be deployed to Inference Endpoints")
print("3. Need to use HuggingFace Spaces or different API")
