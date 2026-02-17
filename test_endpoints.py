"""
Simple test script for API endpoints.
Make sure the server is running: uvicorn src.api.main:app --reload
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Testing API Endpoints")
print("=" * 60)

# Test 1: /ask endpoint
print("\n1. Testing POST /ask")
print("-" * 60)
try:
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": "What is COPD?",
            "guideline_query": "COPD symptoms"
        },
        timeout=120
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Answer: {data.get('answer', 'No answer')[:200]}...")
        print(f"Evidence count: {data.get('num_evidence', 0)}")
    else:
        print(f"Error: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: /chat endpoint (text only)
print("\n2. Testing POST /chat (text only)")
print("-" * 60)
try:
    response = requests.post(
        f"{BASE_URL}/chat",
        data={
            "question": "What is COPD?",
            "guideline_query": "COPD symptoms"
        },
        timeout=120
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Answer: {data.get('answer', 'No answer')[:200]}...")
        print(f"Evidence count: {data.get('num_evidence', 0)}")
    else:
        print(f"Error: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: /predict endpoint
print("\n3. Testing POST /predict")
print("-" * 60)
try:
    response = requests.post(
        f"{BASE_URL}/predict",
        json={
            "embedding": [0.0] * 144,
            "guideline_query": "COPD risk"
        },
        timeout=120
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Diagnosis: {data.get('diagnosis')}")
        print(f"Confidence: {data.get('confidence')}")
    else:
        print(f"Error: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: /explain endpoint
print("\n4. Testing POST /explain")
print("-" * 60)
try:
    response = requests.post(
        f"{BASE_URL}/explain",
        json={
            "embedding": [0.0] * 144,
            "question": "Why is the COPD risk score what it is?",
            "guideline_query": "COPD risk"
        },
        timeout=120
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Question: {data.get('question')}")
        print(f"Evidence count: {data.get('num_evidence', 0)}")
    else:
        print(f"Error: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)
