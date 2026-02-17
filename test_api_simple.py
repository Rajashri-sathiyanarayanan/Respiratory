"""
Simple script to test the API endpoints.
Make sure the API server is running first: python -m uvicorn src.api.main:app --reload
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("="*60)
    print("Testing /health endpoint...")
    print("="*60)
    try:
        response = requests.get(f"{API_URL}/health")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the API server is running:")
        print("  python -m uvicorn src.api.main:app --reload")
        return False

def test_ask(question, guideline_query=None):
    """Test /ask endpoint"""
    print("\n" + "="*60)
    print(f"Question: {question}")
    print("="*60)
    
    payload = {
        "question": question,
        "guideline_query": guideline_query or question
    }
    
    try:
        response = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        print(f"\nAnswer ({data.get('num_evidence', 0)} evidence items):")
        print("-" * 60)
        print(data.get('answer', 'No answer'))
        print("-" * 60)
        print(f"\nLLM Model: {data.get('llm_model', 'unknown')}")
        print(f"LLM Provider: {data.get('llm_provider', 'unknown')}")
        
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Run tests"""
    print("\n" + "="*60)
    print("Respiratory Disease Q&A API Test")
    print("="*60)
    
    # Test health first
    if not test_health():
        return
    
    # Test questions
    questions = [
        "What is COPD?",
        "What are the symptoms of asthma?",
        "How is bronchiectasis treated?",
        "What is the difference between COPD and asthma?",
    ]
    
    print("\n" + "="*60)
    print("Testing Question Answering")
    print("="*60)
    
    for question in questions:
        test_ask(question)
        print("\n")
    
    print("="*60)
    print("Tests completed!")
    print("="*60)
    print("\nTo test interactively, start the server and visit:")
    print("  http://localhost:8000/docs")

if __name__ == "__main__":
    main()
