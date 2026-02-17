"""
Test script to verify Ollama integration and full system functionality.
"""
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from src.llm.providers import OllamaProvider, get_llm_provider
from scripts.load_vector_store import load_vector_store
from scripts.load_guideline_store import load_guideline_store
from src.rag.engine import EvidenceFirstRAG
from src.prediction.llm_predictor import LLMProgressionPredictor
from scripts.ingest_guidelines import ingest_rag_guidelines

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def create_guideline_query_embedding(query_text: str) -> np.ndarray:
    """Create guideline query embedding."""
    if not SKLEARN_AVAILABLE:
        return np.zeros(384, dtype="float32")
    
    try:
        chunks = ingest_rag_guidelines()
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        chunks_tfidf = vectorizer.fit_transform(chunks)
        n_components = min(384, chunks_tfidf.shape[1], len(chunks))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(chunks_tfidf)
        
        query_tfidf = vectorizer.transform([query_text])
        query_embedding = svd.transform(query_tfidf)[0]
        
        if query_embedding.shape[0] < 384:
            padding = np.zeros(384 - query_embedding.shape[0], dtype="float32")
            query_embedding = np.concatenate([query_embedding, padding])
        
        return query_embedding.astype("float32")
    except:
        return np.zeros(384, dtype="float32")


def test_ollama_connection():
    """Test Ollama connection."""
    print("="*60)
    print("Testing Ollama Connection")
    print("="*60)
    
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Check which models are available
    model = None
    try:
        import requests
        models_response = requests.get(f"{base_url}/api/tags", timeout=2)
        if models_response.status_code == 200:
            models_data = models_response.json()
            available_models = [m.get("name", "") for m in models_data.get("models", [])]
            if available_models:
                # Prefer mistral if available, else use first available
                if any("mistral" in m.lower() for m in available_models):
                    model = next(m for m in available_models if "mistral" in m.lower())
                else:
                    model = available_models[0]
                print(f"   Found available models: {', '.join(available_models)}")
                print(f"   Using: {model}")
            else:
                model = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_0")
        else:
            model = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_0")
    except Exception as e:
        print(f"   Warning: Could not check available models: {e}")
        model = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_0")
    
    if not model:
        model = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_0")
    
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print()
    
    try:
        provider = OllamaProvider(base_url=base_url, model=model)
        
        if not provider._available:
            print("[FAIL] Ollama is not accessible")
            print("       Make sure Ollama is running: ollama serve")
            return False
        
        print("[OK] Ollama is accessible")
        print("\nTesting generation...")
        
        response = provider.generate(
            "What is COPD? Answer in one sentence.",
            max_tokens=50
        )
        
        print(f"[SUCCESS] Response: {response[:150]}...")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_full_system():
    """Test full RAG + LLM system with Ollama."""
    print("\n" + "="*60)
    print("Testing Full RAG System with Ollama")
    print("="*60)
    
    # Load vector stores
    print("\n1. Loading vector stores...")
    patient_store = load_vector_store()
    guideline_store = load_guideline_store()
    print(f"   Loaded patient vector store (dim={patient_store.dim}, {len(patient_store.metadata)} samples)")
    print(f"   Loaded guideline vector store (dim={guideline_store.dim}, {len(guideline_store.metadata)} chunks)")
    
    # Initialize RAG engine
    print("\n2. Initializing RAG engine...")
    rag_engine = EvidenceFirstRAG(patient_store, llm_provider="ollama")
    
    # Get LLM provider
    print("\n3. Getting LLM provider...")
    llm_provider = get_llm_provider("ollama")
    if llm_provider:
        print(f"   [OK] Using: {type(llm_provider).__name__} (model: {llm_provider.model})")
    else:
        print("   [WARNING] No LLM provider - will use fallback")
    
    # Initialize predictor
    print("\n4. Initializing LLM predictor...")
    predictor = LLMProgressionPredictor(
        rag_engine=rag_engine,
        llm_provider=llm_provider,
        guideline_store=guideline_store
    )
    
    # Create test query embedding (random for demo)
    print("\n5. Creating test query...")
    test_embedding = np.random.randn(144).astype("float32")
    test_guideline_query = "COPD diagnosis and treatment"
    guideline_query_emb = create_guideline_query_embedding(test_guideline_query)
    
    # Test prediction
    print("\n6. Testing prediction...")
    print("   Query: Patient with respiratory symptoms")
    print("   Guideline query: COPD diagnosis and treatment")
    print()
    
    try:
        pred = predictor.predict_from_embedding(
            test_embedding,
            guideline_query_embedding=guideline_query_emb
        )
        
        print("="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Diagnosis: {pred.diagnosis}")
        print(f"Diagnosis Confidence: {pred.diagnosis_confidence:.2f}")
        print(f"COPD Deterioration Score: {pred.early_copd_deterioration_score:.2f}")
        print(f"Asthma Attack Probability: {pred.asthma_attack_probability:.2f}")
        print(f"Viral vs Non-viral: {pred.viral_vs_nonviral}")
        print(f"Hospitalization Risk: {pred.hospitalization_risk_proxy:.2f}")
        print(f"Confidence: {pred.confidence:.2f}")
        print(f"\nPrescription:")
        print(f"  {pred.prescription}")
        print(f"\nReasoning:")
        print(f"  {pred.reasoning[:300]}...")
        print(f"\nEvidence: {len(pred.evidence)} items retrieved")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qa_system():
    """Test Q&A system for respiratory diseases."""
    print("\n" + "="*60)
    print("Testing Q&A System")
    print("="*60)
    
    # Load stores
    patient_store = load_vector_store()
    guideline_store = load_guideline_store()
    rag_engine = EvidenceFirstRAG(patient_store, llm_provider="ollama")
    
    # Test questions
    questions = [
        "What is COPD and how is it diagnosed?",
        "What are the symptoms of asthma?",
        "How is bronchiectasis treated?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)
        
        # Create guideline query embedding
        guideline_query_emb = create_guideline_query_embedding(question)
        
        # Create dummy patient embedding for retrieval
        test_embedding = np.random.randn(144).astype("float32")
        
        # Retrieve evidence
        evidences = rag_engine.retrieve_similar_samples(
            test_embedding,
            k=5,
            guideline_store=guideline_store,
            guideline_query_embedding=guideline_query_emb,
            k_guidelines=3
        )
        
        # Generate explanation
        explanation = rag_engine.generate_explanation(question, evidences)
        
        print(f"Evidence retrieved: {explanation['num_evidence']} items")
        print(f"Explanation:\n{explanation['explanation'][:200]}...")
        print()


def main():
    """Run all tests."""
    print("="*60)
    print("OLLAMA SYSTEM TEST SUITE")
    print("="*60)
    
    # Test 1: Ollama connection
    ollama_ok = test_ollama_connection()
    
    if not ollama_ok:
        print("\n[ERROR] Ollama is not working. Please:")
        print("1. Make sure Ollama is installed: https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print("3. Pull models: ollama pull mistral:7b-instruct-q4_0")
        print("4. Or: ollama pull llama3:8b")
        return
    
    # Test 2: Full system
    system_ok = test_full_system()
    
    # Test 3: Q&A system
    if system_ok:
        test_qa_system()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Ollama Connection: {'[OK]' if ollama_ok else '[FAIL]'}")
    print(f"Full System: {'[OK]' if system_ok else '[FAIL]'}")
    
    if ollama_ok and system_ok:
        print("\n[SUCCESS] System is working with Ollama!")
        print("You can now use the API endpoints for diagnosis and prescriptions.")


if __name__ == "__main__":
    main()
