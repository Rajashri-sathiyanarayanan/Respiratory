"""
Test LLM-based predictions with RAG evidence.
"""
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from src.prediction.llm_predictor import LLMProgressionPredictor
from src.rag.engine import EvidenceFirstRAG
from src.llm.providers import get_llm_provider
from scripts.load_vector_store import load_vector_store
from scripts.load_guideline_store import load_guideline_store
from scripts.ingest_guidelines import ingest_rag_guidelines

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def create_guideline_query_embedding(query_text: str) -> np.ndarray:
    """Create guideline query embedding using TF-IDF (matching build process)."""
    if not SKLEARN_AVAILABLE:
        return None
    
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


def test_llm_predictions():
    """Test LLM-based prediction system."""
    print("="*60)
    print("Testing LLM-Based Predictions with RAG")
    print("="*60)
    
    # Load vector stores
    print("\n1. Loading vector stores...")
    patient_store = load_vector_store()
    guideline_store = load_guideline_store()
    
    # Initialize RAG engine with HuggingFace (e.g., BioMistral) if available
    print("\n2. Initializing RAG engine...")
    rag = EvidenceFirstRAG(patient_store, llm_provider="huggingface")
    
    # Get matching LLM provider
    print("\n3. Checking LLM provider (HuggingFace)...")
    llm_provider = get_llm_provider("huggingface")
    if llm_provider:
        print(f"   [OK] Using LLM provider: {type(llm_provider).__name__}")
    else:
        print("   [WARNING] HuggingFace LLM provider not available - using fallback")
    
    # Initialize LLM predictor
    print("\n4. Initializing LLM predictor...")
    predictor = LLMProgressionPredictor(
        rag_engine=rag,
        llm_provider=llm_provider,
        guideline_store=guideline_store
    )
    
    # Create test query embedding
    print("\n5. Creating test query...")
    query_embedding = np.random.randn(patient_store.dim).astype("float32")
    guideline_query = create_guideline_query_embedding("COPD progression risk based on spirometry")
    
    # Make prediction
    print("\n6. Making LLM-based prediction...")
    print("   (This may take 10-30 seconds depending on LLM provider)")
    
    prediction = predictor.predict_from_embedding(
        query_embedding,
        guideline_query_embedding=guideline_query
    )
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nEarly COPD Deterioration Score: {prediction.early_copd_deterioration_score:.3f}")
    print(f"Asthma Attack Probability: {prediction.asthma_attack_probability:.3f}")
    print(f"Viral vs Non-viral: {prediction.viral_vs_nonviral}")
    print(f"Hospitalization Risk Proxy: {prediction.hospitalization_risk_proxy:.3f}")
    print(f"Confidence: {prediction.confidence:.3f}")
    
    print(f"\nEvidence Retrieved: {len(prediction.evidence)} items")
    patient_count = sum(1 for e in prediction.evidence if e.modality != "guideline")
    guideline_count = sum(1 for e in prediction.evidence if e.modality == "guideline")
    print(f"  - Patient samples: {patient_count}")
    print(f"  - Guideline chunks: {guideline_count}")
    
    print(f"\nLLM Reasoning:")
    print("-" * 60)
    print(prediction.reasoning)
    print("-" * 60)
    
    # Test explanation
    print("\n" + "="*60)
    print("TESTING EXPLANATION GENERATION")
    print("="*60)
    
    question = "Why is the COPD risk score what it is?"
    explanation = predictor.explain_prediction(
        question,
        query_embedding,
        guideline_query_embedding=guideline_query
    )
    
    print(f"\nQuestion: {explanation['question']}")
    print(f"Explanation:\n{explanation['explanation']}")
    
    print("\n" + "="*60)
    print("[SUCCESS] LLM-based prediction system test completed!")
    print("="*60)


if __name__ == "__main__":
    test_llm_predictions()
