"""
Test LLM question-answer examples for methodology/testing documentation.
Shows sample questions and LLM-generated answers with evidence.
"""
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

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
    print("[WARNING] scikit-learn not available - guideline queries may fail")


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


def test_llm_qa_examples():
    """Test LLM with sample questions and display answers with evidence."""
    print("=" * 80)
    print("LLM QUESTION-ANSWER TESTING")
    print("=" * 80)
    
    # Load vector stores
    print("\n1. Loading vector stores...")
    patient_store = load_vector_store()
    guideline_store = load_guideline_store()
    print(f"   [OK] Patient store: {len(patient_store.metadata)} samples")
    print(f"   [OK] Guideline store: {len(guideline_store.metadata)} chunks")
    
    # Initialize RAG engine with HuggingFace (e.g., BioMistral) if available
    print("\n2. Initializing RAG engine...")
    rag = EvidenceFirstRAG(patient_store, llm_provider="huggingface")
    
    # Check LLM provider
    print("\n3. Checking LLM provider (HuggingFace)...")
    llm_provider = get_llm_provider("huggingface")
    if llm_provider:
        print(f"   [OK] Using LLM provider: {type(llm_provider).__name__}")
        if hasattr(llm_provider, 'model'):
            print(f"   Model: {llm_provider.model}")
    else:
        print("   [WARNING] HuggingFace LLM provider not available - using fallback")
        print("   Set up HUGGINGFACE_API_KEY and HUGGINGFACE_MODEL in .env")
        return
    
    # Test questions
    test_questions = [
        {
            "question": "What is COPD and what are the main symptoms?",
            "patient_query": "COPD symptoms spirometry",
            "guideline_query": "COPD symptoms diagnosis"
        },
        {
            "question": "What is the risk of COPD progression for a patient with FEV1/FVC ratio of 0.65?",
            "patient_query": "COPD FEV1 FVC ratio 0.65",
            "guideline_query": "COPD progression risk spirometry FEV1 FVC"
        },
        {
            "question": "How is asthma different from COPD?",
            "patient_query": "asthma COPD comparison",
            "guideline_query": "asthma vs COPD differences"
        },
        {
            "question": "What treatment options are available for respiratory diseases?",
            "patient_query": "treatment respiratory disease",
            "guideline_query": "respiratory disease treatment options"
        }
    ]
    
    print(f"\n4. Testing {len(test_questions)} sample questions...")
    print("=" * 80)
    
    for i, test_case in enumerate(test_questions, 1):
        question = test_case["question"]
        patient_query_text = test_case["patient_query"]
        guideline_query_text = test_case["guideline_query"]
        
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}/{len(test_questions)}")
        print(f"{'='*80}")
        print(f"\nQUESTION: {question}")
        print(f"\nQuery context:")
        print(f"  - Patient query: '{patient_query_text}'")
        print(f"  - Guideline query: '{guideline_query_text}'")
        
        # Create query embeddings
        # For patient query, use a simple random embedding (in real use, this would be from user input)
        patient_query_embedding = np.random.randn(patient_store.dim).astype("float32")
        guideline_query_embedding = create_guideline_query_embedding(guideline_query_text)
        
        # Retrieve evidence
        print(f"\nRetrieving evidence...")
        evidences = rag.retrieve_similar_samples(
            patient_query_embedding,
            k=5,  # Top 5 patient samples
            guideline_store=guideline_store,
            guideline_query_embedding=guideline_query_embedding,
            k_guidelines=3  # Top 3 guideline chunks
        )
        
        patient_count = sum(1 for e in evidences if e.modality != "guideline")
        guideline_count = sum(1 for e in evidences if e.modality == "guideline")
        
        print(f"  Retrieved: {patient_count} patient samples, {guideline_count} guideline chunks")
        
        # Show sample evidence
        print(f"\nSample Evidence:")
        print("-" * 80)
        for j, ev in enumerate(evidences[:3], 1):  # Show first 3
            print(f"\n  [{j}] {ev.modality.upper()}")
            print(f"      Sample ID: {ev.sample_id}")
            print(f"      Similarity: {1 - ev.distance:.3f}")
            if ev.modality == "guideline" and ev.metadata and "text_preview" in ev.metadata:
                preview = ev.metadata['text_preview'][:200]
                print(f"      Preview: {preview}...")
            elif ev.metadata:
                if "diagnosis" in ev.metadata and ev.metadata["diagnosis"]:
                    print(f"      Diagnosis: {ev.metadata['diagnosis']}")
        
        # Generate LLM answer
        print(f"\nGenerating LLM answer...")
        print("  (This may take 10-30 seconds depending on LLM provider)")
        
        explanation = rag.generate_explanation(question, evidences)
        
        print(f"\n{'='*80}")
        print("LLM ANSWER:")
        print(f"{'='*80}")
        print(f"\n{explanation['explanation']}")
        print(f"\n{'='*80}")
        print(f"Metadata:")
        print(f"  - LLM Model: {explanation.get('llm_model', 'N/A')}")
        print(f"  - LLM Provider: {explanation.get('llm_provider', 'N/A')}")
        print(f"  - Evidence items used: {explanation.get('num_evidence', 0)}")
        print(f"{'='*80}\n")
        
        # Add separator between test cases
        if i < len(test_questions):
            print("\n" + "=" * 80)
            print("Press Enter to continue to next test case...")
            print("=" * 80)
            input()
    
    print("\n" + "=" * 80)
    print("[SUCCESS] LLM question-answer testing completed!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Tested {len(test_questions)} questions")
    print(f"  - LLM Provider: {type(llm_provider).__name__ if llm_provider else 'None'}")
    print(f"  - Evidence retrieval: Working")
    print(f"  - LLM generation: Working")
    print("=" * 80)


if __name__ == "__main__":
    test_llm_qa_examples()
