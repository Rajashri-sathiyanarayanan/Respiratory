"""
Test RAG system with the new RAG knowledge document.
"""
import numpy as np

from src.rag.engine import EvidenceFirstRAG
from src.config import PATHS
from scripts.load_vector_store import load_vector_store
from scripts.load_guideline_store import load_guideline_store

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError, AttributeError):
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Fallback: use TF-IDF for query embedding
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD


def test_rag_with_guidelines():
    """
    Test RAG retrieval including both patient samples and guideline chunks.
    """
    print("=" * 60)
    print("Testing RAG System with RAG Knowledge Document")
    print("=" * 60)
    
    print("\n1. Loading vector store (patients + guidelines)...")
    store = load_vector_store(use_combined=True)
    
    print("\n2. Loading guideline vector store...")
    try:
        guideline_store = load_guideline_store()
        has_guidelines = True
    except FileNotFoundError:
        print("   [WARNING] Guideline store not found. Run build_guideline_embeddings first.")
        has_guidelines = False
        guideline_store = None
    
    print("\n3. Initializing RAG engine...")
    # Prefer HuggingFace (e.g., BioMistral) for medical reasoning if configured
    rag = EvidenceFirstRAG(store, llm_provider="huggingface")
    
    # Create query embeddings
    patient_query_dim = store.dim  # 144-dim for fused patient embeddings
    patient_query_embedding = np.random.randn(patient_query_dim).astype("float32")
    
    # Create guideline query embedding if available
    if has_guidelines:
        print("\n4. Generating guideline query embedding...")
        query_text = "What is the risk of COPD progression based on spirometry and symptoms?"
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                guideline_query_embedding = model.encode([query_text], convert_to_numpy=True)[0].astype("float32")
            except Exception:
                print("   Warning: sentence-transformers failed, using TF-IDF fallback...")
                guideline_query_embedding = None
        else:
            # Load guideline vectors to get dimension, then use TF-IDF
            guideline_vectors_path = PATHS.vector_store_dir / "guideline_vectors.npy"
            if guideline_vectors_path.exists():
                guideline_dim = np.load(guideline_vectors_path).shape[1]
                # Load the actual guideline chunks to fit TF-IDF properly
                from scripts.ingest_guidelines import ingest_rag_guidelines
                guideline_chunks = ingest_rag_guidelines()
                
                # Use same TF-IDF settings as in build_guideline_embeddings
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
                # Fit on actual guideline chunks
                chunks_tfidf = vectorizer.fit_transform(guideline_chunks)
                
                # Fit SVD with proper n_components
                n_components = min(384, chunks_tfidf.shape[1], len(guideline_chunks))
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                svd.fit(chunks_tfidf)
                
                # Transform query
                query_tfidf = vectorizer.transform([query_text])
                query_embedding = svd.transform(query_tfidf)[0]
                
                # Pad to 384 if needed
                if query_embedding.shape[0] < guideline_dim:
                    padding = np.zeros(guideline_dim - query_embedding.shape[0], dtype="float32")
                    guideline_query_embedding = np.concatenate([query_embedding, padding]).astype("float32")
                else:
                    guideline_query_embedding = query_embedding.astype("float32")
                
                print(f"   Generated TF-IDF query embedding (dim={guideline_query_embedding.shape[0]})")
            else:
                guideline_query_embedding = None
    else:
        guideline_query_embedding = None
    
    print(f"\n5. Testing retrieval with query embeddings...")
    print(f"   Patient query dim: {patient_query_dim}")
    if has_guidelines:
        print(f"   Guideline query dim: {guideline_query_embedding.shape[0] if guideline_query_embedding is not None else 'N/A'}")
    print("   Retrieving top 10 patient samples + 3 guideline chunks...")
    
    evidences = rag.retrieve_similar_samples(
        patient_query_embedding, 
        k=10,
        guideline_store=guideline_store,
        guideline_query_embedding=guideline_query_embedding,
        k_guidelines=3
    )
    
    print(f"\n[SUCCESS] Retrieved {len(evidences)} items:")
    print("-" * 60)
    
    patient_count = 0
    guideline_count = 0
    
    for i, ev in enumerate(evidences, 1):
        print(f"\n  [{i}] {ev.modality.upper()}")
        print(f"      Sample ID: {ev.sample_id}")
        print(f"      Dataset: {ev.source_dataset}")
        print(f"      Similarity: {1 - ev.distance:.3f}")
        
        if ev.modality == "guideline":
            guideline_count += 1
            if ev.metadata and "text_preview" in ev.metadata:
                # Handle Unicode encoding for Windows console
                preview = ev.metadata['text_preview']
                try:
                    preview = preview.encode('ascii', 'ignore').decode('ascii')
                except:
                    pass
                print(f"      Preview: {preview}")
        else:
            patient_count += 1
            if ev.metadata:
                if "diagnosis" in ev.metadata and ev.metadata["diagnosis"]:
                    print(f"      Diagnosis: {ev.metadata['diagnosis']}")
                if "fev1" in ev.metadata and ev.metadata["fev1"]:
                    print(f"      FEV1: {ev.metadata['fev1']:.2f}")
    
    print("\n" + "-" * 60)
    print(f"Summary: {patient_count} patient samples, {guideline_count} guideline chunks")
    
    # Test explanation generation with mixed evidence
    print("\n" + "=" * 60)
    print("6. Testing explanation generation with retrieved evidence...")
    print("=" * 60)
    
    question = "What is the risk of COPD progression for this patient based on their spirometry and symptoms?"
    explanation = rag.generate_explanation(question, evidences)
    
    print(f"\nQuestion: {explanation['question']}")
    print(f"Number of evidence items: {explanation['num_evidence']}")
    print(f"LLM Model: {explanation['llm_model']}")
    print(f"\nExplanation:\n{explanation['explanation']}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] RAG system test completed!")
    print("\nNext steps:")
    print("- If you have OpenAI API key, set OPENAI_API_KEY for real LLM explanations")
    print("- Train prediction models (scripts/train_predictors.py)")
    print("- Build API endpoints (src/api/main.py)")


if __name__ == "__main__":
    test_rag_with_guidelines()

