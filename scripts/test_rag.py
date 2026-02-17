"""
Test script to verify RAG retrieval and explanation generation.
"""
import numpy as np

from src.rag.engine import EvidenceFirstRAG
from scripts.load_vector_store import load_vector_store


def test_rag_retrieval():
    """
    Test basic RAG retrieval functionality.
    """
    print("Loading vector store...")
    store = load_vector_store()
    
    print("\nInitializing RAG engine...")
    # Use HuggingFace provider (e.g., BioMistral) for testing, will fallback if not configured
    rag = EvidenceFirstRAG(store, llm_provider="huggingface")
    
    # Create a dummy query embedding (same dimension as stored vectors)
    query_dim = store.dim
    query_embedding = np.random.randn(query_dim).astype("float32")
    
    print(f"\nTesting retrieval with query embedding (dim={query_dim})...")
    evidences = rag.retrieve_similar_samples(query_embedding, k=5)
    
    print(f"\nRetrieved {len(evidences)} similar samples:")
    for i, ev in enumerate(evidences, 1):
        print(f"\n  Evidence {i}:")
        print(f"    Sample ID: {ev.sample_id}")
        print(f"    Dataset: {ev.source_dataset}")
        print(f"    Modality: {ev.modality}")
        print(f"    Similarity: {1 - ev.distance:.3f}")
        if ev.metadata:
            if "diagnosis" in ev.metadata and ev.metadata["diagnosis"]:
                print(f"    Diagnosis: {ev.metadata['diagnosis']}")
            if "fev1" in ev.metadata and ev.metadata["fev1"]:
                print(f"    FEV1: {ev.metadata['fev1']:.2f}")
    
    # Test explanation generation
    print("\n" + "="*60)
    print("Testing explanation generation...")
    question = "What is the risk of COPD progression for this patient?"
    explanation = rag.generate_explanation(question, evidences)
    
    print(f"\nQuestion: {explanation['question']}")
    print(f"Number of evidence items: {explanation['num_evidence']}")
    print(f"LLM Model: {explanation['llm_model']}")
    print(f"\nExplanation:\n{explanation['explanation']}")
    print(f"\nEvidence IDs: {explanation['evidence_ids'][:3]}...")  # Show first 3
    
    print("\n" + "="*60)
    print("✅ RAG system test completed successfully!")
    print("\nNext steps:")
    print("1. If you have OpenAI API key, set OPENAI_API_KEY env var for LLM explanations")
    print("2. Train prediction models (scripts/train_predictors.py - to be created)")
    print("3. Build API endpoints (src/api/main.py - to be updated)")


if __name__ == "__main__":
    test_rag_retrieval()




