"""
Verification script to check if all components are correctly connected.
"""
from pathlib import Path
import numpy as np

from src.config import PATHS
from scripts.load_vector_store import load_vector_store
from scripts.load_guideline_store import load_guideline_store


def verify_data_pipeline():
    """Verify data ingestion and harmonization."""
    print("="*60)
    print("1. VERIFYING DATA PIPELINE")
    print("="*60)
    
    from scripts.ingest_resp_sounds import build_resp_sound_samples
    from scripts.ingest_nhanes_spirometry import build_nhanes_samples
    
    resp_samples = build_resp_sound_samples()
    nhanes_samples = build_nhanes_samples()
    
    print(f"   Respiratory Sound Database: {len(resp_samples)} samples")
    print(f"   NHANES Spirometry: {len(nhanes_samples)} samples")
    print(f"   Total: {len(resp_samples) + len(nhanes_samples)} samples")
    
    # Check sample structure
    if resp_samples:
        sample = resp_samples[0]
        print(f"\n   Sample structure check:")
        print(f"   - Has audio: {sample.cough_audio is not None}")
        print(f"   - Has spirometry: {sample.spirometry is not None}")
        print(f"   - Has symptoms: {len(sample.symptoms)} symptoms")
        print(f"   - Has diagnosis: {sample.diagnosis is not None}")
    
    print("   [OK] Data pipeline verified")
    return True


def verify_vector_stores():
    """Verify vector stores are built and accessible."""
    print("\n" + "="*60)
    print("2. VERIFYING VECTOR STORES")
    print("="*60)
    
    # Check patient vector store
    try:
        store = load_vector_store()
        print(f"   Patient vector store: {len(store.metadata)} vectors, dim={store.dim}")
        
        # Check files exist
        vectors_path = PATHS.vector_store_dir / "fused_vectors.npy"
        metadata_path = PATHS.vector_store_dir / "metadata.jsonl"
        
        if vectors_path.exists() and metadata_path.exists():
            vectors = np.load(vectors_path)
            print(f"   - Vectors file: {vectors.shape}")
            print(f"   - Metadata file: exists")
        else:
            print(f"   [WARNING] Vector files missing")
            return False
    except Exception as e:
        print(f"   [ERROR] Failed to load patient vector store: {e}")
        return False
    
    # Check guideline vector store
    try:
        guideline_store = load_guideline_store()
        print(f"   Guideline vector store: {len(guideline_store.metadata)} chunks, dim={guideline_store.dim}")
        
        guideline_vectors_path = PATHS.vector_store_dir / "guideline_vectors.npy"
        guideline_metadata_path = PATHS.vector_store_dir / "guideline_metadata.jsonl"
        
        if guideline_vectors_path.exists() and guideline_metadata_path.exists():
            guideline_vectors = np.load(guideline_vectors_path)
            print(f"   - Guideline vectors file: {guideline_vectors.shape}")
            print(f"   - Guideline metadata file: exists")
        else:
            print(f"   [WARNING] Guideline files missing")
            return False
    except Exception as e:
        print(f"   [WARNING] Guideline store not found: {e}")
        print(f"   Run: python -m scripts.build_guideline_embeddings")
        return False
    
    print("   [OK] Vector stores verified")
    return True


def verify_knowledge_graph():
    """Verify knowledge graph is built."""
    print("\n" + "="*60)
    print("3. VERIFYING KNOWLEDGE GRAPH")
    print("="*60)
    
    kg_path = PATHS.knowledge_graph_dir / "respiratory_kg.pkl"
    
    if not kg_path.exists():
        print(f"   [ERROR] Knowledge graph not found at {kg_path}")
        print(f"   Run: python -m scripts.build_knowledge_graph")
        return False
    
    try:
        import pickle
        import networkx as nx
        
        with kg_path.open("rb") as f:
            g = pickle.load(f)
        
        print(f"   Knowledge graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        
        # Check node types
        node_types = {}
        for node, data in g.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print(f"   Node types: {dict(node_types)}")
        
    except Exception as e:
        print(f"   [ERROR] Failed to load knowledge graph: {e}")
        return False
    
    print("   [OK] Knowledge graph verified")
    return True


def verify_rag_connection():
    """Verify RAG can retrieve from both stores."""
    print("\n" + "="*60)
    print("4. VERIFYING RAG CONNECTION")
    print("="*60)
    
    try:
        from src.rag.engine import EvidenceFirstRAG
        from scripts.load_vector_store import load_vector_store
        from scripts.load_guideline_store import load_guideline_store
        
        patient_store = load_vector_store()
        guideline_store = load_guideline_store()
        
        rag = EvidenceFirstRAG(patient_store)
        
        # Test retrieval
        query_embedding = np.random.randn(patient_store.dim).astype("float32")
        evidences = rag.retrieve_similar_samples(query_embedding, k=5)
        
        print(f"   Retrieved {len(evidences)} patient samples")
        
        # Test with guidelines
        from scripts.ingest_guidelines import ingest_rag_guidelines
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        chunks = ingest_rag_guidelines()
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        chunks_tfidf = vectorizer.fit_transform(chunks)
        n_components = min(384, chunks_tfidf.shape[1], len(chunks))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(chunks_tfidf)
        
        query_text = "COPD progression"
        query_tfidf = vectorizer.transform([query_text])
        guideline_query = svd.transform(query_tfidf)[0]
        if guideline_query.shape[0] < 384:
            padding = np.zeros(384 - guideline_query.shape[0], dtype="float32")
            guideline_query = np.concatenate([guideline_query, padding])
        
        evidences_with_guidelines = rag.retrieve_similar_samples(
            query_embedding, k=5,
            guideline_store=guideline_store,
            guideline_query_embedding=guideline_query.astype("float32"),
            k_guidelines=2
        )
        
        patient_count = sum(1 for e in evidences_with_guidelines if e.modality != "guideline")
        guideline_count = sum(1 for e in evidences_with_guidelines if e.modality == "guideline")
        
        print(f"   Retrieved {patient_count} patients + {guideline_count} guidelines")
        
    except Exception as e:
        print(f"   [ERROR] RAG connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("   [OK] RAG connection verified")
    return True


def verify_trained_models():
    """Verify trained models exist."""
    print("\n" + "="*60)
    print("5. VERIFYING TRAINED MODELS")
    print("="*60)
    
    models_dir = PATHS.project_root / "artifacts" / "models" / "predictors"
    
    if not models_dir.exists():
        print(f"   [WARNING] Models directory not found")
        print(f"   Run: python -m scripts.train_predictors")
        return False
    
    required_models = ['copd_risk_model.pkl', 'asthma_prob_model.pkl', 
                       'viral_vs_nonviral_model.pkl', 'hospitalization_risk_model.pkl']
    
    missing = []
    for model_file in required_models:
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"   [OK] {model_file}")
        else:
            print(f"   [MISSING] {model_file}")
            missing.append(model_file)
    
    if missing:
        print(f"\n   [WARNING] Missing models: {missing}")
        print(f"   Run: python -m scripts.train_predictors")
        return False
    
    print("   [OK] All trained models verified")
    return True


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("SYSTEM VERIFICATION")
    print("="*60)
    print()
    
    checks = [
        ("Data Pipeline", verify_data_pipeline),
        ("Vector Stores", verify_vector_stores),
        ("Knowledge Graph", verify_knowledge_graph),
        ("RAG Connection", verify_rag_connection),
        ("Trained Models", verify_trained_models),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n   [ERROR] {name} check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"   {status} {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n[SUCCESS] All systems verified!")
    else:
        print("\n[WARNING] Some checks failed. See above for details.")
    
    return all_passed


if __name__ == "__main__":
    main()
