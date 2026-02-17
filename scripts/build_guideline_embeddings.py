"""
Generate embeddings for RAG knowledge document chunks and add to vector store.
"""
from pathlib import Path
from typing import List

import numpy as np

from src.config import PATHS
from src.vector_store.faiss_store import FaissVectorStore, VectorMetadata
from scripts.ingest_guidelines import ingest_rag_guidelines
# Removed load_vector_store import - we create a separate store for guidelines

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError, AttributeError) as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"Warning: sentence-transformers not available ({type(e).__name__}). Using TF-IDF-based embeddings.")


def embed_guideline_chunks_tfidf(chunks: List[str]) -> np.ndarray:
    """
    Fallback: Generate embeddings using TF-IDF + PCA for dimensionality reduction.
    This avoids dependency on sentence-transformers/torchvision compatibility issues.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    
    print("Using TF-IDF + SVD for text embeddings (fallback method)...")
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(chunks)
    
    # Reduce to 384 dimensions (matching sentence-transformers output size)
    # But limit to min(384, num_features) to avoid errors
    n_components = min(384, tfidf_matrix.shape[1], len(chunks))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)
    
    # If we got fewer than 384 dimensions, pad with zeros
    if embeddings.shape[1] < 384:
        padding = np.zeros((embeddings.shape[0], 384 - embeddings.shape[1]), dtype="float32")
        embeddings = np.hstack([embeddings, padding])
    
    print(f"Generated {len(chunks)} embeddings (dim={embeddings.shape[1]}) using TF-IDF+SVD")
    return embeddings.astype("float32")


def embed_guideline_chunks(chunks: List[str]) -> np.ndarray:
    """
    Generate embeddings for guideline text chunks.
    Tries sentence-transformers first, falls back to TF-IDF if unavailable.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return embed_guideline_chunks_tfidf(chunks)
    
    try:
        print("Loading sentence transformer model for text embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, 384-dim embeddings
        
        print(f"Generating embeddings for {len(chunks)} guideline chunks...")
        embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        
        return embeddings.astype("float32")
    except Exception as e:
        print(f"Warning: sentence-transformers failed ({type(e).__name__}: {e})")
        print("Falling back to TF-IDF method...")
        return embed_guideline_chunks_tfidf(chunks)


def create_guideline_vector_store(chunks: List[str], embeddings: np.ndarray) -> FaissVectorStore:
    """
    Create a separate vector store for guideline chunks.
    Note: Guidelines use 384-dim embeddings (sentence-transformers),
    while patient samples use 144-dim fused embeddings.
    We'll search both stores separately in RAG.
    """
    dim = embeddings.shape[1]
    store = FaissVectorStore(dim=dim)
    
    metas = []
    for i, chunk in enumerate(chunks):
        # Extract a short preview for metadata
        preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
        
        metas.append(
            VectorMetadata(
                sample_id=f"guideline_chunk_{i}",
                source_dataset="RAG_Knowledge_Document",
                modality="guideline",
                metadata={
                    "chunk_index": i,
                    "text_preview": preview,
                    "chunk_length": len(chunk),
                    "full_text": chunk,  # Store full text for RAG context
                }
            )
        )
    
    store.add(embeddings, metas)
    print(f"Created guideline vector store with {len(chunks)} chunks (dim={dim})")
    return store


def main() -> None:
    """
    Main function to build guideline embeddings and save as separate vector store.
    """
    print("=" * 60)
    print("Building Guideline Embeddings for RAG")
    print("=" * 60)
    
    # Ingest and chunk guidelines
    print("\n1. Ingesting RAG knowledge document...")
    chunks = ingest_rag_guidelines()
    
    # Generate embeddings
    print("\n2. Generating text embeddings...")
    embeddings = embed_guideline_chunks(chunks)
    
    # Create guideline vector store
    print("\n3. Creating guideline vector store...")
    guideline_store = create_guideline_vector_store(chunks, embeddings)
    
    # Save guideline vectors and metadata
    print("\n4. Saving guideline vector store...")
    out_dir = PATHS.vector_store_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "guideline_vectors.npy", embeddings)
    
    # Save guideline metadata
    import json
    with (out_dir / "guideline_metadata.jsonl").open("w", encoding="utf-8") as f:
        for m in guideline_store.metadata:
            f.write(
                json.dumps(
                    {
                        "sample_id": m.sample_id,
                        "source_dataset": m.source_dataset,
                        "modality": m.modality,
                        "metadata": m.metadata,
                    }
                )
                + "\n"
            )
    
    print(f"\n[SUCCESS]")
    print(f"   Guideline chunks: {len(chunks)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Saved to {out_dir}")
    print(f"\n   Files created:")
    print(f"   - guideline_vectors.npy")
    print(f"   - guideline_metadata.jsonl")


if __name__ == "__main__":
    main()

