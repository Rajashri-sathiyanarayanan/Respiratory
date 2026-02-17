"""
Utility script to load the saved vector store from disk.
"""
from pathlib import Path
import json
import numpy as np
import faiss

from src.config import PATHS
from src.vector_store.faiss_store import FaissVectorStore, VectorMetadata


def load_vector_store(use_combined: bool = True) -> FaissVectorStore:
    """
    Load the FAISS vector store from saved artifacts.
    
    Args:
        use_combined: If True, load combined vectors (patients + guidelines).
                     If False, load only patient sample vectors.
    """
    store_dir = PATHS.vector_store_dir
    
    # Try to load combined vectors first (if guidelines were added)
    if use_combined:
        vectors_path = store_dir / "combined_vectors.npy"
        if not vectors_path.exists():
            vectors_path = store_dir / "fused_vectors.npy"
    else:
        vectors_path = store_dir / "fused_vectors.npy"
    
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors not found at {vectors_path}. Run generate_embeddings first.")
    
    vectors = np.load(vectors_path)
    dim = vectors.shape[1]
    
    # Load metadata
    metadata_path = store_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run generate_embeddings first.")
    
    metas = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            metas.append(
                VectorMetadata(
                    sample_id=data["sample_id"],
                    source_dataset=data["source_dataset"],
                    modality=data["modality"],
                    metadata=data.get("metadata"),
                )
            )
    
    # If using combined vectors, we need to rebuild the full store
    # For now, we'll need to reload guidelines separately if they exist
    # This is a limitation - we should save the full store state
    # For now, just use what we have
    
    # Rebuild FAISS index
    store = FaissVectorStore(dim=dim)
    
    # Only add vectors that have corresponding metadata
    # (metadata.jsonl might not include guidelines if they were added separately)
    num_vectors = min(vectors.shape[0], len(metas))
    if num_vectors < vectors.shape[0]:
        print(f"Warning: {vectors.shape[0]} vectors but only {len(metas)} metadata entries. Using first {num_vectors}.")
        vectors = vectors[:num_vectors]
        metas = metas[:num_vectors]
    
    store.add(vectors, metas)
    
    print(f"Loaded vector store with {len(metas)} vectors (dim={dim})")
    return store


if __name__ == "__main__":
    store = load_vector_store()
    print(f"Vector store ready. Sample metadata count: {len(store.metadata)}")




