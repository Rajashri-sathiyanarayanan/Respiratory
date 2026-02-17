"""
Utility script to load the guideline vector store.
"""
from pathlib import Path
import json
import numpy as np

from src.config import PATHS
from src.vector_store.faiss_store import FaissVectorStore, VectorMetadata


def load_guideline_store() -> FaissVectorStore:
    """
    Load the guideline vector store from saved artifacts.
    """
    store_dir = PATHS.vector_store_dir
    
    # Load vectors
    vectors_path = store_dir / "guideline_vectors.npy"
    if not vectors_path.exists():
        raise FileNotFoundError(
            f"Guideline vectors not found at {vectors_path}. "
            "Run build_guideline_embeddings first."
        )
    
    vectors = np.load(vectors_path)
    dim = vectors.shape[1]
    
    # Load metadata
    metadata_path = store_dir / "guideline_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Guideline metadata not found at {metadata_path}. "
            "Run build_guideline_embeddings first."
        )
    
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
    
    # Rebuild FAISS index
    store = FaissVectorStore(dim=dim)
    store.add(vectors, metas)
    
    print(f"Loaded guideline store with {len(metas)} chunks (dim={dim})")
    return store


if __name__ == "__main__":
    store = load_guideline_store()
    print(f"Guideline store ready. Chunk count: {len(store.metadata)}")

