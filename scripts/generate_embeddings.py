from pathlib import Path
from typing import List

import numpy as np

from src.config import PATHS
from src.embeddings.audio_simple import audio_embedding
from src.embeddings.tabular_simple import tabular_embedding
from src.ontology.schema import PatientSample
from src.vector_store.faiss_store import FaissVectorStore, VectorMetadata
from scripts.ingest_resp_sounds import build_resp_sound_samples
from scripts.ingest_nhanes_spirometry import build_nhanes_samples


def build_all_samples() -> List[PatientSample]:
    """
    Build unified PatientSample list from all ingestors.

    For now we just concatenate Respiratory Sound DB + NHANES.
    """
    resp_samples = build_resp_sound_samples()
    nhanes_samples = build_nhanes_samples()
    return resp_samples + nhanes_samples


def build_embeddings_and_store() -> None:
    """
    Generate simple multi-modal embeddings and populate a FAISS vector store.
    """
    samples = build_all_samples()
    print(f"Total unified samples: {len(samples)}")

    # Define embedding dimensions
    audio_dim = 128
    tab_dim = 16
    fused_dim = audio_dim + tab_dim

    store = FaissVectorStore(dim=fused_dim)

    vectors: List[np.ndarray] = []
    metas: List[VectorMetadata] = []

    for sample in samples:
        # Audio embedding (if available)
        if sample.cough_audio:
            a_emb = audio_embedding(sample.cough_audio, target_dim=audio_dim)
        else:
            a_emb = np.zeros(audio_dim, dtype="float32")

        # Tabular/spirometry embedding
        t_emb = tabular_embedding(sample, target_dim=tab_dim)

        fused = np.concatenate([a_emb, t_emb], axis=0).astype("float32")

        meta_dict = {
            "symptoms": [s.code for s in sample.symptoms],
            "diagnosis": sample.diagnosis.label if sample.diagnosis else None,
            "fev1": sample.spirometry.fev1 if sample.spirometry else None,
            "fvc": sample.spirometry.fvc if sample.spirometry else None,
            "fev1_fvc_ratio": sample.spirometry.fev1_fvc_ratio if sample.spirometry else None,
            "clinical_state": sample.clinical_state.state_label if sample.clinical_state else None,
        }

        vectors.append(fused)
        metas.append(
            VectorMetadata(
                sample_id=sample.sample_id,
                source_dataset=sample.source_dataset,
                modality="fused",
                metadata=meta_dict,
            )
        )

    mat = np.stack(vectors, axis=0)
    store.add(mat, metas)

    # Persist vectors and metadata for later use (optional simple npy save)
    out_dir = PATHS.vector_store_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "fused_vectors.npy", mat)

    # Save lightweight metadata as a JSON lines file for inspection
    import json

    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
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

    print(f"Saved fused embeddings to {out_dir}")


if __name__ == "__main__":
    build_embeddings_and_store()








