from pathlib import Path
import pickle

import networkx as nx

from src.config import PATHS
from src.knowledge_graph.builder import build_knowledge_graph
from scripts.ingest_resp_sounds import build_resp_sound_samples
from scripts.ingest_nhanes_spirometry import build_nhanes_samples


def main() -> None:
    resp_samples = build_resp_sound_samples()
    nhanes_samples = build_nhanes_samples()
    samples = resp_samples + nhanes_samples

    g = build_knowledge_graph(samples)

    out_dir = PATHS.knowledge_graph_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "respiratory_kg.pkl"
    # Use standard pickle to persist the NetworkX graph (works across versions)
    with out_path.open("wb") as f:
        pickle.dump(g, f)

    print(f"Knowledge graph built with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()


