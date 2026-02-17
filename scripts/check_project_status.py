"""
Quick project status check - no LLM/network required.
"""
from pathlib import Path
import sys

# Project root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

def main():
    print("=" * 60)
    print("PROJECT STATUS CHECK")
    print("=" * 60)
    ok = 0
    fail = 0

    # 1. Config / paths
    try:
        from src.config import PATHS
        kg_path = PATHS.knowledge_graph_dir / "respiratory_kg.pkl"
        vs_dir = PATHS.vector_store_dir
        print("\n1. Paths: OK")
        ok += 1
    except Exception as e:
        print(f"\n1. Paths: FAIL - {e}")
        fail += 1
        return

    # 2. Knowledge graph
    try:
        if kg_path.exists():
            import pickle
            with kg_path.open("rb") as f:
                g = pickle.load(f)
            print(f"   Knowledge graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges - OK")
            ok += 1
        else:
            print("   Knowledge graph: file not found - FAIL")
            fail += 1
    except Exception as e:
        print(f"   Knowledge graph: FAIL - {e}")
        fail += 1

    # 3. Vector stores
    try:
        from scripts.load_vector_store import load_vector_store
        from scripts.load_guideline_store import load_guideline_store
        ps = load_vector_store()
        gs = load_guideline_store()
        print(f"\n2. Vector stores: OK")
        print(f"   Patient: {len(ps.metadata)} samples, dim={ps.dim}")
        print(f"   Guideline: {len(gs.metadata)} chunks, dim={gs.dim}")
        ok += 1
    except Exception as e:
        print(f"\n2. Vector stores: FAIL - {e}")
        fail += 1

    # 4. RAG retrieval (no LLM)
    try:
        import numpy as np
        from src.rag.engine import EvidenceFirstRAG
        rag = EvidenceFirstRAG(ps)
        q = np.random.randn(ps.dim).astype("float32")
        ev = rag.retrieve_similar_samples(q, k=3)
        print(f"\n3. RAG retrieval: OK (retrieved {len(ev)} samples)")
        ok += 1
    except Exception as e:
        print(f"\n3. RAG retrieval: FAIL - {e}")
        fail += 1

    # 5. API import
    try:
        from src.api.main import app
        print(f"\n4. API (FastAPI app): OK")
        ok += 1
    except Exception as e:
        print(f"\n4. API: FAIL - {e}")
        fail += 1

    print("\n" + "=" * 60)
    print(f"RESULT: {ok} passed, {fail} failed")
    print("=" * 60)
    if fail == 0:
        print("Project status: READY (local pipeline working)")
        print("For LLM: run python -m scripts.test_llm_connection")
        print("For full test: run python -m scripts.test_llm_predictions")
    else:
        print("Fix failed checks above.")
    return fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
