"""
Display detailed knowledge graph statistics for testing and documentation.
"""
from pathlib import Path
import pickle
import networkx as nx
from collections import Counter

from src.config import PATHS


def show_kg_stats():
    """Display comprehensive knowledge graph statistics."""
    print("=" * 80)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("=" * 80)
    
    kg_path = PATHS.knowledge_graph_dir / "respiratory_kg.pkl"
    
    if not kg_path.exists():
        print(f"\n[ERROR] Knowledge graph not found at {kg_path}")
        print(f"Run: python -m scripts.build_knowledge_graph")
        return
    
    try:
        with kg_path.open("rb") as f:
            g = pickle.load(f)
        
        # Basic statistics
        print(f"\n1. BASIC STATISTICS")
        print("-" * 80)
        print(f"   Total Nodes: {g.number_of_nodes():,}")
        print(f"   Total Edges: {g.number_of_edges():,}")
        print(f"   Graph Type: {type(g).__name__}")
        
        # Node type breakdown
        print(f"\n2. NODE TYPE BREAKDOWN")
        print("-" * 80)
        node_types = Counter()
        node_details = {}
        
        for node, data in g.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] += 1
            
            if node_type not in node_details:
                node_details[node_type] = {
                    'count': 0,
                    'sample_attrs': []
                }
            node_details[node_type]['count'] += 1
            
            # Collect sample attributes for PatientSample nodes
            if node_type == 'PatientSample' and len(node_details[node_type]['sample_attrs']) < 5:
                attrs = {
                    'source_dataset': data.get('source_dataset', 'N/A'),
                    'diagnosis': data.get('diagnosis', 'N/A')
                }
                node_details[node_type]['sample_attrs'].append(attrs)
        
        for node_type, count in node_types.most_common():
            print(f"   {node_type:25s}: {count:6,} nodes")
            if node_type == 'PatientSample' and node_details[node_type]['sample_attrs']:
                print(f"   {'Sample attributes (first 3):':>27s}")
                for i, attrs in enumerate(node_details[node_type]['sample_attrs'][:3], 1):
                    print(f"   {'':>27s}  [{i}] Dataset: {attrs['source_dataset']}, Diagnosis: {attrs['diagnosis']}")
        
        # Edge type breakdown
        print(f"\n3. EDGE TYPE BREAKDOWN")
        print("-" * 80)
        edge_types = Counter()
        for u, v, data in g.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] += 1
        
        for edge_type, count in edge_types.most_common():
            print(f"   {edge_type:25s}: {count:6,} edges")
        
        # Sample node examples
        print(f"\n4. SAMPLE NODES (First 5 PatientSample nodes)")
        print("-" * 80)
        patient_samples = [
            (node, data) for node, data in g.nodes(data=True)
            if data.get('type') == 'PatientSample'
        ][:5]
        
        for i, (node, data) in enumerate(patient_samples, 1):
            print(f"\n   [{i}] Node: {node}")
            print(f"       Dataset: {data.get('source_dataset', 'N/A')}")
            print(f"       Diagnosis: {data.get('diagnosis', 'N/A')}")
            
            # Find connected nodes
            neighbors = list(g.neighbors(node))
            print(f"       Connected to: {len(neighbors)} nodes")
            
            # Show edge types
            edge_types_for_node = Counter()
            for neighbor in neighbors:
                for edge_data in g.get_edge_data(node, neighbor).values():
                    edge_type = edge_data.get('type', 'unknown')
                    edge_types_for_node[edge_type] += 1
            
            if edge_types_for_node:
                print(f"       Edge types: {dict(edge_types_for_node)}")
        
        # Graph connectivity
        print(f"\n5. GRAPH CONNECTIVITY")
        print("-" * 80)
        if nx.is_connected(g.to_undirected()):
            print("   Graph is connected (all nodes reachable)")
        else:
            components = list(nx.connected_components(g.to_undirected()))
            print(f"   Graph has {len(components)} connected components")
            print(f"   Largest component: {len(max(components, key=len))} nodes")
        
        # Degree statistics
        print(f"\n6. DEGREE STATISTICS")
        print("-" * 80)
        degrees = dict(g.degree())
        if degrees:
            degree_values = list(degrees.values())
            print(f"   Average degree: {sum(degree_values) / len(degree_values):.2f}")
            print(f"   Max degree: {max(degree_values)}")
            print(f"   Min degree: {min(degree_values)}")
        
        # Summary
        print(f"\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"   Knowledge graph successfully loaded from: {kg_path}")
        print(f"   Total entities (nodes): {g.number_of_nodes():,}")
        print(f"   Total relationships (edges): {g.number_of_edges():,}")
        print(f"   Node types: {len(node_types)} distinct types")
        print(f"   Edge types: {len(edge_types)} distinct types")
        print(f"\n   [SUCCESS] Knowledge graph statistics displayed")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load knowledge graph: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    show_kg_stats()
