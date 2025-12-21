"""
HippoRAG Knowledge Graph Visualization Script
Visualize the knowledge graph created by HippoRAG
"""

import pickle
import json
import os
from collections import defaultdict

def load_graph(save_dir='outputs', llm_model='gpt-4o', embedding_model='text-embedding-3-large'):
    """Load the igraph object from pickle file."""
    graph_path = os.path.join(save_dir, f"{llm_model}_{embedding_model}", "graph.pickle")

    if not os.path.exists(graph_path):
        print(f"Graph not found at: {graph_path}")
        print("Please run indexing first with test_hipporag.py")
        return None

    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    return graph

def load_openie_results(save_dir='outputs', llm_model='gpt-4o'):
    """Load OpenIE extraction results (entities and triples)."""
    openie_path = os.path.join(save_dir, f"openie_results_ner_{llm_model}.json")

    if not os.path.exists(openie_path):
        print(f"OpenIE results not found at: {openie_path}")
        return None

    with open(openie_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results

def print_graph_stats(graph):
    """Print basic statistics about the graph."""
    print("\n" + "="*60)
    print("  KNOWLEDGE GRAPH STATISTICS")
    print("="*60)

    print(f"\nTotal Nodes: {graph.vcount()}")
    print(f"Total Edges: {graph.ecount()}")

    # Count node types
    if 'type' in graph.vs.attributes():
        node_types = defaultdict(int)
        for v in graph.vs:
            node_types[v['type']] += 1

        print("\nNode Types:")
        for ntype, count in node_types.items():
            print(f"  - {ntype}: {count}")

    # Count edge types
    if 'type' in graph.es.attributes():
        edge_types = defaultdict(int)
        for e in graph.es:
            edge_types[e['type']] += 1

        print("\nEdge Types:")
        for etype, count in edge_types.items():
            print(f"  - {etype}: {count}")

def print_entities_and_triples(openie_results):
    """Print extracted entities and triples."""
    if not openie_results:
        return

    print("\n" + "="*60)
    print("  EXTRACTED KNOWLEDGE (OpenIE Results)")
    print("="*60)

    # Handle the actual structure: {'docs': [...], 'avg_ent_chars': ..., 'avg_ent_words': ...}
    if isinstance(openie_results, dict):
        docs_list = openie_results.get('docs', [])
        print(f"\nTotal documents processed: {len(docs_list)}")
        print(f"Avg entity chars: {openie_results.get('avg_ent_chars', 'N/A')}")
        print(f"Avg entity words: {openie_results.get('avg_ent_words', 'N/A')}")
    else:
        docs_list = openie_results

    for i, doc_result in enumerate(docs_list[:5]):  # Show first 5 documents
        print(f"\n--- Document {i+1} ---")

        # Show passage preview
        if 'passage' in doc_result:
            passage = doc_result['passage'][:150].replace('\n', ' ')
            print(f"Passage: {passage}...")

        # Show extracted entities
        if 'extracted_entities' in doc_result:
            entities = doc_result['extracted_entities']
            print(f"Entities ({len(entities)}): {entities[:8]}{'...' if len(entities) > 8 else ''}")

        # Show extracted triples
        if 'extracted_triples' in doc_result:
            triples = doc_result['extracted_triples']
            print(f"Triples ({len(triples)} total):")
            for triple in triples[:3]:  # Show first 3 triples
                if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                    print(f"  ({triple[0]}) --[{triple[1]}]--> ({triple[2]})")
                elif isinstance(triple, dict):
                    subj = triple.get('subject', triple.get('head', '?'))
                    rel = triple.get('relation', triple.get('predicate', '?'))
                    obj = triple.get('object', triple.get('tail', '?'))
                    print(f"  ({subj}) --[{rel}]--> ({obj})")
            if len(triples) > 3:
                print(f"  ... and {len(triples) - 3} more triples")

    if len(docs_list) > 5:
        print(f"\n... and {len(docs_list) - 5} more documents")

def visualize_with_igraph(graph, output_file='knowledge_graph.png'):
    """Visualize using igraph's built-in plotting."""
    try:
        import igraph as ig
        import matplotlib.pyplot as plt

        # Set up visual style
        visual_style = {}

        # Color nodes by type
        if 'type' in graph.vs.attributes():
            color_map = {
                'phrase': 'lightblue',
                'passage': 'orange',
                'entity': 'lightgreen',
                'document': 'pink'
            }
            visual_style['vertex_color'] = [
                color_map.get(v['type'], 'gray') for v in graph.vs
            ]

        # Label nodes (truncate long labels)
        if 'name' in graph.vs.attributes():
            visual_style['vertex_label'] = [
                v['name'][:20] + '...' if len(str(v['name'])) > 20 else v['name']
                for v in graph.vs
            ]

        visual_style['vertex_size'] = 20
        visual_style['vertex_label_size'] = 8
        visual_style['edge_arrow_size'] = 0.5
        visual_style['bbox'] = (1200, 800)
        visual_style['margin'] = 50

        # Use layout algorithm
        layout = graph.layout('fr')  # Fruchterman-Reingold layout

        # Plot
        fig, ax = plt.subplots(figsize=(15, 10))
        ig.plot(graph, target=ax, layout=layout, **visual_style)
        plt.title("HippoRAG Knowledge Graph", fontsize=16)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nGraph visualization saved to: {output_file}")
        return True

    except ImportError as e:
        print(f"\nCannot visualize with igraph: {e}")
        print("Install with: pip install python-igraph matplotlib cairocffi")
        return False

def export_to_networkx(graph):
    """Export igraph to NetworkX for alternative visualization."""
    try:
        import networkx as nx

        # Create NetworkX graph
        G = nx.DiGraph() if graph.is_directed() else nx.Graph()

        # Add nodes with attributes
        for v in graph.vs:
            attrs = {attr: v[attr] for attr in graph.vs.attributes() if v[attr] is not None}
            G.add_node(v.index, **attrs)

        # Add edges with attributes
        for e in graph.es:
            attrs = {attr: e[attr] for attr in graph.es.attributes() if e[attr] is not None}
            G.add_edge(e.source, e.target, **attrs)

        return G

    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")
        return None

def visualize_with_networkx(graph, output_file='knowledge_graph_nx.png'):
    """Visualize using NetworkX and matplotlib."""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = export_to_networkx(graph)
        if G is None:
            return False

        plt.figure(figsize=(15, 10))

        # Color nodes by type
        color_map = {
            'phrase': 'lightblue',
            'passage': 'orange',
            'entity': 'lightgreen',
            'document': 'pink'
        }

        node_colors = []
        for node in G.nodes():
            ntype = G.nodes[node].get('type', 'unknown')
            node_colors.append(color_map.get(ntype, 'gray'))

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw
        nx.draw(G, pos,
                node_color=node_colors,
                node_size=500,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7)

        plt.title("HippoRAG Knowledge Graph (NetworkX)", fontsize=16)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nNetworkX visualization saved to: {output_file}")
        return True

    except ImportError as e:
        print(f"\nCannot visualize with NetworkX: {e}")
        print("Install with: pip install networkx matplotlib")
        return False

def export_to_gephi(graph, output_file='knowledge_graph.gexf'):
    """Export to GEXF format for Gephi visualization."""
    try:
        import networkx as nx

        G = export_to_networkx(graph)
        if G is None:
            return False

        nx.write_gexf(G, output_file)
        print(f"\nGephi file saved to: {output_file}")
        print("Open this file in Gephi (https://gephi.org/) for interactive visualization")
        return True

    except Exception as e:
        print(f"\nCannot export to Gephi format: {e}")
        return False

def print_sample_nodes(graph, n=10):
    """Print sample nodes from the graph."""
    print("\n" + "="*60)
    print(f"  SAMPLE NODES (first {n})")
    print("="*60)

    for i, v in enumerate(graph.vs[:n]):
        print(f"\nNode {i}:")
        for attr in graph.vs.attributes():
            val = v[attr]
            if val is not None:
                # Truncate long values
                val_str = str(val)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"  {attr}: {val_str}")

def main():
    print("="*60)
    print("  HippoRAG Knowledge Graph Visualization")
    print("="*60)

    # Load graph
    print("\nLoading knowledge graph...")
    graph = load_graph()

    if graph is None:
        print("\nNo graph found. Please run test_hipporag.py first to create the knowledge graph.")
        return

    # Print statistics
    print_graph_stats(graph)

    # Print sample nodes
    print_sample_nodes(graph)

    # Load and print OpenIE results
    print("\nLoading OpenIE results...")
    openie_results = load_openie_results()
    print_entities_and_triples(openie_results)

    # Visualization options
    print("\n" + "="*60)
    print("  VISUALIZATION OPTIONS")
    print("="*60)

    print("\n1. Attempting igraph visualization...")
    igraph_success = visualize_with_igraph(graph, 'outputs/knowledge_graph_igraph.png')

    print("\n2. Attempting NetworkX visualization...")
    nx_success = visualize_with_networkx(graph, 'outputs/knowledge_graph_networkx.png')

    print("\n3. Exporting to Gephi format...")
    gephi_success = export_to_gephi(graph, 'outputs/knowledge_graph.gexf')

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"\nKnowledge graph location: outputs/gpt-4o_text-embedding-3-large/graph.pickle")
    print(f"OpenIE results location: outputs/openie_results_ner_gpt-4o.json")

    if igraph_success or nx_success:
        print("\nVisualization images saved in outputs/ folder")

    if gephi_success:
        print("Gephi file ready for interactive exploration")

if __name__ == "__main__":
    main()
