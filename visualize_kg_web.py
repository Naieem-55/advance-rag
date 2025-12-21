"""
HippoRAG Knowledge Graph - Interactive Web Visualization
Opens an interactive graph visualization in your browser
"""

import pickle
import json
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

def load_graph(save_dir='outputs', llm_model='gpt-4o', embedding_model='text-embedding-3-large'):
    """Load the igraph object from pickle file."""
    graph_path = os.path.join(save_dir, f"{llm_model}_{embedding_model}", "graph.pickle")

    if not os.path.exists(graph_path):
        print(f"Graph not found at: {graph_path}")
        return None

    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    return graph

def load_openie_results(save_dir='outputs', llm_model='gpt-4o'):
    """Load OpenIE extraction results."""
    openie_path = os.path.join(save_dir, f"openie_results_ner_{llm_model}.json")

    if not os.path.exists(openie_path):
        return None

    with open(openie_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results

def create_pyvis_visualization(graph, output_file='outputs/knowledge_graph_interactive.html'):
    """Create interactive visualization using Pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("Pyvis not installed. Installing now...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyvis'])
        from pyvis.network import Network

    # Create Pyvis network
    net = Network(
        height='900px',
        width='100%',
        bgcolor='#222222',
        font_color='white',
        directed=True,
        notebook=False
    )

    # Physics settings for better layout
    net.set_options('''
    {
        "nodes": {
            "font": {
                "size": 14,
                "face": "Arial"
            },
            "scaling": {
                "min": 10,
                "max": 30
            }
        },
        "edges": {
            "color": {
                "inherit": true
            },
            "smooth": {
                "type": "continuous"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            }
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 1000
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": true,
            "navigationButtons": true,
            "keyboard": {
                "enabled": true
            }
        }
    }
    ''')

    # Color scheme for different node types (based on hash_id prefix)
    color_map = {
        'entity': '#4ECDC4',      # Teal - entities/phrases
        'chunk': '#FF6B6B',       # Coral - passages/document chunks
        'passage': '#FF6B6B',     # Coral
        'document': '#FF6B6B',    # Coral
        'default': '#95E1D3'      # Light green
    }

    # Add nodes
    for v in graph.vs:
        node_id = v.index

        # Get node content/label
        content = str(v['content']) if 'content' in graph.vs.attributes() else ''
        name = v['name'] if 'name' in graph.vs.attributes() else str(node_id)
        hash_id = v['hash_id'] if 'hash_id' in graph.vs.attributes() else ''

        # Determine node type from hash_id prefix
        node_type = hash_id.split('-')[0] if '-' in hash_id else 'default'

        # Create label (truncate if too long)
        if content and len(content) > 0:
            label = content[:30] + '...' if len(content) > 30 else content
        else:
            label = name[:30] + '...' if len(name) > 30 else name

        # Skip empty nodes - show type instead
        if not label or label.strip() == '':
            label = f"{node_type}_{node_id}"

        # Node color based on type
        color = color_map.get(node_type, color_map['default'])

        # Node size: chunks are larger, entities sized by connections
        degree = graph.degree(node_id)
        if node_type == 'chunk':
            size = 40  # Larger size for passage/chunk nodes
        else:
            size = 15 + min(degree * 3, 30)

        # Hover title with full info
        title = f"<b>{label}</b><br>"
        title += f"Type: {node_type}<br>"
        title += f"Connections: {degree}<br>"
        if content and len(content) > 30:
            title += f"<br>Full content:<br>{content[:200]}..."

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=2,
            borderWidthSelected=4
        )

    # Add edges
    for e in graph.es:
        source = e.source
        target = e.target

        # Edge label/title
        edge_type = e['type'] if 'type' in graph.es.attributes() else ''
        weight = e['weight'] if 'weight' in graph.es.attributes() else 1.0

        title = f"Type: {edge_type}<br>Weight: {weight:.4f}" if edge_type else f"Weight: {weight:.4f}"

        net.add_edge(
            source,
            target,
            title=title,
            width=1 + float(weight) * 2 if weight else 1,
            color='#888888'
        )

    # Save to HTML
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    net.save_graph(output_file)

    # Add custom title and legend to HTML
    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    legend_html = '''
    <div style="position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; color: white; font-family: Arial; z-index: 1000;">
        <h3 style="margin: 0 0 10px 0;">HippoRAG Knowledge Graph</h3>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #4ECDC4; border-radius: 50%; margin-right: 10px;"></div>
            <span>Entity Nodes (51)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #FF6B6B; border-radius: 50%; margin-right: 10px;"></div>
            <span>Chunk/Passage Nodes (9)</span>
        </div>
        <hr style="border-color: #444; margin: 10px 0;">
        <small>
            <b>Controls:</b><br>
            - Scroll: Zoom<br>
            - Drag: Pan/Move nodes<br>
            - Click: Select node<br>
            - Hover: View details
        </small>
    </div>
    '''

    # Insert legend after body tag
    html_content = html_content.replace('<body>', f'<body>{legend_html}')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_file

def create_triples_visualization(openie_results, output_file='outputs/triples_graph.html'):
    """Create a separate visualization for extracted triples."""
    try:
        from pyvis.network import Network
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyvis'])
        from pyvis.network import Network

    if not openie_results:
        return None

    docs_list = openie_results.get('docs', []) if isinstance(openie_results, dict) else openie_results

    net = Network(
        height='900px',
        width='100%',
        bgcolor='#1a1a2e',
        font_color='white',
        directed=True,
        notebook=False
    )

    net.set_options('''
    {
        "nodes": {
            "font": {"size": 16, "face": "Arial"},
            "shape": "dot"
        },
        "edges": {
            "font": {"size": 12, "face": "Arial", "color": "#aaaaaa"},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.005,
                "springLength": 150,
                "springConstant": 0.05
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 500}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true
        }
    }
    ''')

    # Track unique entities
    entities = {}
    entity_id = 0

    # Process all documents
    for doc in docs_list:
        triples = doc.get('extracted_triples', [])

        for triple in triples:
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                subj, rel, obj = triple[0], triple[1], triple[2]
            elif isinstance(triple, dict):
                subj = triple.get('subject', triple.get('head', ''))
                rel = triple.get('relation', triple.get('predicate', ''))
                obj = triple.get('object', triple.get('tail', ''))
            else:
                continue

            # Skip empty values
            if not subj or not obj:
                continue

            # Add subject node
            if subj not in entities:
                entities[subj] = entity_id
                net.add_node(
                    entity_id,
                    label=subj[:25] + '...' if len(subj) > 25 else subj,
                    title=f"<b>{subj}</b>",
                    color='#e94560',
                    size=20
                )
                entity_id += 1

            # Add object node
            if obj not in entities:
                entities[obj] = entity_id
                net.add_node(
                    entity_id,
                    label=obj[:25] + '...' if len(obj) > 25 else obj,
                    title=f"<b>{obj}</b>",
                    color='#0f3460',
                    size=20
                )
                entity_id += 1

            # Add edge with relation label
            net.add_edge(
                entities[subj],
                entities[obj],
                label=rel[:20] + '...' if len(rel) > 20 else rel,
                title=f"Relation: {rel}",
                color='#16c79a',
                width=2
            )

    # Save
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    net.save_graph(output_file)

    # Add legend
    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    legend_html = '''
    <div style="position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.9); padding: 15px; border-radius: 8px; color: white; font-family: Arial; z-index: 1000;">
        <h3 style="margin: 0 0 10px 0;">Extracted Triples (Subject-Relation-Object)</h3>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #e94560; border-radius: 50%; margin-right: 10px;"></div>
            <span>Subject Entities</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #0f3460; border-radius: 50%; margin-right: 10px;"></div>
            <span>Object Entities</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 30px; height: 3px; background: #16c79a; margin-right: 10px;"></div>
            <span>Relations</span>
        </div>
    </div>
    '''

    html_content = html_content.replace('<body>', f'<body>{legend_html}')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_file

def start_server_and_open(html_file, port=8888):
    """Start a simple HTTP server and open the visualization in browser."""
    import functools

    # Get absolute path
    abs_html_file = os.path.abspath(html_file)
    html_dir = os.path.dirname(abs_html_file)
    html_filename = os.path.basename(abs_html_file)

    print(f"\nServing from directory: {html_dir}")
    print(f"HTML file: {html_filename}")

    # Create a custom handler that serves from the correct directory
    handler = functools.partial(SimpleHTTPRequestHandler, directory=html_dir)

    # Create server
    httpd = HTTPServer(('localhost', port), handler)

    # Start server in background thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    url = f'http://localhost:{port}/{html_filename}'
    print(f"\nServer started at: {url}")
    print("Press Ctrl+C to stop the server\n")

    # Open in browser
    webbrowser.open(url)

    return httpd

def main():
    print("="*60)
    print("  HippoRAG Knowledge Graph - Web Visualization")
    print("="*60)

    # Load graph
    print("\nLoading knowledge graph...")
    graph = load_graph()

    if graph is None:
        print("\nNo graph found. Please run test_hipporag.py first.")
        return

    print(f"Graph loaded: {graph.vcount()} nodes, {graph.ecount()} edges")

    # Load OpenIE results
    print("\nLoading OpenIE results...")
    openie_results = load_openie_results()

    # Create main graph visualization
    print("\nCreating interactive knowledge graph...")
    kg_file = create_pyvis_visualization(graph, 'outputs/knowledge_graph_interactive.html')
    print(f"Created: {kg_file}")

    # Create triples visualization
    if openie_results:
        print("\nCreating triples visualization...")
        triples_file = create_triples_visualization(openie_results, 'outputs/triples_graph.html')
        print(f"Created: {triples_file}")

    # Start server and open in browser
    print("\n" + "="*60)
    print("  Starting web server...")
    print("="*60)

    try:
        httpd = start_server_and_open(kg_file, port=8888)

        print("\nVisualization files created:")
        print(f"  1. Knowledge Graph: outputs/knowledge_graph_interactive.html")
        if openie_results:
            print(f"  2. Triples Graph:   outputs/triples_graph.html")

        print(f"\nOpen in browser:")
        print(f"  - http://localhost:8888/knowledge_graph_interactive.html")
        if openie_results:
            print(f"  - http://localhost:8888/triples_graph.html")

        print("\nPress Ctrl+C to stop the server...")

        # Keep server running
        while True:
            pass

    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except Exception as e:
        print(f"\nError starting server: {e}")
        print(f"\nYou can still open the HTML files directly:")
        print(f"  - {os.path.abspath(kg_file)}")

if __name__ == "__main__":
    main()
