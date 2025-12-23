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

def load_graph(save_dir='outputs', llm_model=None, embedding_model=None):
    """Load the igraph object from pickle file. Auto-detects model directories if not specified."""

    # Auto-detect model directory if not specified
    if llm_model is None or embedding_model is None:
        if os.path.exists(save_dir):
            for item in os.listdir(save_dir):
                item_path = os.path.join(save_dir, item)
                if os.path.isdir(item_path):
                    graph_file = os.path.join(item_path, "graph.pickle")
                    if os.path.exists(graph_file):
                        print(f"Auto-detected graph at: {graph_file}")
                        with open(graph_file, 'rb') as f:
                            return pickle.load(f)
        print(f"No graph found in {save_dir}")
        return None

    graph_path = os.path.join(save_dir, f"{llm_model}_{embedding_model}", "graph.pickle")

    if not os.path.exists(graph_path):
        print(f"Graph not found at: {graph_path}")
        return None

    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    return graph

def load_openie_results(save_dir='outputs', llm_model=None):
    """Load OpenIE extraction results. Auto-detects if llm_model not specified."""

    # Auto-detect openie file if not specified
    if llm_model is None:
        if os.path.exists(save_dir):
            for item in os.listdir(save_dir):
                if item.startswith('openie_results_ner_') and item.endswith('.json'):
                    openie_path = os.path.join(save_dir, item)
                    print(f"Auto-detected OpenIE results at: {openie_path}")
                    with open(openie_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
        return None

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

    # Create Pyvis network with professional styling
    net = Network(
        height='100vh',
        width='100vw',
        bgcolor='#0f1419',
        font_color='#e7e9ea',
        directed=True,
        notebook=False
    )

    # Professional physics settings
    net.set_options('''
    {
        "nodes": {
            "font": {
                "size": 12,
                "face": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "color": "#e7e9ea"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4
        },
        "edges": {
            "color": {"color": "#38444d", "highlight": "#1d9bf0"},
            "smooth": {"type": "continuous", "roundness": 0.5},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.4}}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.01,
                "springLength": 120,
                "springConstant": 0.08,
                "avoidOverlap": 0.5
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"enabled": true, "iterations": 200, "fit": true}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 50,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true,
            "zoomView": true,
            "dragView": true,
            "keyboard": {"enabled": true}
        }
    }
    ''')

    # Professional color scheme for different node types
    color_map = {
        'entity': '#1d9bf0',      # Blue - entities/phrases
        'chunk': '#f91880',       # Pink - passages/document chunks
        'passage': '#f91880',     # Pink
        'document': '#f91880',    # Pink
        'default': '#00ba7c'      # Green
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

    # Count node types
    has_hash_id = 'hash_id' in graph.vs.attributes()
    entity_count = sum(1 for v in graph.vs if has_hash_id and v['hash_id'].startswith('entity'))
    chunk_count = sum(1 for v in graph.vs if has_hash_id and v['hash_id'].startswith('chunk'))

    # Add custom title and legend to HTML
    with open(output_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    legend_html = f'''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; box-sizing: border-box; }}
        html, body {{ margin: 0; padding: 0; width: 100vw; height: 100vh; overflow: hidden; background: #0f1419; }}
        .card {{ width: 100vw !important; height: 100vh !important; margin: 0 !important; padding: 0 !important; border: none !important; }}
        #mynetwork {{ width: 100vw !important; height: 100vh !important; position: fixed !important; top: 0 !important; left: 0 !important; border: none !important; }}
        #loadingBar {{ display: none !important; }}

        .kg-panel {{
            position: fixed;
            background: linear-gradient(145deg, rgba(22, 32, 42, 0.98), rgba(15, 20, 25, 0.98));
            border: 1px solid #38444d;
            border-radius: 16px;
            color: #e7e9ea;
            z-index: 1000;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            padding: 20px;
        }}
        .kg-panel h2 {{
            margin: 0 0 4px 0;
            font-size: 18px;
            font-weight: 700;
            color: #e7e9ea;
        }}
        .kg-panel .subtitle {{
            font-size: 12px;
            color: #71767b;
            margin-bottom: 16px;
        }}
        .kg-panel hr {{
            border: none;
            border-top: 1px solid #38444d;
            margin: 16px 0;
        }}
        .stat-row {{
            display: flex;
            gap: 12px;
            margin: 16px 0;
        }}
        .stat-box {{
            flex: 1;
            background: rgba(29, 155, 240, 0.1);
            border: 1px solid rgba(29, 155, 240, 0.2);
            border-radius: 12px;
            padding: 12px;
            text-align: center;
        }}
        .stat-box.entity {{ border-color: rgba(29, 155, 240, 0.3); }}
        .stat-box.passage {{ border-color: rgba(249, 24, 128, 0.3); background: rgba(249, 24, 128, 0.1); }}
        .stat-box.edge {{ border-color: rgba(0, 186, 124, 0.3); background: rgba(0, 186, 124, 0.1); }}
        .stat-number {{
            font-size: 28px;
            font-weight: 700;
        }}
        .stat-box.entity .stat-number {{ color: #1d9bf0; }}
        .stat-box.passage .stat-number {{ color: #f91880; }}
        .stat-box.edge .stat-number {{ color: #00ba7c; }}
        .stat-label {{
            font-size: 10px;
            color: #71767b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }}
        .legend-section {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #71767b;
            margin-bottom: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 0;
            font-size: 13px;
        }}
        .legend-dot {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .legend-box {{
            width: 16px;
            height: 12px;
            border-radius: 4px;
            flex-shrink: 0;
        }}
        .controls {{
            font-size: 12px;
            color: #71767b;
            line-height: 1.8;
        }}
        .controls kbd {{
            background: #253341;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            color: #e7e9ea;
        }}
    </style>

    <div class="kg-panel" style="top: 20px; left: 20px; width: 280px;">
        <h2>🧠 HippoRAG</h2>
        <div class="subtitle">Knowledge Graph Explorer</div>

        <div class="stat-row">
            <div class="stat-box entity">
                <div class="stat-number">{entity_count}</div>
                <div class="stat-label">Entities</div>
            </div>
            <div class="stat-box passage">
                <div class="stat-number">{chunk_count}</div>
                <div class="stat-label">Passages</div>
            </div>
            <div class="stat-box edge">
                <div class="stat-number">{graph.ecount()}</div>
                <div class="stat-label">Edges</div>
            </div>
        </div>

        <hr>
        <div class="legend-section">Node Types</div>
        <div class="legend-item">
            <div class="legend-dot" style="background: #1d9bf0;"></div>
            <span>Entity Nodes</span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background: #f91880;"></div>
            <span>Passage/Chunk Nodes</span>
        </div>

        <hr>
        <div class="legend-section">Controls</div>
        <div class="controls">
            <kbd>Scroll</kbd> Zoom in/out<br>
            <kbd>Drag</kbd> Pan view / Move nodes<br>
            <kbd>Click</kbd> Select node<br>
            <kbd>Hover</kbd> View details
        </div>
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
